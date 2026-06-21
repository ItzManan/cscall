from __future__ import annotations

from collections.abc import Callable
import json
import logging
import os
from email import policy
from email.parser import BytesParser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time
import tempfile
from urllib.parse import urlsplit
import wave

from cscall.fusion import fuse_words, group_speaker_words
from cscall.streaming.audio import validate_pcm_wav


LOGGER = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
_SERVICE_UNAVAILABLE_MESSAGE = (
    "The transcription service is temporarily unavailable."
)
_INTERNAL_SERVER_ERROR_MESSAGE = "Internal server error"
_INVALID_WAV_MESSAGE = "Invalid WAV upload"
_PLACEHOLDER_HTML = "<!doctype html><html><body><h1>Upload transcription API</h1></body></html>"


class RequestError(ValueError):
    def __init__(self, message: str, status_code: int = HTTPStatus.BAD_REQUEST):
        super().__init__(message)
        self.status_code = status_code


def _default_transcriber_factory():
    from cscall.asr_baseline import WhisperTranscriber

    return WhisperTranscriber()


def _default_diarizer_factory():
    from cscall.diarization import PyannoteDiarizer

    return PyannoteDiarizer()


class SpeakerTranscriptionService:
    def __init__(
        self,
        *,
        transcriber=None,
        diarizer=None,
        transcriber_factory: Callable[[], object] | None = None,
        diarizer_factory: Callable[[], object] | None = None,
        clock: Callable[[], float] | None = None,
        lock=None,
    ):
        self._transcriber = transcriber
        self._diarizer = diarizer
        self._transcriber_factory = (
            transcriber_factory
            if transcriber_factory is not None
            else _default_transcriber_factory
        )
        self._diarizer_factory = (
            diarizer_factory if diarizer_factory is not None else _default_diarizer_factory
        )
        self._clock = clock if clock is not None else time.perf_counter
        self._lock = lock if lock is not None else threading.Lock()

    def transcribe_wav(self, path):
        path_str = str(path)
        started = self._clock()
        validate_pcm_wav(path_str)
        audio_seconds = self._audio_seconds(path_str)

        with self._lock:
            turns = self._get_diarizer().diarize(path_str)
            words = self._get_transcriber().transcribe_words(path_str)

        segments = []
        for group in group_speaker_words(fuse_words(words, turns)):
            segments.append(
                {
                    "start": group[0].start,
                    "end": group[-1].end,
                    "speaker": group[0].speaker,
                    "text": " ".join(word.text for word in group),
                }
            )

        finished = self._clock()
        processing_ms = int(round((finished - started) * 1000))
        rtf = 0.0 if audio_seconds == 0 else (processing_ms / 1000) / audio_seconds

        return {
            "segments": segments,
            "processing_ms": processing_ms,
            "audio_seconds": audio_seconds,
            "rtf": rtf,
        }

    def _get_transcriber(self):
        if self._transcriber is None:
            self._transcriber = self._transcriber_factory()
        return self._transcriber

    def _get_diarizer(self):
        if self._diarizer is None:
            self._diarizer = self._diarizer_factory()
        return self._diarizer

    def _audio_seconds(self, path_str: str) -> float:
        with wave.open(path_str, "rb") as wav:
            return wav.getnframes() / wav.getframerate()


def _json_bytes(payload: object) -> bytes:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _response_bytes(
    handler: BaseHTTPRequestHandler, status: HTTPStatus, content_type: str, body: bytes
) -> None:
    try:
        handler.send_response(status)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        if body:
            handler.wfile.write(body)
    except (BrokenPipeError, ConnectionResetError):
        return


def _send_json(
    handler: BaseHTTPRequestHandler, status: HTTPStatus, payload: object
) -> None:
    _response_bytes(handler, status, "application/json; charset=utf-8", _json_bytes(payload))


def _send_html(handler: BaseHTTPRequestHandler, status: HTTPStatus, body: bytes) -> None:
    _response_bytes(handler, status, "text/html; charset=utf-8", body)


def _request_path(handler: BaseHTTPRequestHandler) -> str:
    return urlsplit(handler.path).path


def _parse_content_length(headers) -> int:
    raw_value = headers.get("Content-Length")
    if raw_value is None:
        raise RequestError("missing Content-Length")
    try:
        content_length = int(raw_value)
    except (TypeError, ValueError):
        raise RequestError("invalid Content-Length")
    if content_length < 0:
        raise RequestError("invalid Content-Length")
    if content_length > MAX_UPLOAD_BYTES:
        raise RequestError("Content-Length exceeds upload limit")
    return content_length


def _read_request_body(rfile, headers) -> bytes:
    content_length = _parse_content_length(headers)
    body = rfile.read(content_length)
    if body is None:
        body = b""
    if len(body) != content_length:
        raise RequestError("request body was truncated")
    return body


def _parse_multipart_audio_upload(body: bytes, content_type: str) -> bytes:
    if "multipart/form-data" not in content_type.lower():
        raise RequestError("multipart/form-data uploads only")

    synthetic_message = (
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
        + body
    )
    message = BytesParser(policy=policy.default).parsebytes(synthetic_message)

    if message.get_content_type() != "multipart/form-data":
        raise RequestError("multipart/form-data uploads only")

    if not message.get_boundary():
        raise RequestError("missing multipart boundary")

    boundary = message.get_boundary().encode("utf-8")
    closing_boundary = b"--" + boundary + b"--"
    if not body.endswith(closing_boundary + b"\r\n") and not body.endswith(closing_boundary):
        raise RequestError("malformed multipart/form-data upload")

    if message.defects:
        raise RequestError("malformed multipart/form-data upload")

    if not message.is_multipart():
        raise RequestError("malformed multipart/form-data upload")

    parts = list(message.iter_parts())
    if any(part.defects for part in parts):
        raise RequestError("malformed multipart/form-data upload")
    if len(parts) != 1:
        raise RequestError("multipart upload must contain exactly one part")

    part = parts[0]
    if part.get_content_disposition() != "form-data":
        raise RequestError("multipart upload is malformed")

    field_name = part.get_param("name", header="content-disposition")
    if field_name != "audio":
        raise RequestError("multipart upload must contain an audio field")

    filename = part.get_filename()
    if not filename or not filename.lower().endswith(".wav"):
        raise RequestError("audio filename must end with .wav")

    audio_bytes = part.get_payload(decode=True) or b""
    if not audio_bytes:
        raise RequestError("audio upload must not be empty")

    return audio_bytes


def _is_model_setup_runtime_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "hf_token",
            "pyannote.audio is required",
            "accept the model conditions",
        )
    )


def _transcribe_uploaded_wav(service, audio_bytes: bytes) -> object:
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            temp_path = tmp.name
        try:
            validate_pcm_wav(temp_path)
        except ValueError as exc:
            raise RequestError(_INVALID_WAV_MESSAGE) from exc
        return service.transcribe_wav(temp_path)
    finally:
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass


def make_handler(service, html: str | bytes = _PLACEHOLDER_HTML):
    html_bytes = html.encode("utf-8") if isinstance(html, str) else bytes(html)

    class UploadHTTPRequestHandler(BaseHTTPRequestHandler):
        service = None
        html_bytes = b""
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):  # noqa: A003 - stdlib signature
            return

        def do_GET(self):
            try:
                path = _request_path(self)
                if path == "/health":
                    _send_json(self, HTTPStatus.OK, {"status": "ok"})
                    return
                if path == "/":
                    _send_html(self, HTTPStatus.OK, self.html_bytes)
                    return
                _send_json(self, HTTPStatus.NOT_FOUND, {"error": "not found"})
            except (BrokenPipeError, ConnectionResetError):
                return

        def do_POST(self):
            try:
                path = _request_path(self)
                if path != "/api/transcribe":
                    _send_json(self, HTTPStatus.NOT_FOUND, {"error": "not found"})
                    return

                body = _read_request_body(self.rfile, self.headers)
                content_type = self.headers.get("Content-Type")
                if content_type is None:
                    raise RequestError("missing Content-Type")
                audio_bytes = _parse_multipart_audio_upload(body, content_type)
                result = _transcribe_uploaded_wav(self.service, audio_bytes)
                _send_json(self, HTTPStatus.OK, result)
            except RequestError as exc:
                _send_json(self, HTTPStatus(exc.status_code), {"error": str(exc)})
            except RuntimeError as exc:
                if _is_model_setup_runtime_error(exc):
                    _send_json(
                        self,
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        {"error": _SERVICE_UNAVAILABLE_MESSAGE},
                    )
                else:
                    LOGGER.exception("Unexpected RuntimeError while handling upload")
                    _send_json(
                        self,
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        {"error": _INTERNAL_SERVER_ERROR_MESSAGE},
                    )
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception:
                LOGGER.exception("Unexpected exception while handling upload")
                _send_json(
                    self,
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": _INTERNAL_SERVER_ERROR_MESSAGE},
                )

    UploadHTTPRequestHandler.service = service
    UploadHTTPRequestHandler.html_bytes = html_bytes
    return UploadHTTPRequestHandler


def create_server(host: str, port: int, service) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(service))
    server.daemon_threads = True
    return server
