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
FULL_UPLOAD_UI_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Upload transcript</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f8fafc;
      --surface: #ffffff;
      --surface-alt: #f1f5f9;
      --text: #0f172a;
      --muted: #475569;
      --border: #cbd5e1;
      --accent: #1d4ed8;
      --accent-strong: #1e3a8a;
      --success: #166534;
      --speaker-a: #dbeafe;
      --speaker-b: #ffedd5;
      --speaker-other: #e2e8f0;
      --shadow: 0 24px 60px rgba(15, 23, 42, 0.08);
    }

    * {
      box-sizing: border-box;
    }

    html {
      background: var(--bg);
      color: var(--text);
    }

    body {
      margin: 0;
      min-height: 100vh;
      font: 16px/1.5 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top, rgba(29, 78, 216, 0.1), transparent 48%),
        var(--bg);
    }

    main {
      width: min(100%, 64rem);
      margin: 0 auto;
      padding: clamp(1rem, 3vw, 2rem);
    }

    header,
    form,
    section {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 1rem;
      box-shadow: var(--shadow);
    }

    header {
      padding: 1.25rem 1.5rem;
      margin-bottom: 1rem;
    }

    h1,
    h2,
    p {
      margin: 0;
    }

    h1 {
      font-size: clamp(1.75rem, 3vw, 2.5rem);
      line-height: 1.1;
      letter-spacing: -0.03em;
    }

    .lede,
    .setup-note,
    .status {
      color: var(--muted);
    }

    .lede {
      margin-top: 0.5rem;
      max-width: 58ch;
    }

    form {
      display: grid;
      gap: 1rem;
      padding: 1.25rem 1.5rem 1.5rem;
      margin-bottom: 1rem;
    }

    .field {
      display: grid;
      gap: 0.5rem;
    }

    label {
      font-weight: 650;
    }

    input[type="file"] {
      width: 100%;
      padding: 0.75rem;
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      background: var(--surface-alt);
      color: var(--text);
    }

    button {
      justify-self: start;
      border: 0;
      border-radius: 999px;
      padding: 0.8rem 1.25rem;
      background: var(--accent);
      color: #ffffff;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
      transition:
        transform 120ms ease,
        background-color 120ms ease,
        opacity 120ms ease;
    }

    button:hover,
    button:focus-visible {
      background: var(--accent-strong);
    }

    button:active {
      transform: translateY(1px);
    }

    button:disabled,
    input[type="file"]:disabled {
      opacity: 0.65;
      cursor: progress;
    }

    :focus-visible {
      outline: 3px solid #0f766e;
      outline-offset: 3px;
    }

    .status {
      min-height: 1.5rem;
      font-weight: 600;
    }

    .setup-note {
      font-size: 0.95rem;
    }

    section {
      padding: 1.25rem 1.5rem 1.5rem;
    }

    .results-header {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 1rem;
      margin-bottom: 1rem;
    }

    .metrics {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(11rem, 1fr));
      gap: 0.75rem;
      margin: 0 0 1rem;
    }

    .metric {
      margin: 0;
      padding: 0.85rem 1rem;
      border-radius: 0.85rem;
      background: var(--surface-alt);
      border: 1px solid var(--border);
    }

    .metric dt {
      font-size: 0.78rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }

    .metric dd {
      margin: 0.35rem 0 0;
      font-size: 1.2rem;
      font-weight: 700;
      color: var(--text);
      word-break: break-word;
    }

    .transcript-list {
      list-style: none;
      display: grid;
      gap: 0.75rem;
      padding: 0;
      margin: 0;
    }

    .transcript-item {
      display: grid;
      gap: 0.25rem;
      padding: 1rem 1rem 1rem 1.15rem;
      border-radius: 0.95rem;
      border: 1px solid var(--border);
      border-left-width: 0.45rem;
      background: var(--surface-alt);
    }

    .speaker-a {
      border-left-color: #2563eb;
      background: var(--speaker-a);
    }

    .speaker-b {
      border-left-color: #d97706;
      background: var(--speaker-b);
    }

    .speaker-other {
      border-left-color: #64748b;
      background: var(--speaker-other);
    }

    .speaker-label {
      font-size: 0.85rem;
      font-weight: 800;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }

    .timestamp {
      font-size: 0.92rem;
      color: var(--muted);
    }

    .transcript-text {
      margin: 0;
      white-space: pre-wrap;
    }

    .empty-state {
      padding: 1rem;
      border-radius: 0.95rem;
      border: 1px dashed var(--border);
      color: var(--muted);
      background: var(--surface-alt);
    }

    [hidden] {
      display: none !important;
    }

    @media (max-width: 40rem) {
      main {
        padding: 0.75rem;
      }

      header,
      form,
      section {
        border-radius: 0.9rem;
      }

      header,
      form,
      section {
        padding-inline: 1rem;
      }

      button {
        width: 100%;
      }
    }

    @media (prefers-reduced-motion: reduce) {
      *,
      *::before,
      *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
        scroll-behavior: auto !important;
      }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Upload transcript</h1>
      <p class="lede">Upload a WAV file to transcribe locally and view speaker-attributed results in the browser.</p>
    </header>

    <form id="upload-form" enctype="multipart/form-data" novalidate>
      <div class="field">
        <label for="audio">WAV file</label>
        <input id="audio" name="audio" type="file" accept=".wav,audio/wav">
      </div>
      <button id="submit-button" type="submit">Transcribe</button>
      <p class="setup-note">Set <code>HF_TOKEN</code> in the environment before starting the server so diarization can load.</p>
      <p id="status" class="status" aria-live="polite" role="status"></p>
    </form>

    <section aria-labelledby="results-heading">
      <div class="results-header">
        <h2 id="results-heading">Results</h2>
      </div>
      <dl id="metrics" class="metrics" hidden>
        <div class="metric">
          <dt>Processing</dt>
          <dd id="processing-ms"></dd>
        </div>
        <div class="metric">
          <dt>Audio</dt>
          <dd id="audio-seconds"></dd>
        </div>
        <div class="metric">
          <dt>RTF</dt>
          <dd id="rtf"></dd>
        </div>
      </dl>
      <ol id="transcript-list" class="transcript-list" hidden></ol>
    </section>
  </main>

  <script>
    (() => {
      const form = document.getElementById("upload-form");
      const fileInput = document.getElementById("audio");
      const submitButton = document.getElementById("submit-button");
      const status = document.getElementById("status");
      const metrics = document.getElementById("metrics");
      const transcriptList = document.getElementById("transcript-list");

      function setStatus(message) {
        status.textContent = message;
      }

      function formatTimestamp(seconds) {
        const numeric = Number(seconds);
        const safeSeconds = Number.isFinite(numeric) && numeric > 0 ? numeric : 0;
        const minutes = Math.floor(safeSeconds / 60);
        const remainder = safeSeconds - (minutes * 60);
        return String(minutes).padStart(2, "0") + ":" + remainder.toFixed(2).padStart(5, "0");
      }

      function formatProcessingMs(value) {
        const numeric = Number(value);
        return Number.isFinite(numeric) ? `${Math.max(0, Math.round(numeric))} ms` : "n/a";
      }

      function formatRtf(value) {
        const numeric = Number(value);
        return Number.isFinite(numeric) ? numeric.toFixed(3) : "n/a";
      }

      function speakerClassFactory() {
        const labels = new Map();
        return (speaker) => {
          const label = String(speaker || "").trim();
          if (!label) {
            return "speaker-other";
          }
          if (labels.has(label)) {
            return labels.get(label);
          }
          const assigned =
            labels.size === 0 ? "speaker-a" : labels.size === 1 ? "speaker-b" : "speaker-other";
          labels.set(label, assigned);
          return assigned;
        };
      }

      function renderMetrics(data) {
        const fragment = document.createDocumentFragment();
        const items = [
          ["Processing", formatProcessingMs(data.processing_ms)],
          ["Audio", formatTimestamp(data.audio_seconds)],
          ["RTF", formatRtf(data.rtf)],
        ];

        for (const [label, value] of items) {
          const wrap = document.createElement("div");
          wrap.className = "metric";
          const dt = document.createElement("dt");
          dt.textContent = label;
          const dd = document.createElement("dd");
          dd.textContent = value;
          wrap.append(dt, dd);
          fragment.append(wrap);
        }

        metrics.replaceChildren(fragment);
        metrics.hidden = false;
      }

      function renderTranscript(segments) {
        const speakerClass = speakerClassFactory();
        const fragment = document.createDocumentFragment();

        if (!Array.isArray(segments) || segments.length === 0) {
          const empty = document.createElement("li");
          empty.className = "empty-state";
          empty.textContent = "No transcript segments were returned.";
          fragment.append(empty);
          transcriptList.replaceChildren(fragment);
          transcriptList.hidden = false;
          return;
        }

        for (const segment of segments) {
          const item = document.createElement("li");
          item.className = `transcript-item ${speakerClass(segment && segment.speaker)}`;

          const speaker = document.createElement("div");
          speaker.className = "speaker-label";
          speaker.textContent = segment && segment.speaker ? String(segment.speaker) : "Unknown speaker";

          const timestamp = document.createElement("div");
          timestamp.className = "timestamp";
          timestamp.textContent = `${formatTimestamp(segment && segment.start)} — ${formatTimestamp(segment && segment.end)}`;

          const text = document.createElement("p");
          text.className = "transcript-text";
          text.textContent = segment && segment.text ? String(segment.text) : "";

          item.append(speaker, timestamp, text);
          fragment.append(item);
        }

        transcriptList.replaceChildren(fragment);
        transcriptList.hidden = false;
      }

      async function handleSubmit(event) {
        event.preventDefault();

        const file = fileInput.files && fileInput.files[0];
        if (!file) {
          setStatus("Choose a WAV file before transcribing.");
          fileInput.focus();
          return;
        }

        const isWav = file.type === "audio/wav" || /\.wav$/i.test(file.name || "");
        if (!isWav) {
          setStatus("Please choose a WAV file.");
          fileInput.focus();
          return;
        }

        const originalButtonText = submitButton.textContent;
        submitButton.disabled = true;
        fileInput.disabled = true;
        setStatus("Transcribing…");

        try {
          const formData = new FormData();
          formData.append("audio", file, file.name || "upload.wav");

          const response = await fetch("/api/transcribe", {
            method: "POST",
            body: formData,
          });

          const rawText = await response.text();
          let payload = null;
          if (rawText) {
            try {
              payload = JSON.parse(rawText);
            } catch {
              payload = null;
            }
          }

          if (!response.ok) {
            const message =
              payload && typeof payload.error === "string"
                ? payload.error
                : rawText.trim() || `Request failed with ${response.status}`;
            setStatus(message);
            return;
          }

          if (!payload || typeof payload !== "object") {
            setStatus("Unexpected response from server.");
            return;
          }

          renderMetrics(payload);
          renderTranscript(payload.segments);
          setStatus("Transcription complete.");
        } catch (error) {
          setStatus("Upload failed. Check the server and try again.");
        } finally {
          submitButton.disabled = false;
          fileInput.disabled = false;
          submitButton.textContent = originalButtonText;
        }
      }

      form.addEventListener("submit", handleSubmit);
    })();
  </script>
</body>
</html>
"""

_PLACEHOLDER_HTML = FULL_UPLOAD_UI_HTML


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
        _validate_complete_pcm_wav(temp_path)
        return service.transcribe_wav(temp_path)
    finally:
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass


def _validate_complete_pcm_wav(path) -> object:
    path_str = str(path)
    try:
        wav_info = validate_pcm_wav(path_str)
    except ValueError as exc:
        raise RequestError(_INVALID_WAV_MESSAGE) from exc

    try:
        with wave.open(path_str, "rb") as wav:
            frame_size = wav_info.channels * wav_info.sample_width
            expected_bytes = wav.getnframes() * frame_size
            payload = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError) as exc:
        raise RequestError(_INVALID_WAV_MESSAGE) from exc

    if len(payload) % frame_size != 0 or len(payload) != expected_bytes:
        raise RequestError(_INVALID_WAV_MESSAGE)

    return wav_info


def make_handler(service, html: str | bytes = FULL_UPLOAD_UI_HTML):
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


def create_server(
    host: str, port: int, service, html: str | bytes = FULL_UPLOAD_UI_HTML
) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(service, html))
    server.daemon_threads = True
    return server
