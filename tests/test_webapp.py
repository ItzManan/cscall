from __future__ import annotations

import contextlib
import http.client
import json
import wave
from pathlib import Path
import threading

import pytest

from cscall.fusion import SpeakerTurn, TimedWord
import cscall.webapp as webapp
from cscall.webapp import SpeakerTranscriptionService


def _write_pcm_wav(
    path: Path,
    *,
    frames: int = 8000,
    sample_rate: int = 8000,
    channels: int = 1,
    sample_width: int = 2,
) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00" * frames * channels * sample_width)


def _make_multipart_body(
    parts: list[tuple[str, str, bytes]],
    *,
    boundary: str = "boundary123",
    closing: bool = True,
) -> tuple[bytes, str]:
    chunks: list[bytes] = []
    for name, filename, payload in parts:
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                (
                    'Content-Disposition: form-data; '
                    f'name="{name}"; filename="{filename}"\r\n'
                ).encode(),
                b"Content-Type: audio/wav\r\n\r\n",
                payload,
                b"\r\n",
            ]
        )
    if closing:
        chunks.append(f"--{boundary}--\r\n".encode())
    content_type = f"multipart/form-data; boundary={boundary}"
    return b"".join(chunks), content_type


@contextlib.contextmanager
def _running_http_server(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _http_request(server, method: str, path: str, *, body: bytes = b"", headers=None):
    host, port = server.server_address[:2]
    conn = http.client.HTTPConnection(host, port, timeout=5)
    conn.request(method, path, body=body, headers=headers or {})
    response = conn.getresponse()
    payload = response.read()
    conn.close()
    return response, payload


def _assert_json_response(response, payload: bytes):
    assert response.getheader("Content-Type") == "application/json; charset=utf-8"
    assert response.getheader("Content-Length") == str(len(payload))
    return json.loads(payload.decode())


class RecordingClock:
    def __init__(self, values: list[float]):
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


class RecordingLock:
    def __init__(self, log: list[object]):
        self.log = log
        self.depth = 0

    def __enter__(self):
        self.log.append("lock_enter")
        self.depth += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        self.depth -= 1
        self.log.append("lock_exit")
        return False


def test_transcribe_wav_orders_diarization_before_asr_and_groups_words(
    tmp_path: Path,
):
    audio_path = tmp_path / "sample.wav"
    _write_pcm_wav(audio_path)

    log: list[object] = []
    lock = RecordingLock(log)

    class FakeDiarizer:
        def diarize(self, path: str):
            log.append(("diarize", path, lock.depth))
            assert lock.depth == 1
            return [
                SpeakerTurn(0.0, 0.25, "SPEAKER_00"),
                SpeakerTurn(0.25, 1.0, "SPEAKER_01"),
            ]

    class FakeTranscriber:
        def transcribe_words(self, path: str):
            log.append(("transcribe", path, lock.depth))
            assert lock.depth == 1
            return [
                TimedWord(0.0, 0.1, "hello"),
                TimedWord(0.1, 0.2, "how"),
                TimedWord(0.3, 0.4, "there"),
            ]

    service = SpeakerTranscriptionService(
        transcriber=FakeTranscriber(),
        diarizer=FakeDiarizer(),
        clock=RecordingClock([100.0, 100.842]),
        lock=lock,
    )

    result = service.transcribe_wav(audio_path)

    assert log == [
        "lock_enter",
        ("diarize", str(audio_path), 1),
        ("transcribe", str(audio_path), 1),
        "lock_exit",
    ]
    assert result == {
        "segments": [
            {
                "start": 0.0,
                "end": 0.2,
                "speaker": "SPEAKER_00",
                "text": "hello how",
            },
            {
                "start": 0.3,
                "end": 0.4,
                "speaker": "SPEAKER_01",
                "text": "there",
            },
        ],
        "processing_ms": 842,
        "audio_seconds": pytest.approx(1.0),
        "rtf": pytest.approx(0.842),
    }


def test_transcribe_wav_uses_lazy_factories_once_and_reuses_objects(tmp_path: Path):
    audio_path = tmp_path / "sample.wav"
    _write_pcm_wav(audio_path)

    log: list[object] = []
    lock = RecordingLock(log)
    transcriber_factory_calls = 0
    diarizer_factory_calls = 0

    class FactoryDiarizer:
        def __init__(self):
            self.calls: list[str] = []

        def diarize(self, path: str):
            self.calls.append(path)
            return [SpeakerTurn(0.0, 1.0, "SPEAKER_00")]

    class FactoryTranscriber:
        def __init__(self):
            self.calls: list[str] = []

        def transcribe_words(self, path: str):
            self.calls.append(path)
            return [TimedWord(0.0, 0.5, "hello")]

    diarizer_instance: FactoryDiarizer | None = None
    transcriber_instance: FactoryTranscriber | None = None

    def diarizer_factory():
        nonlocal diarizer_factory_calls, diarizer_instance
        diarizer_factory_calls += 1
        diarizer_instance = FactoryDiarizer()
        return diarizer_instance

    def transcriber_factory():
        nonlocal transcriber_factory_calls, transcriber_instance
        transcriber_factory_calls += 1
        transcriber_instance = FactoryTranscriber()
        return transcriber_instance

    service = SpeakerTranscriptionService(
        transcriber_factory=transcriber_factory,
        diarizer_factory=diarizer_factory,
        clock=RecordingClock([10.0, 10.5, 20.0, 20.5]),
        lock=lock,
    )

    first = service.transcribe_wav(audio_path)
    second = service.transcribe_wav(audio_path)

    assert diarizer_factory_calls == 1
    assert transcriber_factory_calls == 1
    assert diarizer_instance is not None
    assert transcriber_instance is not None
    assert diarizer_instance.calls == [str(audio_path), str(audio_path)]
    assert transcriber_instance.calls == [str(audio_path), str(audio_path)]
    assert first == second


def test_transcribe_wav_rejects_non_pcm_wav_before_model_work(
    tmp_path: Path,
):
    audio_path = tmp_path / "bad.wav"
    audio_path.write_text("not a wav")

    lock_log: list[object] = []
    lock = RecordingLock(lock_log)

    def transcriber_factory():
        raise AssertionError("transcriber factory should not be called")

    def diarizer_factory():
        raise AssertionError("diarizer factory should not be called")

    service = SpeakerTranscriptionService(
        transcriber_factory=transcriber_factory,
        diarizer_factory=diarizer_factory,
        clock=RecordingClock([0.0, 0.1]),
        lock=lock,
    )

    with pytest.raises(ValueError, match="supported PCM WAV"):
        service.transcribe_wav(audio_path)

    assert lock_log == []


def test_transcribe_wav_allows_empty_words_and_segments(tmp_path: Path):
    audio_path = tmp_path / "silent.wav"
    _write_pcm_wav(audio_path)

    class EmptyDiarizer:
        def diarize(self, path: str):
            return []

    class EmptyTranscriber:
        def transcribe_words(self, path: str):
            return []

    service = SpeakerTranscriptionService(
        transcriber=EmptyTranscriber(),
        diarizer=EmptyDiarizer(),
        clock=RecordingClock([2.0, 2.25]),
        lock=RecordingLock([]),
    )

    result = service.transcribe_wav(audio_path)

    assert result == {
        "segments": [],
        "processing_ms": 250,
        "audio_seconds": pytest.approx(1.0),
        "rtf": pytest.approx(0.25),
    }


def test_transcribe_wav_holds_the_lock_around_both_model_calls(tmp_path: Path):
    audio_path = tmp_path / "locked.wav"
    _write_pcm_wav(audio_path)

    log: list[object] = []
    lock = RecordingLock(log)

    class GuardedDiarizer:
        def diarize(self, path: str):
            log.append(("diarize", lock.depth))
            assert lock.depth == 1
            return [SpeakerTurn(0.0, 1.0, "SPEAKER_00")]

    class GuardedTranscriber:
        def transcribe_words(self, path: str):
            log.append(("transcribe", lock.depth))
            assert lock.depth == 1
            return [TimedWord(0.0, 0.5, "hello")]

    service = SpeakerTranscriptionService(
        transcriber=GuardedTranscriber(),
        diarizer=GuardedDiarizer(),
        clock=RecordingClock([50.0, 50.5]),
        lock=lock,
    )

    service.transcribe_wav(audio_path)

    assert log == [
        "lock_enter",
        ("diarize", 1),
        ("transcribe", 1),
        "lock_exit",
    ]


def test_transcribe_wav_reads_final_clock_after_grouping(
    monkeypatch, tmp_path: Path
):
    audio_path = tmp_path / "timed.wav"
    _write_pcm_wav(audio_path)

    grouped = {"value": False}

    class FinalClock:
        def __init__(self):
            self.calls = 0

        def __call__(self) -> float:
            self.calls += 1
            if self.calls == 1:
                return 1.0
            assert grouped["value"], "grouping should happen before the final clock"
            return 1.25

    class FakeDiarizer:
        def diarize(self, path: str):
            return [SpeakerTurn(0.0, 1.0, "SPEAKER_00")]

    class FakeTranscriber:
        def transcribe_words(self, path: str):
            return [TimedWord(0.0, 0.5, "hello")]

    def fake_group_speaker_words(words):
        grouped["value"] = True
        return [words]

    monkeypatch.setattr(webapp, "group_speaker_words", fake_group_speaker_words)

    service = SpeakerTranscriptionService(
        transcriber=FakeTranscriber(),
        diarizer=FakeDiarizer(),
        clock=FinalClock(),
        lock=RecordingLock([]),
    )

    result = service.transcribe_wav(audio_path)

    assert result["processing_ms"] == 250
    assert grouped["value"] is True


def test_transcribe_wav_retains_falsey_injected_factories_clock_and_lock(
    monkeypatch, tmp_path: Path
):
    audio_path = tmp_path / "falsey.wav"
    _write_pcm_wav(audio_path)

    log: list[str] = []

    class FalseyClock:
        def __init__(self):
            self.values = iter([2.0, 2.25])

        def __bool__(self):
            return False

        def __call__(self) -> float:
            log.append("clock")
            return next(self.values)

    class FalseyLock:
        def __bool__(self):
            return False

        def __enter__(self):
            log.append("lock_enter")
            return self

        def __exit__(self, exc_type, exc, tb):
            log.append("lock_exit")
            return False

    class FakeDiarizer:
        def diarize(self, path: str):
            log.append("diarize")
            return [SpeakerTurn(0.0, 1.0, "SPEAKER_00")]

    class FakeTranscriber:
        def transcribe_words(self, path: str):
            log.append("transcribe")
            return [TimedWord(0.0, 0.5, "hello")]

    class FalseyFactory:
        def __init__(self, name: str, value):
            self.name = name
            self.value = value

        def __bool__(self):
            return False

        def __call__(self):
            log.append(self.name)
            return self.value

    monkeypatch.setattr(
        webapp,
        "_default_transcriber_factory",
        lambda: (_ for _ in ()).throw(AssertionError("default transcriber factory")),
    )
    monkeypatch.setattr(
        webapp,
        "_default_diarizer_factory",
        lambda: (_ for _ in ()).throw(AssertionError("default diarizer factory")),
    )

    service = SpeakerTranscriptionService(
        transcriber_factory=FalseyFactory("transcriber_factory", FakeTranscriber()),
        diarizer_factory=FalseyFactory("diarizer_factory", FakeDiarizer()),
        clock=FalseyClock(),
        lock=FalseyLock(),
    )

    result = service.transcribe_wav(audio_path)

    assert log == [
        "clock",
        "lock_enter",
        "diarizer_factory",
        "diarize",
        "transcriber_factory",
        "transcribe",
        "lock_exit",
        "clock",
    ]
    assert result["processing_ms"] == 250


def test_read_request_body_rejects_invalid_content_length_before_read():
    class BoomRFile:
        def read(self, size: int):
            raise AssertionError("read should not be called")

    with pytest.raises(webapp.RequestError, match="Content-Length"):
        webapp._read_request_body(
            BoomRFile(),
            {"Content-Length": str(webapp.MAX_UPLOAD_BYTES + 1)},
        )


@pytest.mark.parametrize(
    ("content_length", "body", "message"),
    [
        ("abc", b"", "Content-Length"),
        ("-1", b"", "Content-Length"),
        ("4", b"abc", "truncated"),
    ],
)
def test_read_request_body_rejects_bad_length_and_short_body(
    content_length: str, body: bytes, message: str
):
    class BodyRFile:
        def __init__(self, payload: bytes):
            self.payload = payload
            self.calls: list[int] = []

        def read(self, size: int):
            self.calls.append(size)
            return self.payload

    rfile = BodyRFile(body)

    with pytest.raises(webapp.RequestError, match=message):
        webapp._read_request_body(rfile, {"Content-Length": content_length})


@pytest.mark.parametrize(
    ("content_type", "body", "message"),
    [
        ("text/plain", b"hello", "multipart/form-data"),
        ("multipart/form-data", b"hello", "boundary"),
        ("multipart/form-data; boundary=broken", b"not multipart", "multipart"),
    ],
)
def test_parse_multipart_audio_upload_rejects_invalid_content_types(
    content_type: str, body: bytes, message: str
):
    with pytest.raises((webapp.RequestError, ValueError), match=message):
        webapp._parse_multipart_audio_upload(body, content_type)


@pytest.mark.parametrize(
    ("parts", "message"),
    [
        ([], "malformed"),
        ([("audio", "clip.mp3", b"123")], ".wav"),
        ([("audio", "clip.wav", b"")], "empty"),
        (
            [("audio", "clip.wav", b"123"), ("audio", "clip2.wav", b"456")],
            "exactly one",
        ),
        ([("other", "clip.wav", b"123")], "audio"),
    ],
)
def test_parse_multipart_audio_upload_rejects_missing_empty_duplicate_and_bad_filename(
    parts: list[tuple[str, str, bytes]], message: str
):
    body, content_type = _make_multipart_body(parts)

    with pytest.raises((webapp.RequestError, ValueError), match=message):
        webapp._parse_multipart_audio_upload(body, content_type)


def test_create_server_returns_threading_http_server():
    service = object()
    server = webapp.create_server("127.0.0.1", 0, service)
    try:
        assert server.__class__.__name__ == "ThreadingHTTPServer"
        assert server.RequestHandlerClass is not None
    finally:
        server.server_close()


def test_http_health_and_routes_return_json_html_and_404():
    service = object()
    handler_cls = webapp.make_handler(service, html="<html><body>hello</body></html>")
    server = webapp.create_server("127.0.0.1", 0, service)
    server.RequestHandlerClass = handler_cls
    with _running_http_server(server):
        response, payload = _http_request(server, "GET", "/health")
        assert response.status == 200
        assert _assert_json_response(response, payload) == {"status": "ok"}

        response, payload = _http_request(server, "GET", "/")
        assert response.status == 200
        assert response.getheader("Content-Type") == "text/html; charset=utf-8"
        assert response.getheader("Content-Length") == str(len(payload))
        assert payload.decode() == "<html><body>hello</body></html>"

        response, payload = _http_request(server, "GET", "/missing")
        assert response.status == 404
        assert _assert_json_response(response, payload) == {"error": "not found"}


def test_http_post_transcribe_returns_json_and_cleans_up_temp_file(tmp_path: Path):
    audio_path = tmp_path / "sample.wav"
    _write_pcm_wav(audio_path)
    audio_bytes = audio_path.read_bytes()
    body, content_type = _make_multipart_body([("audio", "sample.WAV", audio_bytes)])

    recorded: list[Path] = []

    class FakeService:
        def transcribe_wav(self, path):
            recorded.append(Path(path))
            assert Path(path).exists()
            return {
                "segments": [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "hi"}],
                "processing_ms": 7,
                "audio_seconds": 1.0,
                "rtf": 0.007,
            }

    service = FakeService()
    server = webapp.create_server("127.0.0.1", 0, service)
    with _running_http_server(server):
        response, payload = _http_request(
            server,
            "POST",
            "/api/transcribe",
            body=body,
            headers={
                "Content-Type": content_type,
                "Content-Length": str(len(body)),
            },
        )

    assert response.status == 200
    assert _assert_json_response(response, payload) == {
        "segments": [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "hi"}],
        "processing_ms": 7,
        "audio_seconds": 1.0,
        "rtf": 0.007,
    }
    assert len(recorded) == 1
    assert not recorded[0].exists()


def test_http_post_transcribe_cleans_up_temp_file_when_service_raises(tmp_path: Path):
    audio_path = tmp_path / "sample.wav"
    _write_pcm_wav(audio_path)
    audio_bytes = audio_path.read_bytes()
    body, content_type = _make_multipart_body([("audio", "sample.wav", audio_bytes)])

    recorded: list[Path] = []

    class BoomService:
        def transcribe_wav(self, path):
            recorded.append(Path(path))
            assert Path(path).exists()
            raise RuntimeError("boom")

    service = BoomService()
    server = webapp.create_server("127.0.0.1", 0, service)
    with _running_http_server(server):
        response, payload = _http_request(
            server,
            "POST",
            "/api/transcribe",
            body=body,
            headers={
                "Content-Type": content_type,
                "Content-Length": str(len(body)),
            },
        )

    assert response.status == 500
    assert _assert_json_response(response, payload) == {"error": "Internal server error"}
    assert len(recorded) == 1
    assert not recorded[0].exists()


def test_http_post_transcribe_maps_model_runtime_error_to_503_without_leaking_details(
    tmp_path: Path,
):
    audio_path = tmp_path / "sample.wav"
    _write_pcm_wav(audio_path)
    audio_bytes = audio_path.read_bytes()
    body, content_type = _make_multipart_body([("audio", "sample.wav", audio_bytes)])

    class ModelService:
        def transcribe_wav(self, path):
            raise RuntimeError(
                "Missing HF_TOKEN for pyannote model agreement before upload"
            )

    service = ModelService()
    server = webapp.create_server("127.0.0.1", 0, service)
    with _running_http_server(server):
        response, payload = _http_request(
            server,
            "POST",
            "/api/transcribe",
            body=body,
            headers={
                "Content-Type": content_type,
                "Content-Length": str(len(body)),
            },
        )

    decoded = _assert_json_response(response, payload)
    assert response.status == 503
    assert decoded == {"error": "The transcription service is temporarily unavailable."}
    assert "HF_TOKEN" not in payload.decode()
    assert "pyannote" not in payload.decode().lower()
    assert "agreement" not in payload.decode().lower()


@pytest.mark.parametrize(
    "message",
    [
        "HF_TOKEN is missing",
        "pyannote.audio is required for diarization",
        "Please accept the model conditions",
    ],
)
def test_http_post_transcribe_maps_known_model_setup_errors_to_503(
    tmp_path: Path, message: str
):
    audio_path = tmp_path / "sample.wav"
    _write_pcm_wav(audio_path)
    audio_bytes = audio_path.read_bytes()
    body, content_type = _make_multipart_body([("audio", "sample.wav", audio_bytes)])

    class ModelService:
        def transcribe_wav(self, path):
            raise RuntimeError(message)

    service = ModelService()
    server = webapp.create_server("127.0.0.1", 0, service)
    with _running_http_server(server):
        response, payload = _http_request(
            server,
            "POST",
            "/api/transcribe",
            body=body,
            headers={
                "Content-Type": content_type,
                "Content-Length": str(len(body)),
            },
        )

    assert response.status == 503
    assert _assert_json_response(response, payload) == {
        "error": "The transcription service is temporarily unavailable."
    }


def test_http_post_transcribe_keeps_generic_pyannote_runtime_errors_as_500(
    tmp_path: Path,
):
    audio_path = tmp_path / "sample.wav"
    _write_pcm_wav(audio_path)
    audio_bytes = audio_path.read_bytes()
    body, content_type = _make_multipart_body([("audio", "sample.wav", audio_bytes)])

    class ModelService:
        def transcribe_wav(self, path):
            raise RuntimeError("pyannote runtime bug")

    service = ModelService()
    server = webapp.create_server("127.0.0.1", 0, service)
    with _running_http_server(server):
        response, payload = _http_request(
            server,
            "POST",
            "/api/transcribe",
            body=body,
            headers={
                "Content-Type": content_type,
                "Content-Length": str(len(body)),
            },
        )

    assert response.status == 500
    assert _assert_json_response(response, payload) == {"error": "Internal server error"}


def test_http_post_transcribe_rejects_invalid_content_type_before_service():
    class ExplodingService:
        def transcribe_wav(self, path):
            raise AssertionError("service should not be called")

    service = ExplodingService()
    server = webapp.create_server("127.0.0.1", 0, service)
    body = b"not multipart"
    with _running_http_server(server):
        response, payload = _http_request(
            server,
            "POST",
            "/api/transcribe",
            body=body,
            headers={
                "Content-Type": "text/plain",
                "Content-Length": str(len(body)),
            },
        )

    assert response.status == 400
    assert _assert_json_response(response, payload) == {
        "error": "multipart/form-data uploads only"
    }


def test_http_post_transcribe_returns_500_for_service_value_error(tmp_path: Path):
    audio_path = tmp_path / "sample.wav"
    _write_pcm_wav(audio_path)
    audio_bytes = audio_path.read_bytes()
    body, content_type = _make_multipart_body([("audio", "sample.wav", audio_bytes)])

    class ValueErrorService:
        def transcribe_wav(self, path):
            raise ValueError(f"{path} is not a supported PCM WAV")

    service = ValueErrorService()
    server = webapp.create_server("127.0.0.1", 0, service)
    with _running_http_server(server):
        response, payload = _http_request(
            server,
            "POST",
            "/api/transcribe",
            body=body,
            headers={
                "Content-Type": content_type,
                "Content-Length": str(len(body)),
            },
        )

    decoded = _assert_json_response(response, payload)
    assert response.status == 500
    assert decoded == {"error": "Internal server error"}
    assert str(tmp_path) not in payload.decode()


def test_transcribe_uploaded_wav_converts_validation_failure_to_request_error(
    monkeypatch, tmp_path: Path
):
    audio_path = tmp_path / "sample.wav"
    _write_pcm_wav(audio_path)
    audio_bytes = audio_path.read_bytes()

    called = []

    def fake_validate_pcm_wav(path):
        called.append(path)
        raise ValueError("bad wav")

    class FakeService:
        def transcribe_wav(self, path):
            raise AssertionError("service should not be called")

    monkeypatch.setattr(webapp, "validate_pcm_wav", fake_validate_pcm_wav)

    with pytest.raises(webapp.RequestError, match="Invalid WAV upload"):
        webapp._transcribe_uploaded_wav(FakeService(), audio_bytes)

    assert called and called[0].endswith(".wav")


def test_validate_complete_pcm_wav_rejects_truncated_data_payload(tmp_path: Path):
    audio_path = tmp_path / "sample.wav"
    _write_pcm_wav(audio_path)
    truncated_path = tmp_path / "truncated.wav"
    truncated_path.write_bytes(audio_path.read_bytes()[:-1])

    webapp.validate_pcm_wav(truncated_path)

    with pytest.raises(webapp.RequestError, match="Invalid WAV upload"):
        webapp._validate_complete_pcm_wav(truncated_path)


def test_http_post_transcribe_rejects_truncated_pcm_payload_and_skips_service(tmp_path: Path):
    audio_path = tmp_path / "sample.wav"
    _write_pcm_wav(audio_path)
    truncated_path = tmp_path / "truncated.wav"
    truncated_path.write_bytes(audio_path.read_bytes()[:-1])
    body, content_type = _make_multipart_body(
        [("audio", "sample.wav", truncated_path.read_bytes())]
    )

    class ExplodingService:
        def transcribe_wav(self, path):
            raise AssertionError("service should not be called")

    server = webapp.create_server("127.0.0.1", 0, ExplodingService())
    with _running_http_server(server):
        response, payload = _http_request(
            server,
            "POST",
            "/api/transcribe",
            body=body,
            headers={
                "Content-Type": content_type,
                "Content-Length": str(len(body)),
            },
        )

    assert response.status == 400
    assert _assert_json_response(response, payload) == {"error": "Invalid WAV upload"}


@pytest.mark.parametrize(
    ("scenario", "exc_factory"),
    [
        ("send_response", lambda h: BrokenPipeError("broken")),
        ("send_header", lambda h: BrokenPipeError("broken")),
        ("end_headers", lambda h: ConnectionResetError("reset")),
        ("write", lambda h: BrokenPipeError("broken")),
    ],
)
def test_response_bytes_swallow_disconnects_during_error_responses(scenario, exc_factory):
    events = []

    class FakeWriter:
        def write(self, data):
            events.append(("write", data))
            if scenario == "write":
                raise exc_factory(None)

    class FakeHandler:
        def __init__(self):
            self.wfile = FakeWriter()

        def send_response(self, status):
            events.append(("send_response", status))
            if scenario == "send_response":
                raise exc_factory(None)

        def send_header(self, name, value):
            events.append(("send_header", name, value))
            if scenario == "send_header":
                raise exc_factory(None)

        def end_headers(self):
            events.append(("end_headers",))
            if scenario == "end_headers":
                raise exc_factory(None)

    webapp._response_bytes(FakeHandler(), 500, "application/json; charset=utf-8", b"{}")
    assert events


def test_parse_multipart_audio_upload_rejects_unclosed_final_boundary_and_ignores_embedded_boundary_bytes():
    body, content_type = _make_multipart_body(
        [("audio", "clip.wav", b"noise--boundary123--noise")],
        closing=False,
    )

    with pytest.raises(webapp.RequestError, match="boundary|defect|malformed"):
        webapp._parse_multipart_audio_upload(body, content_type)


def test_parse_multipart_audio_upload_rejects_part_defects_even_with_valid_headers():
    boundary = "boundary123"
    body = (
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"audio\"; filename=\"clip.wav\"\r\n"
        f"Content-Type: audio/wav\r\n"
        f"BadHeader\r\n"
        f"\r\n"
        f"abc\r\n"
        f"--{boundary}--\r\n"
    ).encode()

    with pytest.raises(webapp.RequestError, match="defect|malformed"):
        webapp._parse_multipart_audio_upload(body, f"multipart/form-data; boundary={boundary}")
