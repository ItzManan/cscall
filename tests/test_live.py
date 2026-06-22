from __future__ import annotations

from pathlib import Path
import wave

import pytest

from cscall.streaming.metrics import StreamingMetrics
from cscall.streaming.session import AudioChunk, StreamingEvent


def test_pcm_chunk_rejects_empty_odd_and_oversized_frames():
    import cscall.live as live

    with pytest.raises(live.LiveProtocolError, match="invalid pcm frame"):
        live.pcm_chunk(b"", timestamp_ms=0)

    with pytest.raises(live.LiveProtocolError, match="invalid pcm frame"):
        live.pcm_chunk(b"\x01", timestamp_ms=0)

    with pytest.raises(live.LiveProtocolError, match="invalid pcm frame"):
        live.pcm_chunk(b"\x00\x00" * (live.MAX_FRAME_BYTES // 2 + 1), timestamp_ms=0)


def test_pcm_chunk_computes_duration_timestamp_and_energy_decision():
    import cscall.live as live

    silence = b"\x00\x00" * live.SAMPLE_RATE
    chunk = live.pcm_chunk(silence, timestamp_ms=250)

    assert chunk == AudioChunk(
        timestamp_ms=1250,
        duration_ms=1000,
        data=silence,
        is_speech=False,
    )

    sample = (1000).to_bytes(2, "little", signed=True)
    speech = sample * live.SAMPLE_RATE
    speech_chunk = live.pcm_chunk(speech, timestamp_ms=0)

    assert speech_chunk.is_speech is True
    assert speech_chunk.duration_ms == 1000
    assert speech_chunk.timestamp_ms == 1000


def test_serialize_event_renders_text_and_metrics_payloads():
    import cscall.live as live

    text_event = StreamingEvent(type="partial", timestamp_ms=42, text="hello")
    assert live.serialize_event(text_event) == {
        "type": "partial",
        "timestamp_ms": 42,
        "text": "hello",
    }

    metrics_event = StreamingEvent(
        type="metrics",
        timestamp_ms=99,
        metrics=StreamingMetrics(
            audio_ms=2000,
            decode_ms=500,
            first_partial_latency_ms=180,
            final_latency_ms=260,
        ),
    )
    assert live.serialize_event(metrics_event) == {
        "type": "metrics",
        "timestamp_ms": 99,
        "audio_seconds": 2.0,
        "latency_ms": 180,
        "final_latency_ms": 260,
        "rtf": 0.25,
    }


def test_pcm_adapter_writes_16k_mono_wav_and_cleans_up_on_error(tmp_path: Path):
    import cscall.live as live

    pcm = (1234).to_bytes(2, "little", signed=True) * 160
    seen_path: Path | None = None

    class InspectingTranscriber:
        def transcribe(self, path: str) -> str:
            nonlocal seen_path
            seen_path = Path(path)
            with wave.open(path, "rb") as wav:
                assert wav.getframerate() == live.SAMPLE_RATE
                assert wav.getnchannels() == 1
                assert wav.getsampwidth() == 2
            raise RuntimeError("boom")

    adapter = live._PCMTranscriptionAdapter(InspectingTranscriber())

    with pytest.raises(RuntimeError, match="boom"):
        adapter.transcribe(pcm)

    assert seen_path is not None
    assert not seen_path.exists()


def _make_client(app):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    return TestClient(app)


def _require_fastapi():
    pytest.importorskip("fastapi")


def test_live_routes_return_placeholders():
    _require_fastapi()
    import cscall.live as live

    app = live.create_live_app(session_factory=lambda: None)
    client = _make_client(app)

    html = client.get("/").text
    worklet = client.get("/audio-worklet.js").text

    assert "<title>Live transcript</title>" in html
    assert '<h1 id="live-heading">Live transcript</h1>' in html
    assert "Start microphone" in html
    assert "Stop" in html
    assert 'aria-live="polite"' in html
    assert 'role="status"' in html
    assert "Final transcript" in html
    assert "Stable transcript" in html
    assert "Partial transcript" in html
    assert "Latency" in html
    assert "RTF" in html
    assert "speaker labels are not available yet" in html.lower()
    assert "WAV upload is the speaker-attributed path" in html
    assert "innerHTML" not in html
    assert "script src=" not in html.lower()
    assert "link rel=" not in html.lower()

    assert "AudioWorkletProcessor" in worklet
    assert "Int16Array" in worklet
    assert "16000" in worklet
    assert "1600" in worklet
    assert "registerProcessor" in worklet
    assert "postMessage" in worklet
    assert "inputs[0] && inputs[0][0]" in worklet
    assert "no external assets" not in worklet.lower()
    assert client.get("/health").json() == {"status": "ok"}


def test_live_ui_script_uses_safe_dom_and_reconnect_limits():
    _require_fastapi()
    import cscall.live as live

    app = live.create_live_app(session_factory=lambda: None)
    client = _make_client(app)
    html = client.get("/").text

    assert "navigator.mediaDevices.getUserMedia" in html
    assert "audioWorklet.addModule('/audio-worklet.js')" in html or "audioWorklet.addModule(\"/audio-worklet.js\")" in html
    assert "new WebSocket(" in html
    assert "/ws/transcribe" in html
    assert "location.protocol === 'https:'" in html or 'location.protocol === "https:"' in html
    assert "wss:" in html
    assert "ws:" in html
    assert "replaceChildren" in html
    assert "createElement" in html
    assert "textContent" in html
    assert "innerHTML" not in html
    assert "maximum 3 attempts" not in html.lower()
    assert "MAX_RECONNECT_ATTEMPTS" in html
    assert "reconnectAttempts" in html
    assert "STOP_TIMEOUT_MS" in html
    assert 'payload.type === "stable"' in html


def test_websocket_streams_events_and_stop_closes_cleanly():
    _require_fastapi()
    import cscall.live as live

    class FakeSession:
        def __init__(self):
            self.update_calls: list[AudioChunk] = []
            self.flush_calls: list[int] = []

        def update(self, chunk: AudioChunk):
            self.update_calls.append(chunk)
            return [
                StreamingEvent(type="partial", timestamp_ms=chunk.timestamp_ms, text="hel"),
                StreamingEvent(
                    type="metrics",
                    timestamp_ms=chunk.timestamp_ms,
                    metrics=StreamingMetrics(
                        audio_ms=1000,
                        decode_ms=250,
                        first_partial_latency_ms=180,
                        final_latency_ms=260,
                    ),
                ),
            ]

        def flush(self, timestamp_ms: int):
            self.flush_calls.append(timestamp_ms)
            return [
                StreamingEvent(
                    type="final",
                    timestamp_ms=timestamp_ms,
                    text="hello",
                )
            ]

    sessions: list[FakeSession] = []

    def session_factory():
        session = FakeSession()
        sessions.append(session)
        return session

    app = live.create_live_app(session_factory=session_factory)
    client = _make_client(app)

    with client.websocket_connect("/ws/transcribe") as ws:
        ws.send_json({"type": "start", "sample_rate": 16000})
        ws.send_bytes(b"\x00\x00" * 16000)

        assert ws.receive_json() == {
            "type": "partial",
            "timestamp_ms": 1000,
            "text": "hel",
        }
        assert ws.receive_json() == {
            "type": "metrics",
            "timestamp_ms": 1000,
            "audio_seconds": 1.0,
            "latency_ms": 180,
            "final_latency_ms": 260,
            "rtf": 0.25,
        }

        ws.send_json({"type": "stop"})
        assert ws.receive_json() == {
            "type": "final",
            "timestamp_ms": 1000,
            "text": "hello",
        }
        assert ws.receive_json() == {"type": "stopped"}

    assert len(sessions) == 1
    assert len(sessions[0].update_calls) == 1
    assert sessions[0].flush_calls == [1000]


def test_websocket_rejects_audio_before_start_with_exact_error():
    _require_fastapi()
    import cscall.live as live

    app = live.create_live_app(session_factory=lambda: None)
    client = _make_client(app)

    with client.websocket_connect("/ws/transcribe") as ws:
        ws.send_bytes(b"\x00\x00")
        assert ws.receive_json() == {
            "type": "error",
            "message": "send start before audio",
        }


@pytest.mark.parametrize(
    "message",
    [
        '{"type": "bogus"}',
        "not-json",
        '{"type": "start", "sample_rate": 8000}',
        "[]",
        "null",
        '"start"',
    ],
)
def test_websocket_rejects_malformed_or_unsupported_controls(message: str):
    _require_fastapi()
    import cscall.live as live

    app = live.create_live_app(session_factory=lambda: None)
    client = _make_client(app)

    with client.websocket_connect("/ws/transcribe") as ws:
        ws.send_text(message)
        assert ws.receive_json() == {
            "type": "error",
            "message": "invalid request",
        }


def test_websocket_rejects_invalid_pcm_frames_after_start():
    _require_fastapi()
    import cscall.live as live

    app = live.create_live_app(session_factory=lambda: None)
    client = _make_client(app)

    with client.websocket_connect("/ws/transcribe") as ws:
        ws.send_json({"type": "start", "sample_rate": 16000})
        ws.send_bytes(b"\x01")
        assert ws.receive_json() == {
            "type": "error",
            "message": "invalid request",
        }


def test_websocket_rejects_audio_after_cumulative_limit(monkeypatch):
    _require_fastapi()
    import cscall.live as live

    class FakeSession:
        def update(self, chunk: AudioChunk):
            return []

        def flush(self, timestamp_ms: int):
            return []

    def session_factory():
        return FakeSession()

    def fake_pcm_chunk(data: bytes, timestamp_ms: int, energy_threshold: int = 200):
        return AudioChunk(
            timestamp_ms=timestamp_ms + 61000,
            duration_ms=61000,
            data=data,
            is_speech=True,
        )

    monkeypatch.setattr(live, "pcm_chunk", fake_pcm_chunk)

    app = live.create_live_app(session_factory=session_factory)
    client = _make_client(app)

    with client.websocket_connect("/ws/transcribe") as ws:
        ws.send_json({"type": "start", "sample_rate": 16000})
        ws.send_bytes(b"\x00\x00")
        ws.send_bytes(b"\x00\x00")
        assert ws.receive_json() == {
            "type": "error",
            "message": "invalid request",
        }


def test_websocket_flushes_once_on_disconnect():
    _require_fastapi()
    import cscall.live as live

    class FakeSession:
        def __init__(self):
            self.flush_calls = 0
            self.update_calls = 0

        def update(self, chunk: AudioChunk):
            self.update_calls += 1
            return []

        def flush(self, timestamp_ms: int):
            self.flush_calls += 1
            return []

    session = FakeSession()

    app = live.create_live_app(session_factory=lambda: session)
    client = _make_client(app)

    with client.websocket_connect("/ws/transcribe") as ws:
        ws.send_json({"type": "start", "sample_rate": 16000})
        ws.send_bytes(b"\x00\x00" * 16000)

    assert session.flush_calls == 1
    assert session.update_calls == 1


def test_websocket_protocol_error_after_start_flushes_once():
    _require_fastapi()
    import cscall.live as live

    class FakeSession:
        def __init__(self):
            self.flush_calls = 0

        def update(self, chunk: AudioChunk):
            return []

        def flush(self, timestamp_ms: int):
            self.flush_calls += 1
            return []

    session = FakeSession()
    client = _make_client(live.create_live_app(session_factory=lambda: session))

    with client.websocket_connect("/ws/transcribe") as ws:
        ws.send_json({"type": "start", "sample_rate": 16000})
        ws.send_json({"type": "unknown"})
        assert ws.receive_json() == {
            "type": "error",
            "message": "invalid request",
        }

    assert session.flush_calls == 1


def test_websocket_hides_update_failure_and_flushes_once():
    _require_fastapi()
    import cscall.live as live

    class FailingSession:
        def __init__(self):
            self.flush_calls = 0

        def update(self, chunk: AudioChunk):
            raise RuntimeError("secret model detail")

        def flush(self, timestamp_ms: int):
            self.flush_calls += 1
            return []

    session = FailingSession()
    client = _make_client(live.create_live_app(session_factory=lambda: session))

    with client.websocket_connect("/ws/transcribe") as ws:
        ws.send_json({"type": "start", "sample_rate": 16000})
        ws.send_bytes(b"\x00\x00")
        assert ws.receive_json() == {
            "type": "error",
            "message": "transcription failed",
        }

    assert session.flush_calls == 1


def test_websocket_connections_use_isolated_sessions():
    _require_fastapi()
    import cscall.live as live

    class FakeSession:
        def __init__(self):
            self.flush_calls = 0

        def flush(self, timestamp_ms: int):
            self.flush_calls += 1
            return []

    sessions: list[FakeSession] = []

    def session_factory():
        session = FakeSession()
        sessions.append(session)
        return session

    client = _make_client(live.create_live_app(session_factory=session_factory))
    for _ in range(2):
        with client.websocket_connect("/ws/transcribe") as ws:
            ws.send_json({"type": "start", "sample_rate": 16000})
            ws.send_json({"type": "stop"})
            assert ws.receive_json() == {"type": "stopped"}

    assert len(sessions) == 2
    assert sessions[0] is not sessions[1]
    assert [session.flush_calls for session in sessions] == [1, 1]
