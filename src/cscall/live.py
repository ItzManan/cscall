import asyncio
import json
from pathlib import Path
import tempfile
import threading
import wave

from cscall.asr_baseline import WhisperTranscriber
from cscall.streaming.audio import is_speech_pcm
from cscall.streaming.endpointing import EndpointConfig, EndpointDetector
from cscall.streaming.session import AudioChunk, StreamingEvent, StreamingSession


SAMPLE_RATE = 16000
MAX_FRAME_BYTES = 32000
MAX_BUFFER_SECONDS = 120

_GENERIC_ERROR_MESSAGE = "invalid request"
_START_AUDIO_ERROR_MESSAGE = "send start before audio"
_LIVE_INDEX_HTML = "<!doctype html><html><body>Live transcript</body></html>"
_AUDIO_WORKLET_JS = (
    "class LiveAudioWorkletProcessor extends AudioWorkletProcessor {}"
    "registerProcessor('live-audio-worklet-processor', LiveAudioWorkletProcessor);"
)


class LiveProtocolError(ValueError):
    pass


def pcm_chunk(
    data: bytes,
    timestamp_ms: int,
    energy_threshold: int = 200,
) -> AudioChunk:
    if not data or len(data) % 2 != 0 or len(data) > MAX_FRAME_BYTES:
        raise LiveProtocolError("invalid pcm frame")

    samples = len(data) // 2
    duration_ms = max(1, round(samples * 1000 / SAMPLE_RATE))
    return AudioChunk(
        timestamp_ms=timestamp_ms + duration_ms,
        duration_ms=duration_ms,
        data=data,
        is_speech=is_speech_pcm(data, sample_width=2, threshold=energy_threshold),
    )


def serialize_event(event: StreamingEvent) -> dict[str, object]:
    payload: dict[str, object] = {
        "type": event.type,
        "timestamp_ms": event.timestamp_ms,
    }

    if event.metrics is None:
        if event.text:
            payload["text"] = event.text
        return payload

    metrics = event.metrics
    payload.update(
        {
            "audio_seconds": getattr(metrics, "audio_ms", 0) / 1000,
            "latency_ms": getattr(metrics, "first_partial_latency_ms", None),
            "final_latency_ms": getattr(metrics, "final_latency_ms", None),
            "rtf": getattr(metrics, "rtf", 0.0),
        }
    )
    return payload


class _PCMTranscriptionAdapter:
    def __init__(self, transcriber: WhisperTranscriber):
        self._transcriber = transcriber
        self._lock = threading.Lock()

    def transcribe(self, pcm_data: bytes) -> str:
        temp_path: Path | None = None
        handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            temp_path = Path(handle.name)
            handle.close()
            with wave.open(str(temp_path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(SAMPLE_RATE)
                wav.writeframes(pcm_data)

            with self._lock:
                return self._transcriber.transcribe(str(temp_path))
        finally:
            try:
                handle.close()
            except Exception:
                pass
            if temp_path is not None:
                try:
                    temp_path.unlink()
                except FileNotFoundError:
                    pass


def create_live_app(
    session_factory=None,
    model: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = None,
):
    try:
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.responses import HTMLResponse, Response
    except ImportError as exc:  # pragma: no cover - dependency gate
        raise RuntimeError("fastapi is required for live mode") from exc

    app = FastAPI()
    session_lock = threading.Lock()
    shared: dict[str, object | None] = {"transcriber": None, "adapter": None}

    def default_session_factory() -> StreamingSession:
        with session_lock:
            adapter = shared["adapter"]
            if adapter is None:
                transcriber = WhisperTranscriber(
                    model_size=model,
                    device=device,
                    compute_type=compute_type,
                    language=language,
                )
                adapter = _PCMTranscriptionAdapter(transcriber)
                shared["transcriber"] = transcriber
                shared["adapter"] = adapter

        detector = EndpointDetector(EndpointConfig(frame_ms=100))
        return StreamingSession(
            transcribe=adapter.transcribe,  # type: ignore[union-attr]
            step_ms=500,
            agreement=2,
            endpoint_detector=detector,
        )

    build_session = session_factory or default_session_factory

    @app.get("/")
    def root() -> HTMLResponse:
        return HTMLResponse(_LIVE_INDEX_HTML)

    @app.get("/audio-worklet.js")
    def audio_worklet() -> Response:
        return Response(_AUDIO_WORKLET_JS, media_type="application/javascript")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.websocket("/ws/transcribe")
    async def transcribe_socket(websocket: WebSocket) -> None:
        await websocket.accept()
        session = None
        started = False
        flushed = False
        audio_ms = 0

        async def send_error_and_close(message: str) -> None:
            await websocket.send_json({"type": "error", "message": message})
            await websocket.close()

        async def flush(send_events: bool) -> None:
            nonlocal flushed
            if session is None or flushed:
                return
            flushed = True
            events = await asyncio.to_thread(session.flush, audio_ms)
            if send_events:
                for event in events or []:
                    await websocket.send_json(serialize_event(event))

        try:
            session = build_session()
            while True:
                try:
                    message = await websocket.receive()
                except WebSocketDisconnect:
                    break

                message_type = message.get("type")
                if message_type == "websocket.disconnect":
                    break

                if message.get("bytes") is not None:
                    if not started:
                        await send_error_and_close(_START_AUDIO_ERROR_MESSAGE)
                        return

                    try:
                        chunk = pcm_chunk(message["bytes"], audio_ms)
                    except LiveProtocolError:
                        await send_error_and_close(_GENERIC_ERROR_MESSAGE)
                        return

                    audio_ms = chunk.timestamp_ms
                    if audio_ms > MAX_BUFFER_SECONDS * 1000:
                        await send_error_and_close(_GENERIC_ERROR_MESSAGE)
                        return

                    try:
                        events = await asyncio.to_thread(session.update, chunk)
                    except Exception:
                        await send_error_and_close("transcription failed")
                        return
                    for event in events or []:
                        await websocket.send_json(serialize_event(event))
                    continue

                text = message.get("text")
                if text is None:
                    await send_error_and_close(_GENERIC_ERROR_MESSAGE)
                    return

                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    await send_error_and_close(_GENERIC_ERROR_MESSAGE)
                    return
                if not isinstance(payload, dict):
                    await send_error_and_close(_GENERIC_ERROR_MESSAGE)
                    return

                if not started:
                    if (
                        payload.get("type") != "start"
                        or payload.get("sample_rate") != SAMPLE_RATE
                    ):
                        await send_error_and_close(_GENERIC_ERROR_MESSAGE)
                        return

                    started = True
                    continue

                if payload.get("type") != "stop":
                    await send_error_and_close(_GENERIC_ERROR_MESSAGE)
                    return

                try:
                    await flush(send_events=True)
                except Exception:
                    await send_error_and_close("transcription failed")
                    return
                await websocket.send_json({"type": "stopped"})
                await websocket.close()
                return
        except Exception:
            try:
                await send_error_and_close("transcription failed")
            except Exception:
                pass
        finally:
            if started and not flushed:
                try:
                    await flush(send_events=False)
                except Exception:
                    pass

    return app
