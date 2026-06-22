import asyncio
import json
from pathlib import Path
import tempfile
import threading
from textwrap import dedent
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


def _live_index_html() -> str:
    return dedent(
        """
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Live transcript</title>
            <style>
              :root {
                color-scheme: light dark;
                font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                line-height: 1.45;
              }

              body {
                margin: 0;
                min-height: 100vh;
                background: Canvas;
                color: CanvasText;
              }

              main {
                max-width: 48rem;
                margin: 0 auto;
                padding: 1.25rem;
              }

              .card {
                border: 1px solid color-mix(in oklab, CanvasText 18%, Canvas 82%);
                border-radius: 1rem;
                padding: 1rem;
                background: color-mix(in oklab, Canvas 94%, CanvasText 6%);
                box-shadow: 0 1rem 2rem color-mix(in oklab, CanvasText 12%, transparent);
              }

              h1,
              h2,
              p {
                margin-top: 0;
              }

              .controls {
                display: flex;
                flex-wrap: wrap;
                gap: 0.75rem;
                margin: 1rem 0;
              }

              button {
                appearance: none;
                border: 1px solid color-mix(in oklab, CanvasText 18%, Canvas 82%);
                border-radius: 999px;
                padding: 0.7rem 1rem;
                background: AccentColor;
                color: AccentColorText;
                font: inherit;
              }

              button:disabled {
                opacity: 0.55;
              }

              #status,
              #notice,
              #metrics {
                margin-bottom: 0.75rem;
              }

              #final-transcript {
                margin: 0;
                padding-left: 1.25rem;
              }

              #partial-tail {
                margin: 0.75rem 0 0;
                padding: 0.75rem 0.85rem;
                border-left: 0.35rem solid AccentColor;
                background: color-mix(in oklab, AccentColor 12%, Canvas 88%);
                border-radius: 0.4rem;
                font-variant-ligatures: none;
              }

              .label {
                font-size: 0.92rem;
                opacity: 0.82;
                margin-bottom: 0.2rem;
              }

              .metric-value {
                font-variant-numeric: tabular-nums;
                font-weight: 600;
              }

              @media (prefers-reduced-motion: reduce) {
                *,
                *::before,
                *::after {
                  animation: none !important;
                  transition: none !important;
                  scroll-behavior: auto !important;
                }
              }
            </style>
          </head>
          <body>
            <main>
              <section class="card" aria-labelledby="live-heading">
                <h1 id="live-heading">Live transcript</h1>
                <p>
                  Start microphone to capture audio locally. Live speaker labels are not available yet and WAV upload is the speaker-attributed path.
                </p>

                <div class="controls" aria-label="Microphone controls">
                  <button type="button" id="start-button">Start microphone</button>
                  <button type="button" id="stop-button" disabled>Stop</button>
                </div>

                <p id="status" role="status" aria-live="polite"></p>
                <p id="notice" aria-live="polite"></p>

                <section aria-labelledby="transcript-heading">
                  <h2 id="transcript-heading">Transcript</h2>

                  <div id="metrics" aria-label="Streaming metrics">
                    <p><span class="label">Latency:</span> <output class="metric-value" id="latency-value">—</output></p>
                    <p><span class="label">RTF:</span> <output class="metric-value" id="rtf-value">—</output></p>
                  </div>

                  <div aria-label="Final transcript" id="final-transcript-region">
                    <p class="label">Final transcript</p>
                    <ul id="final-transcript"></ul>
                  </div>

                  <div aria-label="Stable transcript">
                    <p class="label">Stable transcript</p>
                    <p id="stable-tail"></p>
                  </div>

                  <div aria-label="Partial transcript" id="partial-transcript-region">
                    <p class="label">Partial transcript</p>
                    <p id="partial-tail"></p>
                  </div>
                </section>
              </section>
            </main>

            <script>
              (() => {
                const SAMPLE_RATE = 16000;
                const MAX_RECONNECT_ATTEMPTS = 3;
                const STOP_TIMEOUT_MS = 2000;
                const startButton = document.getElementById("start-button");
                const stopButton = document.getElementById("stop-button");
                const statusElement = document.getElementById("status");
                const noticeElement = document.getElementById("notice");
                const finalTranscriptElement = document.getElementById("final-transcript");
                const stableTailElement = document.getElementById("stable-tail");
                const partialTailElement = document.getElementById("partial-tail");
                const latencyValueElement = document.getElementById("latency-value");
                const rtfValueElement = document.getElementById("rtf-value");

                const state = {
                  stream: null,
                  socket: null,
                  audioContext: null,
                  sourceNode: null,
                  workletNode: null,
                  pendingAudio: [],
                  desiredRecording: false,
                  stopping: false,
                  awaitingStopped: false,
                  sessionToken: 0,
                  reconnectAttempts: 0,
                  reconnectTimer: null,
                  stopTimer: null,
                  finalTextSeen: "",
                  stableText: "",
                };

                function setStatus(connectionStatus, micStatus) {
                  statusElement.textContent = `Connection: ${connectionStatus}. Microphone: ${micStatus}.`;
                }

                function setNotice(message) {
                  noticeElement.textContent = message || "";
                }

                function setMetrics(payload) {
                  if (!payload) {
                    latencyValueElement.textContent = "—";
                    rtfValueElement.textContent = "—";
                    return;
                  }

                  latencyValueElement.textContent =
                    payload.latency_ms == null ? "—" : `${payload.latency_ms} ms`;
                  rtfValueElement.textContent = payload.rtf == null ? "—" : String(payload.rtf);
                }

                function setButtons(running) {
                  startButton.disabled = running;
                  stopButton.disabled = !running;
                }

                function clearPartial() {
                  partialTailElement.textContent = "";
                }

                function clearStable() {
                  state.stableText = "";
                  stableTailElement.textContent = "";
                }

                function clearFinals() {
                  finalTranscriptElement.replaceChildren();
                  state.finalTextSeen = "";
                }

                function appendFinal(text) {
                  if (!text || text === state.finalTextSeen) {
                    return;
                  }

                  state.finalTextSeen = text;
                  const item = document.createElement("li");
                  item.textContent = text;
                  finalTranscriptElement.appendChild(item);
                }

                function buildSocketUrl() {
                  const scheme = location.protocol === "https:" ? "wss:" : "ws:";
                  return `${scheme}//${location.host}/ws/transcribe`;
                }

                function stopTracks() {
                  if (!state.stream) {
                    return;
                  }

                  for (const track of state.stream.getTracks()) {
                    track.stop();
                  }
                  state.stream = null;
                }

                async function closeAudioContext() {
                  const context = state.audioContext;
                  state.audioContext = null;
                  if (!context) {
                    return;
                  }

                  try {
                    await context.close();
                  } catch (_error) {
                    // Ignore shutdown errors.
                  }
                }

                async function teardownPipeline({ keepStream = false } = {}) {
                  if (state.sourceNode) {
                    try {
                      state.sourceNode.disconnect();
                    } catch (_error) {
                      // Ignore disconnect races.
                    }
                    state.sourceNode = null;
                  }

                  if (state.workletNode) {
                    try {
                      state.workletNode.port.onmessage = null;
                      state.workletNode.disconnect();
                    } catch (_error) {
                      // Ignore disconnect races.
                    }
                    state.workletNode = null;
                  }

                  state.pendingAudio = [];
                  await closeAudioContext();
                  if (!keepStream) {
                    stopTracks();
                  }
                }

                function flushPendingAudio() {
                  if (!state.socket || state.socket.readyState !== WebSocket.OPEN) {
                    return;
                  }

                  while (state.pendingAudio.length > 0) {
                    const buffer = state.pendingAudio.shift();
                    if (buffer) {
                      state.socket.send(buffer);
                    }
                  }
                }

                function enqueueAudio(buffer) {
                  if (!buffer) {
                    return;
                  }

                  if (state.socket && state.socket.readyState === WebSocket.OPEN) {
                    state.socket.send(buffer);
                    return;
                  }

                  state.pendingAudio.push(buffer);
                }

                async function attachAudioPipeline(token) {
                  if (!state.stream) {
                    return;
                  }

                  await teardownPipeline({ keepStream: true });
                  if (token !== state.sessionToken || !state.desiredRecording) {
                    return;
                  }

                  const context = new AudioContext();
                  state.audioContext = context;
                  await context.audioWorklet.addModule("/audio-worklet.js");
                  if (token !== state.sessionToken || !state.desiredRecording) {
                    await closeAudioContext();
                    return;
                  }

                  state.sourceNode = context.createMediaStreamSource(state.stream);
                  state.workletNode = new AudioWorkletNode(
                    context,
                    "live-audio-worklet-processor",
                  );

                  state.workletNode.port.onmessage = (event) => {
                    if (token !== state.sessionToken) {
                      return;
                    }

                    if (event.data instanceof ArrayBuffer) {
                      enqueueAudio(event.data);
                    }
                  };

                  state.sourceNode.connect(state.workletNode);
                  state.workletNode.connect(context.destination);

                  if (context.state === "suspended") {
                    await context.resume();
                  }
                }

                function resetForFreshSession({ reconnect = false } = {}) {
                  clearPartial();
                  setMetrics(null);
                  setNotice("");
                  if (!reconnect) {
                    clearFinals();
                  }
                }

                function finishStoppedSession(message) {
                  state.desiredRecording = false;
                  state.stopping = false;
                  state.awaitingStopped = false;
                  state.reconnectAttempts = 0;
                  clearTimeout(state.reconnectTimer);
                  state.reconnectTimer = null;
                  clearTimeout(state.stopTimer);
                  state.stopTimer = null;
                  setButtons(false);
                  setStatus("idle", "stopped");
                  setNotice(message || "");
                  state.pendingAudio = [];

                  const socket = state.socket;
                  state.socket = null;
                  if (socket && socket.readyState < WebSocket.CLOSING) {
                    try {
                      socket.close();
                    } catch (_error) {
                      // Ignore close races.
                    }
                  }

                  void teardownPipeline({ keepStream: false });
                }

                function scheduleReconnect(token) {
                  if (state.reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
                    finishStoppedSession(
                      "Connection ended. Prior partial cannot continue, and the reconnect limit was reached.",
                    );
                    return;
                  }

                  state.reconnectAttempts += 1;
                  setNotice(
                    `Connection ended. Prior partial cannot continue. Reconnecting attempt ${state.reconnectAttempts} of ${MAX_RECONNECT_ATTEMPTS}.`,
                  );
                  setStatus("reconnecting", "microphone still active");
                  void teardownPipeline({ keepStream: true });
                  state.socket = null;
                  state.reconnectTimer = window.setTimeout(() => {
                    if (token !== state.sessionToken || !state.desiredRecording) {
                      return;
                    }
                    void openSession({ reconnect: true, token });
                  }, 200);
                }

                function handleMessage(event, token) {
                  if (token !== state.sessionToken) {
                    return;
                  }

                  let payload = event.data;
                  if (typeof payload === "string") {
                    try {
                      payload = JSON.parse(payload);
                    } catch (_error) {
                      return;
                    }
                  }

                  if (!payload || typeof payload !== "object") {
                    return;
                  }

                  if (payload.type === "partial") {
                    partialTailElement.textContent = payload.text || "";
                    return;
                  }

                  if (payload.type === "stable") {
                    state.stableText += payload.text || "";
                    stableTailElement.textContent = state.stableText;
                    return;
                  }

                  if (payload.type === "final") {
                    appendFinal(payload.text || "");
                    clearStable();
                    clearPartial();
                    return;
                  }

                  if (payload.type === "metrics") {
                    setMetrics(payload);
                    return;
                  }

                  if (payload.type === "error") {
                    setNotice(payload.message || "transcription failed");
                    setStatus("error", "microphone active");
                    return;
                  }

                  if (payload.type === "stopped") {
                    finishStoppedSession("Stopped.");
                  }
                }

                function handleSocketClose(token) {
                  if (token !== state.sessionToken) {
                    return;
                  }

                  state.socket = null;
                  if (state.desiredRecording && !state.stopping) {
                    scheduleReconnect(token);
                    return;
                  }

                  if (state.awaitingStopped || state.stopping) {
                    finishStoppedSession("Stopped.");
                  }
                }

                async function openSession({ reconnect = false, token = null } = {}) {
                  const sessionToken = token ?? ++state.sessionToken;
                  state.sessionToken = sessionToken;
                  state.desiredRecording = true;
                  state.stopping = false;
                  state.awaitingStopped = false;
                  state.pendingAudio = [];
                  clearStable();
                  setButtons(true);
                  setStatus(reconnect ? "connecting" : "starting", "requesting microphone");
                  resetForFreshSession({ reconnect });

                  try {
                    if (!state.stream) {
                      state.stream = await navigator.mediaDevices.getUserMedia({
                        audio: true,
                        video: false,
                      });
                    }
                  } catch (_error) {
                    finishStoppedSession("Microphone permission is required.");
                    return;
                  }

                  if (sessionToken !== state.sessionToken || !state.desiredRecording) {
                    return;
                  }

                  const socket = new WebSocket(buildSocketUrl());
                  state.socket = socket;
                  socket.binaryType = "arraybuffer";

                  socket.onopen = async () => {
                    if (sessionToken !== state.sessionToken || !state.desiredRecording) {
                      try {
                        socket.close();
                      } catch (_error) {
                        // Ignore.
                      }
                      return;
                    }

                    try {
                      socket.send(JSON.stringify({ type: "start", sample_rate: SAMPLE_RATE }));
                      await attachAudioPipeline(sessionToken);
                      flushPendingAudio();
                      setStatus("connected", "listening");
                    } catch (_error) {
                      setNotice("Unable to start live audio.");
                      finishStoppedSession("Stopped.");
                    }
                  };

                  socket.onmessage = (event) => {
                    handleMessage(event, sessionToken);
                  };

                  socket.onclose = () => {
                    handleSocketClose(sessionToken);
                  };

                  socket.onerror = () => {
                    if (sessionToken === state.sessionToken) {
                      setStatus("connection error", "listening");
                    }
                  };
                }

                async function stopSession() {
                  if (!state.desiredRecording && !state.socket) {
                    return;
                  }

                  state.desiredRecording = false;
                  state.stopping = true;
                  state.awaitingStopped = true;
                  clearTimeout(state.reconnectTimer);
                  state.reconnectTimer = null;
                  setStatus("stopping", "waiting for final transcript");

                  if (state.socket && state.socket.readyState === WebSocket.OPEN) {
                    try {
                      state.socket.send(JSON.stringify({ type: "stop" }));
                      state.stopTimer = window.setTimeout(() => {
                        if (state.awaitingStopped) {
                          finishStoppedSession("Stopped before the server confirmed final text.");
                        }
                      }, STOP_TIMEOUT_MS);
                      return;
                    } catch (_error) {
                      // Fall through to local shutdown.
                    }
                  }

                  finishStoppedSession("Stopped.");
                }

                startButton.addEventListener("click", () => {
                  if (state.desiredRecording) {
                    return;
                  }

                  void openSession();
                });

                stopButton.addEventListener("click", () => {
                  void stopSession();
                });

                setButtons(false);
                setStatus("idle", "not started");
                setMetrics(null);
              })();
            </script>
          </body>
        </html>
        """
    ).strip()


def _audio_worklet_js() -> str:
    return dedent(
        """
        class LiveAudioWorkletProcessor extends AudioWorkletProcessor {
          constructor() {
            super();
            this._inputRate = sampleRate;
            this._outputRate = 16000;
            this._step = this._inputRate / this._outputRate;
            this._source = [];
            this._nextSourcePos = 0;
            this._pending = new Int16Array(1600);
            this._pendingIndex = 0;
          }

          _pushSample(sample) {
            const clamped = Math.max(-1, Math.min(1, Number.isFinite(sample) ? sample : 0));
            const int16 = clamped < 0 ? Math.round(clamped * 32768) : Math.round(clamped * 32767);
            this._pending[this._pendingIndex] = int16;
            this._pendingIndex += 1;
            if (this._pendingIndex === this._pending.length) {
              const frame = new Int16Array(this._pending);
              this.port.postMessage(frame.buffer, [frame.buffer]);
              this._pendingIndex = 0;
            }
          }

          _appendSamples(samples) {
            for (let index = 0; index < samples.length; index += 1) {
              this._source.push(Number.isFinite(samples[index]) ? samples[index] : 0);
            }

            while (this._nextSourcePos + 1 < this._source.length) {
              const leftIndex = Math.floor(this._nextSourcePos);
              const fraction = this._nextSourcePos - leftIndex;
              const left = this._source[leftIndex];
              const right = this._source[leftIndex + 1];
              this._pushSample(left + (right - left) * fraction);
              this._nextSourcePos += this._step;
            }

            const consumed = Math.max(0, Math.floor(this._nextSourcePos) - 1);
            if (consumed > 0) {
              this._source = this._source.slice(consumed);
              this._nextSourcePos -= consumed;
            }
          }

          process(inputs) {
            const channel = inputs[0] && inputs[0][0] ? inputs[0][0] : null;
            if (channel && channel.length > 0) {
              this._appendSamples(channel);
            } else {
              this._appendSamples(new Float32Array(128));
            }
            return true;
          }
        }

        registerProcessor("live-audio-worklet-processor", LiveAudioWorkletProcessor);
        """
    ).strip()


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
        return HTMLResponse(_live_index_html())

    @app.get("/audio-worklet.js")
    def audio_worklet() -> Response:
        return Response(_audio_worklet_js(), media_type="application/javascript")

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
