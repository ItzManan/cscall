# Phase 5 Live Transcription Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add microphone-to-WebSocket live ASR, accessible browser captions, and a locally runnable Docker image while preserving the existing upload UI.

**Architecture:** A small optional FastAPI/uvicorn server owns one `StreamingSession` per WebSocket and reuses one locked Whisper model. The browser uses an `AudioWorklet` to send 16 kHz mono PCM16 frames and renders JSON transcript events with native DOM APIs.

**Tech Stack:** Python 3.11, FastAPI, uvicorn, existing faster-whisper streaming core, Web Audio API, pytest, Docker.

---

## File Map

- Create `src/cscall/live.py`: FastAPI app, WebSocket protocol, PCM-to-Whisper adapter, embedded live HTML, and AudioWorklet JavaScript.
- Create `tests/test_live.py`: protocol, isolation, cleanup, UI, and CLI-facing app tests with fake sessions/transcribers.
- Modify `src/cscall/streaming/session.py`: explicit idempotent flush for stop/disconnect.
- Modify `tests/test_streaming_session.py`: flush behavior.
- Modify `src/cscall/cli.py`: `live` command and uvicorn launcher.
- Modify `tests/test_cli.py`: parser and launcher wiring.
- Modify `pyproject.toml`: optional live dependencies.
- Create `Dockerfile` and `.dockerignore`: local live-server packaging.
- Modify `README.md`: installation, live usage, limitations, and Docker commands.

### Task 1: Explicit Streaming Flush

**Files:**
- Modify: `src/cscall/streaming/session.py`
- Modify: `tests/test_streaming_session.py`

- [ ] **Step 1: Write failing flush tests**

Add tests proving that an active short utterance is finalized and that repeated or idle flushes return no events:

```python
def test_streaming_session_flush_finalizes_active_audio_once():
    session = StreamingSession(
        transcribe=lambda audio: "hello",
        step_ms=500,
        endpoint_detector=EndpointDetector(
            EndpointConfig(frame_ms=100, min_speech_ms=100, trailing_silence_ms=500)
        ),
        decode_ms=10,
    )
    session.update(AudioChunk(100, 100, b"a", True))

    events = session.flush(150)

    assert [event.type for event in events] == ["partial", "final", "metrics"]
    assert next(event.text for event in events if event.type == "final") == "hello"
    assert session.flush(200) == []


def test_streaming_session_flush_is_empty_before_speech():
    session = StreamingSession(transcribe=lambda audio: "unused")
    assert session.flush(0) == []
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
python -m pytest tests/test_streaming_session.py -k flush -v
```

Expected: failure because `StreamingSession.flush` does not exist.

- [ ] **Step 3: Implement the minimal flush method**

Add:

```python
def flush(self, timestamp_ms: int) -> list[StreamingEvent]:
    if not self._in_utterance:
        return []
    return self._finalize(timestamp_ms)
```

- [ ] **Step 4: Run streaming tests**

Run:

```bash
python -m pytest tests/test_streaming_session.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cscall/streaming/session.py tests/test_streaming_session.py
git commit -m "feat: flush live streaming sessions"
```

### Task 2: Live Protocol Core

**Files:**
- Create: `src/cscall/live.py`
- Create: `tests/test_live.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add optional dependencies and failing protocol tests**

Add:

```toml
live = ["fastapi>=0.115,<1", "uvicorn>=0.34,<1"]
```

Test these public helpers without loading a model:

```python
def test_pcm_chunk_rejects_odd_and_oversized_payloads():
    with pytest.raises(LiveProtocolError, match="whole PCM16"):
        pcm_chunk(b"\x00", timestamp_ms=0)
    with pytest.raises(LiveProtocolError, match="too large"):
        pcm_chunk(b"\x00\x00" * 16001, timestamp_ms=0)


def test_pcm_chunk_computes_duration_and_speech():
    chunk = pcm_chunk(
        (1000).to_bytes(2, "little", signed=True) * 1600,
        timestamp_ms=100,
        energy_threshold=200,
    )
    assert chunk.duration_ms == 100
    assert chunk.timestamp_ms == 200
    assert chunk.is_speech is True


def test_serialize_metrics_event():
    payload = serialize_event(
        StreamingEvent("metrics", 500, metrics=StreamingMetrics(1000, 500, 250, 20))
    )
    assert payload == {
        "type": "metrics",
        "timestamp_ms": 500,
        "audio_seconds": 1.0,
        "latency_ms": 250,
        "final_latency_ms": 20,
        "rtf": 0.5,
    }
```

- [ ] **Step 2: Verify the tests fail**

Run:

```bash
python -m pytest tests/test_live.py -v
```

Expected: import failure for `cscall.live`.

- [ ] **Step 3: Implement protocol helpers and the PCM transcriber**

Implement these minimal boundaries in `live.py`:

```python
SAMPLE_RATE = 16_000
MAX_FRAME_BYTES = SAMPLE_RATE * 2
MAX_BUFFER_SECONDS = 120

class LiveProtocolError(ValueError):
    pass

def pcm_chunk(data: bytes, timestamp_ms: int, energy_threshold: int = 200) -> AudioChunk:
    if not data or len(data) % 2:
        raise LiveProtocolError("audio must contain whole PCM16 samples")
    if len(data) > MAX_FRAME_BYTES:
        raise LiveProtocolError("audio frame is too large")
    duration_ms = max(1, round(len(data) / 2 * 1000 / SAMPLE_RATE))
    return AudioChunk(
        timestamp_ms=timestamp_ms + duration_ms,
        duration_ms=duration_ms,
        data=data,
        is_speech=is_speech_pcm(data, 2, energy_threshold),
    )

def serialize_event(event: StreamingEvent) -> dict[str, object]:
    if event.type != "metrics":
        return {"type": event.type, "timestamp_ms": event.timestamp_ms, "text": event.text}
    metrics = event.metrics
    return {
        "type": "metrics",
        "timestamp_ms": event.timestamp_ms,
        "audio_seconds": metrics.audio_ms / 1000,
        "latency_ms": metrics.first_partial_latency_ms,
        "final_latency_ms": metrics.final_latency_ms,
        "rtf": metrics.rtf,
    }
```

Add one private callable that writes growing PCM bytes to a temporary 16 kHz
mono WAV, calls `WhisperTranscriber.transcribe`, and always unlinks the file.
Protect it with a single `threading.Lock`; do not add a model pool.

- [ ] **Step 4: Run protocol tests**

Run:

```bash
python -m pytest tests/test_live.py -v
```

Expected: protocol helper tests pass.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/cscall/live.py tests/test_live.py
git commit -m "feat: add live PCM protocol core"
```

### Task 3: FastAPI WebSocket Session

**Files:**
- Modify: `src/cscall/live.py`
- Modify: `tests/test_live.py`

- [ ] **Step 1: Write failing app tests**

Use FastAPI `TestClient` with an injected `session_factory`. Cover:

```python
def test_health_and_static_routes():
    client = TestClient(create_live_app(session_factory=lambda: FakeSession()))
    assert client.get("/health").json() == {"status": "ok"}
    assert "Live transcript" in client.get("/").text
    assert "AudioWorkletProcessor" in client.get("/audio-worklet.js").text


def test_websocket_requires_start_then_streams_events():
    session = FakeSession(events=[StreamingEvent("partial", 100, "namaste")])
    client = TestClient(create_live_app(session_factory=lambda: session))
    with client.websocket_connect("/ws/transcribe") as socket:
        socket.send_json({"type": "start", "sample_rate": 16000, "language": "hi"})
        socket.send_bytes(b"\x01\x00" * 1600)
        assert socket.receive_json()["text"] == "namaste"
        socket.send_json({"type": "stop"})
        assert socket.receive_json() == {"type": "stopped"}
    assert session.flush_calls == 1


def test_websocket_rejects_audio_before_start_without_internal_details():
    client = TestClient(create_live_app(session_factory=lambda: FakeSession()))
    with client.websocket_connect("/ws/transcribe") as socket:
        socket.send_bytes(b"\x00\x00")
        payload = socket.receive_json()
        assert payload == {"type": "error", "message": "send start before audio"}
```

Also assert two socket connections receive distinct fake session instances and
disconnect flushes exactly once.

- [ ] **Step 2: Verify failure**

Run:

```bash
python -m pytest tests/test_live.py -k "websocket or routes or health" -v
```

Expected: failure because `create_live_app` is absent.

- [ ] **Step 3: Implement the app**

Add `create_live_app(session_factory=None)` with:

- `GET /` returning `LIVE_UI_HTML`;
- `GET /audio-worklet.js` returning `AUDIO_WORKLET_JS`;
- `GET /health` returning `{"status": "ok"}`;
- `WS /ws/transcribe` accepting one start message, binary PCM frames, and stop;
- `await asyncio.to_thread(session.update, chunk)` for blocking inference;
- event serialization and `send_json`;
- a cumulative 120-second per-connection limit;
- one idempotent `flush` in `finally`;
- generic client errors only.

The default factory lazily creates one shared `WhisperTranscriber` and returns a
fresh `StreamingSession` configured for 100 ms endpoint frames, 500 ms decode
steps, LocalAgreement-2, and the shared locked PCM transcriber.

- [ ] **Step 4: Run app tests and the full suite**

Run:

```bash
python -m pytest tests/test_live.py -v
python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cscall/live.py tests/test_live.py
git commit -m "feat: stream live ASR over WebSocket"
```

### Task 4: Native Microphone UI

**Files:**
- Modify: `src/cscall/live.py`
- Modify: `tests/test_live.py`

- [ ] **Step 1: Write failing static UI assertions**

Assert the embedded assets contain:

```python
def test_live_ui_uses_accessible_native_microphone_controls():
    html = LIVE_UI_HTML
    worklet = AUDIO_WORKLET_JS
    assert 'id="start-button"' in html
    assert 'id="stop-button"' in html
    assert 'aria-live="polite"' in html
    assert "navigator.mediaDevices.getUserMedia" in html
    assert "new WebSocket" in html
    assert "audioWorklet.addModule" in html
    assert "textContent" in html
    assert "innerHTML" not in html
    assert "AudioWorkletProcessor" in worklet
    assert "16000" in worklet
    assert "Int16Array" in worklet
```

- [ ] **Step 2: Verify failure**

Run:

```bash
python -m pytest tests/test_live.py -k live_ui -v
```

Expected: missing asset assertions fail.

- [ ] **Step 3: Implement the minimal UI and worklet**

Reuse Phase 4 colors and typography without introducing a frontend framework.
The worklet accumulates resampled values until 1,600 samples, converts them to
clamped PCM16, and posts the underlying `ArrayBuffer`.

The page must:

- request microphone permission only after Start;
- open `ws://` or `wss://` based on `location.protocol`;
- send the start control before binary frames;
- render stable text, a separate partial tail, final lines, latency, and RTF
  using `textContent`/`replaceChildren`;
- stop tracks and disconnect audio nodes on Stop;
- make at most three reconnect attempts after unexpected closure;
- state clearly that speaker labels are available through WAV upload, not live.

- [ ] **Step 4: Run UI and full tests**

Run:

```bash
python -m pytest tests/test_live.py -v
python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cscall/live.py tests/test_live.py
git commit -m "feat: add browser microphone captions"
```

### Task 5: CLI and Local Smoke

**Files:**
- Modify: `src/cscall/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing CLI tests**

```python
def test_live_command_forwards_server_options(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "run_live_server", lambda **kwargs: calls.append(kwargs))
    cli.main([
        "live", "--host", "0.0.0.0", "--port", "9000", "--model", "tiny",
        "--device", "cpu", "--compute-type", "int8", "--language", "hi",
    ])
    assert calls == [{
        "host": "0.0.0.0", "port": 9000, "model": "tiny",
        "device": "cpu", "compute_type": "int8", "language": "hi",
    }]
```

Also assert invalid ports are rejected by argparse.

- [ ] **Step 2: Verify failure**

Run:

```bash
python -m pytest tests/test_cli.py -k live -v
```

Expected: parser rejects the unknown `live` command.

- [ ] **Step 3: Add the launcher**

Add a `live` parser using the existing host, port, and speaker-model arguments.
Implement `run_live_server` in `live.py` as the thin boundary:

```python
def run_live_server(host, port, model, device, compute_type, language):
    import uvicorn
    app = create_live_app(
        model=model,
        device=device,
        compute_type=compute_type,
        language=language,
    )
    uvicorn.run(app, host=host, port=port)
```

Import `run_live_server` lazily inside the CLI branch so ordinary commands do
not require FastAPI.

- [ ] **Step 4: Run CLI and help tests**

Run:

```bash
python -m pytest tests/test_cli.py -k live -v
python -m cscall.cli live --help
```

Expected: tests pass and help exits zero without loading a model.

- [ ] **Step 5: Commit**

```bash
git add src/cscall/cli.py src/cscall/live.py tests/test_cli.py
git commit -m "feat: add live transcription command"
```

### Task 6: Docker and Documentation

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`
- Modify: `README.md`
- Modify: `tests/test_live.py`

- [ ] **Step 1: Write failing packaging assertions**

```python
def test_dockerfile_runs_live_server_without_embedding_secrets():
    dockerfile = Path("Dockerfile").read_text()
    assert '.[live]' in dockerfile
    assert "cscall.cli" in dockerfile
    assert "live" in dockerfile
    assert "HF_TOKEN" not in dockerfile
```

- [ ] **Step 2: Verify failure**

Run:

```bash
python -m pytest tests/test_live.py -k dockerfile -v
```

Expected: failure because `Dockerfile` does not exist.

- [ ] **Step 3: Add minimal packaging**

Use one CPU image:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir ".[live]"
EXPOSE 8000
CMD ["python", "-m", "cscall.cli", "live", "--host", "0.0.0.0", "--port", "8000"]
```

Ignore `.git`, `.venv`, caches, datasets, local models, `.env`, IDE files, and
recordings. Document:

```bash
python -m pip install -e ".[dev,live]"
python -m cscall.cli live
docker build -t cscall .
docker run --rm -p 8000:8000 cscall
```

State that localhost microphone capture works in modern browsers, remote
deployment requires HTTPS, live captions do not yet include speaker labels,
and the first model use downloads weights.

- [ ] **Step 4: Run fresh verification**

Run:

```bash
python -m pytest -q
python -m compileall -q src tests
python -m cscall.cli live --help
git diff --check
docker build -t cscall .
```

Expected: tests, compile, help, diff check, and image build all succeed.

Start the image and verify:

```bash
docker run --rm -d --name cscall-smoke -p 18000:8000 cscall
curl --fail http://127.0.0.1:18000/health
docker stop cscall-smoke
```

Expected health body: `{"status":"ok"}`.

- [ ] **Step 5: Perform manual browser smoke**

Run `python -m cscall.cli live --model tiny`, open localhost, allow microphone
access, speak a short Hinglish phrase, and verify partial text, final text,
latency, RTF, Stop, and a second fresh session.

- [ ] **Step 6: Commit**

```bash
git add Dockerfile .dockerignore README.md tests/test_live.py
git commit -m "docs: package live transcription demo"
```

