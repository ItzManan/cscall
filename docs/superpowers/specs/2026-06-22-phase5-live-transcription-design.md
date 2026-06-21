# Phase 5 Design: Live Microphone Transcription

**Date:** 2026-06-22  
**Status:** Approved direction; awaiting written-spec review

## Goal

Add a dependable live demo to the existing upload UI:

- capture microphone audio in the browser;
- stream 16 kHz mono PCM16 frames over one WebSocket;
- show partial and final Hindi-English transcripts with latency and RTF;
- recover cleanly from stop, disconnect, and reconnect;
- package the application for local Docker use.

The first live release prioritizes reliable ASR. True incremental speaker
diarization is deferred to Phase 6 because Community-1 is an offline pipeline
and `diart` currently brings an older, conflicting pyannote dependency stack.

## Chosen Approach

Use a small FastAPI/uvicorn live server, the native Web Audio API, and the
existing `StreamingSession`.

The alternatives are:

1. `MediaRecorder` with compressed chunks. Browser code is shorter, but the
   server then needs codec/container decoding and incurs extra latency.
2. Separate ASR and diarization services. This is easier to scale later but
   adds deployment and synchronization complexity that the local demo does not
   need.

Raw PCM over one WebSocket is the smallest approach that preserves predictable
latency and directly matches the current streaming core.

## Scope

### Included

- A `live` CLI command.
- `GET /`, `GET /health`, and `WS /ws/transcribe`.
- Browser microphone permission, start/stop controls, connection state, and
  accessible live transcript updates.
- An `AudioWorklet` that downsamples browser audio to 16 kHz mono PCM16 and
  posts approximately 100–200 ms frames.
- One isolated `StreamingSession` per WebSocket connection.
- Partial, stable, final, metrics, error, and stopped server events.
- Backpressure by retaining audio while skipping redundant intermediate
  decodes when inference falls behind.
- Clean final flush and resource teardown on stop or disconnect.
- Client reconnect for unexpected network loss; reconnect starts a fresh
  transcript session.
- Dockerfile, `.dockerignore`, and concise run documentation.

### Deferred

- Incremental speaker labels while a person is still speaking.
- Stable speaker identity across reconnects or separate calls.
- Running Community-1 repeatedly over rolling windows.
- `diart` or a custom embedding/clustering pipeline.
- Authentication, TLS termination, multi-process model sharing, persistent
  transcripts, and production autoscaling.

The existing upload workflow remains the accurate speaker-attributed path.
After a live call, users can upload the saved WAV when speaker attribution is
required. Browser-side recording/download is not included in this phase; add
it with Phase 6 speaker work so there is one intentional post-call workflow.

## Architecture

The live server is separate from the existing standard-library upload server
so Phase 4 remains dependency-light and stable.

```text
Microphone
  -> AudioWorklet (browser-rate float audio)
  -> downsample + PCM16 frames
  -> WebSocket
  -> per-connection StreamingSession
  -> JSON transcript/metric events
  -> accessible live captions
```

FastAPI and uvicorn are optional `live` dependencies. The server constructs the
Whisper transcriber once and reuses it across connections. Each connection owns
only mutable streaming state. Inference is serialized with the existing model
lock because concurrent use has not been proven safe.

## Protocol

The browser first sends:

```json
{"type":"start","sample_rate":16000,"language":"hi"}
```

Audio then travels as binary little-endian signed PCM16 frames. The browser
sends `{"type":"stop"}` to request a final flush.

Server events use JSON:

```json
{"type":"partial","text":"..."}
{"type":"stable","text":"..."}
{"type":"final","text":"..."}
{"type":"metrics","latency_ms":420,"rtf":0.63,"audio_seconds":4.8}
{"type":"error","message":"..."}
{"type":"stopped"}
```

Unknown control messages, binary audio before `start`, invalid frame sizes,
unsupported sample rates, and oversized buffered audio close the connection
with a concise error. Secrets and internal exception details never cross the
WebSocket.

## Browser Behavior

The existing visual style is reused. A live panel adds:

- Start microphone / Stop buttons;
- connection and microphone status;
- stable transcript text plus a visually distinct partial tail;
- current latency and RTF;
- an `aria-live="polite"` status region.

Start requests microphone access, opens the socket, loads the worklet, and
begins sending frames. Stop halts tracks and the audio graph, sends the stop
control, waits briefly for the final event, then closes the socket. Unexpected
disconnects trigger bounded reconnect attempts with a fresh session and a
clear notice that prior partial text cannot continue.

## Backpressure

Audio is never silently discarded. The session keeps incoming audio but allows
only one decode at a time. If frames arrive during inference, they are folded
into the next decode rather than creating a queue of obsolete re-decodes.
Metrics expose RTF so the limitation is visible. A hard per-connection audio
buffer limit prevents unbounded memory growth and returns an actionable error.

## Error Handling

- Microphone denial stays in the browser and leaves Start enabled.
- A malformed frame or protocol violation returns `error` and closes only that
  connection.
- Model failure returns a generic transcription error, then tears down the
  session.
- Stop and disconnect are idempotent.
- `/health` reports process readiness without downloading or running a model.
- Container startup does not require `HF_TOKEN`; live ASR has no diarization
  dependency.

## Testing

Implementation follows TDD.

- Unit tests cover PCM validation, protocol state, event serialization,
  backpressure coalescing, stop/flush, and disconnect cleanup using fake
  sessions.
- FastAPI WebSocket tests cover start, binary audio, transcript events,
  malformed input, and isolated connections without loading Whisper.
- Static UI tests verify the worklet route, microphone controls, safe DOM APIs,
  and accessibility status.
- A browser smoke test verifies microphone start/stop and rendered events using
  a fake server/session.
- A manual real-model smoke test records a short Hinglish phrase and confirms
  partial text, final convergence, latency, and RTF.
- Docker verification builds the image and checks `/health`.

## Success Criteria

Phase 5 is complete when:

1. a user can start speaking from a localhost browser and see partial and final
   transcript updates;
2. stop and reconnect do not leak state between sessions;
3. overload coalesces redundant inference work without dropping audio;
4. automated tests pass without a microphone, model download, or `HF_TOKEN`;
5. the Docker image starts and serves a healthy live UI;
6. the README clearly distinguishes live ASR from offline speaker attribution.

