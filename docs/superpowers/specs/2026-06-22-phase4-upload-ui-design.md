# Phase 4 Upload Transcript UI Design

**Date:** 2026-06-22
**Status:** Approved

## Goal

Add a framework-free local web page that uploads a PCM WAV file and displays a
speaker-colored, timestamped transcript using the Phase 3 ASR, diarization, and
fusion logic.

## Scope

Phase 4 will:

- serve one static HTML page with embedded CSS and JavaScript;
- accept a WAV upload through a small local HTTP endpoint;
- validate the uploaded WAV before model work;
- run two-speaker diarization, timestamped ASR, and fusion;
- return a stable JSON transcript shape;
- render grouped speaker turns, timestamps, processing time, and errors;
- remain accessible by keyboard and screen readers;
- add no frontend framework or build tooling.

Phase 4 will not:

- capture the microphone;
- use WebSockets;
- emit partial live transcripts;
- provide authentication, persistent storage, or public deployment;
- add Docker packaging.

Those belong to Phase 5, which will reuse the JSON event and transcript shape.

## Runtime and Dependency Strategy

The HTTP service uses Python's standard library:

- `http.server.ThreadingHTTPServer`;
- `BaseHTTPRequestHandler`;
- `cgi.FieldStorage` is avoided because it is removed in newer Python versions;
- multipart parsing is implemented with the standard-library `email` parser.

No FastAPI dependency is introduced yet. Phase 5 will add FastAPI when WebSocket
support is actually needed.

The existing optional `diarization` dependency and `HF_TOKEN` requirements
remain unchanged.

## API

### `GET /`

Returns the embedded HTML application.

### `GET /health`

Returns:

```json
{"status": "ok"}
```

This endpoint is kept because Phase 5 packaging will need it.

### `POST /api/transcribe`

Accepts `multipart/form-data` with one field named `audio`.

Validation:

- missing field: HTTP 400;
- empty upload: HTTP 400;
- filename must end in `.wav`: HTTP 400;
- malformed/unsupported PCM WAV: HTTP 400;
- upload limit: 50 MiB, enforced before processing;
- model/token/dependency failures: HTTP 503;
- unexpected processing failures: HTTP 500.

Uploaded bytes are written to a temporary `.wav` file and always deleted.

Successful response:

```json
{
  "segments": [
    {
      "start": 1.2,
      "end": 3.4,
      "speaker": "SPEAKER_00",
      "text": "hello how can I help"
    }
  ],
  "processing_ms": 842,
  "audio_seconds": 6.1,
  "rtf": 0.138
}
```

Empty/silent audio returns an empty `segments` array and valid metrics.

Error response:

```json
{"error": "Human-readable message"}
```

No traceback, token, temporary path, or internal model detail is returned.

## Application Service

`SpeakerTranscriptionService` owns one lazily loaded `WhisperTranscriber` and one
lazily loaded `PyannoteDiarizer`. It is shared by the HTTP server so models do
not reload for every upload.

Processing:

1. Validate PCM WAV.
2. Measure audio duration from WAV metadata.
3. Run diarization first so token/access failures happen before ASR.
4. Run timestamped ASR.
5. Fuse words with turns.
6. Group adjacent words by speaker.
7. Return plain dictionaries for JSON serialization.
8. Measure total wall-clock processing time and compute RTF.

Model loading stays lazy: starting the UI or opening `/health` does not require
`HF_TOKEN`.

The service accepts injected transcriber, diarizer, and clock in tests.

## UI

The page contains:

- project title and one-sentence explanation;
- labeled WAV file input;
- Transcribe button;
- status area with `aria-live="polite"`;
- metrics row for processing time, audio duration, and RTF;
- transcript list;
- compact setup note for `HF_TOKEN`.

Speaker styling uses a deterministic class based on label order:

- first observed speaker: blue;
- second observed speaker: amber;
- unknown/additional speakers: neutral.

Each transcript item includes:

- visible speaker label;
- formatted timestamp range;
- transcript text.

The interface has clear focus styles, sufficient contrast, semantic buttons and
labels, and respects `prefers-reduced-motion`.

JavaScript:

1. Blocks submission without a selected WAV.
2. Disables the button while processing.
3. Sends `FormData` to `/api/transcribe`.
4. Handles non-JSON and non-2xx responses safely.
5. Replaces prior results atomically.
6. Restores controls in `finally`.

## CLI

Add:

```bash
HF_TOKEN=... python -m cscall.cli ui \
  --host 127.0.0.1 --port 8000 --model small
```

Options:

- `--host`, default `127.0.0.1`;
- `--port`, default `8000`, range 1–65535;
- `--model`, `--device`, `--compute-type`, and `--language`.

The command prints the local URL and runs until interrupted.

## Error Handling

- Request parsing rejects invalid content type and oversized uploads.
- Temporary files are cleaned in success and failure paths.
- Client disconnects do not crash the server loop.
- Known user-actionable model errors are sanitized into HTTP 503 messages:
  missing `HF_TOKEN`, missing optional dependency, or unaccepted Community-1
  agreement.
- Unexpected exceptions are logged server-side and returned as a generic HTTP
  500 response.
- Concurrent requests are serialized around shared model inference with one
  lock. This is sufficient for a local demo.

## Testing

Tests remain model-free:

- service returns grouped segment dictionaries and correct metrics;
- service runs diarization before ASR;
- service model objects are reused;
- empty audio result is valid;
- multipart parser accepts one WAV and rejects malformed/oversized requests;
- temporary file cleanup occurs after success and exception;
- HTTP routes return correct status, JSON, and content type;
- model/token errors are sanitized;
- HTML includes required accessible controls and no external framework;
- CLI parses defaults and forwards model options;
- existing test suite remains green.

One manual smoke opens the local page and uploads a WAV after optional
dependencies and `HF_TOKEN` are configured.

## Completion Criteria

Phase 4 is complete when:

1. all automated tests pass without `HF_TOKEN`;
2. `/`, `/health`, and `/api/transcribe` work in model-free integration tests;
3. a real configured environment can upload a WAV and display speaker-attributed
   segments;
4. README documents setup and limitations;
5. no uploaded audio, token, or model cache is tracked by Git.

## Deferred Work

- Browser microphone and AudioWorklet capture.
- WebSocket partial/final events.
- Stable online speaker identity across utterances.
- Container packaging and public deployment.
