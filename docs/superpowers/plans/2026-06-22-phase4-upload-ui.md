# Phase 4 Upload Transcript UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve a dependency-free local upload page that returns and renders speaker-attributed transcript segments from the Phase 3 pipeline.

**Architecture:** Add one reusable transcription service and one standard-library HTTP module containing the embedded page and request handler. Keep model loading lazy and serialize inference with a lock; Phase 5 can replace transport while reusing the service/JSON shape.

**Tech Stack:** Python 3.11 standard library, existing faster-whisper/optional pyannote, pytest.

---

### Task 1: Reusable Speaker Transcription Service

**Files:**
- Create: `src/cscall/webapp.py`
- Create: `tests/test_webapp.py`

- [ ] Write failing tests for injected transcriber/diarizer, diarization-before-ASR
  ordering, grouped JSON segments, empty results, duration/RTF, lazy factory reuse,
  and one-lock serialization.
- [ ] Run `pytest tests/test_webapp.py -v` and verify RED.
- [ ] Implement `SpeakerTranscriptionService` with injected instances/factories,
  clock, and lock.
- [ ] Return:

```python
{
    "segments": [{"start": ..., "end": ..., "speaker": ..., "text": ...}],
    "processing_ms": int,
    "audio_seconds": float,
    "rtf": float,
}
```

- [ ] Verify focused/full tests and commit:

```bash
git commit -m "feat: add reusable speaker transcription service"
```

### Task 2: Safe Multipart HTTP Endpoint

**Files:**
- Modify: `src/cscall/webapp.py`
- Modify: `tests/test_webapp.py`

- [ ] Write failing tests for multipart WAV extraction, missing field, invalid
  content type, 50 MiB limit, temporary cleanup, `/health`, successful JSON,
  400/503/500 sanitization, and shared service use.
- [ ] Verify RED.
- [ ] Implement multipart parsing with `email.parser.BytesParser`, a
  `BaseHTTPRequestHandler` subclass factory, `ThreadingHTTPServer`, JSON helpers,
  and temporary `.wav` cleanup.
- [ ] Never include traceback, token, or temp path in responses.
- [ ] Verify focused/full tests and commit:

```bash
git commit -m "feat: serve uploaded WAV transcription API"
```

### Task 3: Accessible Embedded UI and CLI

**Files:**
- Modify: `src/cscall/webapp.py`
- Modify: `src/cscall/cli.py`
- Modify: `tests/test_webapp.py`
- Modify: `tests/test_cli.py`

- [ ] Write failing tests that HTML includes labeled WAV input, semantic button,
  aria-live status, metrics, transcript container, focus/reduced-motion CSS,
  fetch/FormData logic, and no external framework/script.
- [ ] Write failing CLI parser/forwarding tests for:

```bash
python -m cscall.cli ui --host 127.0.0.1 --port 8000 \
  --model small --device cpu --compute-type int8 --language hi
```

- [ ] Validate port 1–65535 during parsing.
- [ ] Implement embedded HTML and `run_server(...)`; add `ui` CLI wiring.
- [ ] Verify tests/help and commit:

```bash
git commit -m "feat: add upload transcript web UI"
```

### Task 4: Documentation and Final Verification

**Files:**
- Modify: `README.md`

- [ ] Document optional install, `HF_TOKEN`, `ui` command, upload workflow, and
  offline/file-level limitation.
- [ ] Run:

```bash
VIRTUAL_ENV="$PWD/.venv" .venv/bin/python -m pytest -q
VIRTUAL_ENV="$PWD/.venv" .venv/bin/python -m compileall -q src data
VIRTUAL_ENV="$PWD/.venv" .venv/bin/python -m cscall.cli ui --help
git diff --check
```

- [ ] Run a model-free HTTP smoke from tests. Attempt a real browser/upload smoke
  only when `HF_TOKEN` and `pyannote.audio` are available; otherwise report the
  exact prerequisite.
- [ ] Commit:

```bash
git commit -m "docs: explain upload transcript UI"
```

## Plan Self-Review

- Covers the service, API, accessibility, CLI, docs, cleanup, size limit, lazy
  loading, and JSON contract.
- Adds no runtime dependency or build system.
- Does not implement microphone, WebSocket, Docker, auth, or persistence.
- All non-trivial logic has model-free tests.
