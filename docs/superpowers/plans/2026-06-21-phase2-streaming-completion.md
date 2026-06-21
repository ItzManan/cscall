# Phase 2 Streaming Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the local WAV streaming demo with PCM energy detection, simple backpressure, aggregate latency benchmarks, explicit language selection, and reproducible documentation.

**Architecture:** Keep the existing `StreamingSession`, endpoint detector, and LocalAgreement implementation. Add one standard-library audio helper module, extend the existing metrics module with aggregate summaries, and keep CLI orchestration in `cscall.cli`; no new runtime dependency is needed.

**Tech Stack:** Python 3.11, standard-library `audioop`/`wave`/`statistics`, existing `faster-whisper`, pytest.

---

## File Structure

```text
src/cscall/streaming/audio.py       # PCM energy classification and WAV validation
src/cscall/streaming/session.py     # existing session plus one-step backpressure
src/cscall/streaming/metrics.py     # existing metrics plus percentile summary
src/cscall/cli.py                   # stream/benchmark wiring and --language
tests/test_streaming_audio.py       # audio helper behavior
tests/test_streaming_session.py     # backpressure behavior
tests/test_streaming_metrics.py     # aggregate percentile behavior
tests/test_streaming_cli.py         # stream/benchmark integration
tests/test_cli.py                   # parser and language forwarding
README.md                           # native and portable run commands
```

### Task 1: Checkpoint the Existing Streaming Foundation

**Files:**
- Add: `src/cscall/streaming/__init__.py`
- Add: `src/cscall/streaming/endpointing.py`
- Add: `src/cscall/streaming/local_agreement.py`
- Add: `src/cscall/streaming/metrics.py`
- Add: `src/cscall/streaming/session.py`
- Add: `tests/test_endpointing.py`
- Add: `tests/test_local_agreement.py`
- Add: `tests/test_streaming_metrics.py`
- Add: `tests/test_streaming_session.py`
- Add: `tests/test_streaming_cli.py`
- Modify: `src/cscall/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `README.md`

- [ ] **Step 1: Verify the inherited foundation**

Run:

```bash
VIRTUAL_ENV="$PWD/.venv" .venv/bin/python -m pytest \
  tests/test_endpointing.py tests/test_local_agreement.py \
  tests/test_streaming_metrics.py tests/test_streaming_session.py \
  tests/test_streaming_cli.py tests/test_cli.py -v
```

Expected: all selected tests pass.

- [ ] **Step 2: Run the fake-transcriber smoke command**

```bash
VIRTUAL_ENV="$PWD/.venv" .venv/bin/python -m cscall.cli stream \
  --audio tests/fixtures/audio/a.wav --fake-transcript "hello world"
```

Expected: output contains `final`, `hello world`, and `Streaming metrics`.

- [ ] **Step 3: Commit only the inherited Phase 2 foundation**

```bash
git add README.md src/cscall/cli.py src/cscall/streaming tests/test_cli.py \
  tests/test_endpointing.py tests/test_local_agreement.py \
  tests/test_streaming_cli.py tests/test_streaming_metrics.py \
  tests/test_streaming_session.py
git commit -m "feat: add local streaming ASR foundation"
```

Do not add `.idea`, `output_model`, `uv.lock`, or personal audio files.

### Task 2: PCM Energy Detection and WAV Validation

**Files:**
- Create: `src/cscall/streaming/audio.py`
- Create: `tests/test_streaming_audio.py`
- Modify: `src/cscall/cli.py`
- Modify: `tests/test_streaming_cli.py`

- [ ] **Step 1: Write failing unit tests**

Add tests equivalent to:

```python
import struct
import wave

import pytest

from cscall.streaming.audio import is_speech_pcm, validate_pcm_wav


def test_energy_detector_distinguishes_silence_and_speech():
    silence = struct.pack("<4h", 0, 0, 0, 0)
    speech = struct.pack("<4h", 2000, -2000, 2000, -2000)
    assert not is_speech_pcm(silence, sample_width=2, threshold=100)
    assert is_speech_pcm(speech, sample_width=2, threshold=100)


def test_energy_detector_honors_threshold():
    audio = struct.pack("<4h", 50, -50, 50, -50)
    assert is_speech_pcm(audio, sample_width=2, threshold=40)
    assert not is_speech_pcm(audio, sample_width=2, threshold=60)


def test_validate_pcm_wav_rejects_compressed_wav(tmp_path):
    path = tmp_path / "bad.wav"
    path.write_bytes(b"not a wav")
    with pytest.raises(ValueError, match="PCM WAV"):
        validate_pcm_wav(path)
```

Add a CLI test that passes a non-WAV file and asserts model construction is not
attempted.

- [ ] **Step 2: Verify RED**

```bash
VIRTUAL_ENV="$PWD/.venv" .venv/bin/python -m pytest \
  tests/test_streaming_audio.py tests/test_streaming_cli.py -v
```

Expected: failure because `cscall.streaming.audio` does not exist.

- [ ] **Step 3: Implement the minimum audio helpers**

Use standard-library PCM RMS calculation. The public API is:

```python
@dataclass(frozen=True)
class WavInfo:
    sample_rate: int
    channels: int
    sample_width: int


def is_speech_pcm(data: bytes, sample_width: int, threshold: int) -> bool:
    ...


def validate_pcm_wav(path: str | Path) -> WavInfo:
    ...
```

`validate_pcm_wav` must open the file with `wave`, require `comptype == "NONE"`,
and wrap `wave.Error`/`EOFError` as `ValueError("<path> is not a supported PCM WAV")`.
`is_speech_pcm` returns `False` for empty data and rejects negative thresholds.

Replace `any(byte != 0 for byte in data)` in `_iter_wav_chunks` with
`is_speech_pcm(data, sampwidth, energy_threshold)`. Add
`--energy-threshold` as a non-negative integer, default `200`.

- [ ] **Step 4: Verify GREEN**

```bash
VIRTUAL_ENV="$PWD/.venv" .venv/bin/python -m pytest \
  tests/test_streaming_audio.py tests/test_streaming_cli.py tests/test_cli.py -v
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cscall/streaming/audio.py src/cscall/cli.py \
  tests/test_streaming_audio.py tests/test_streaming_cli.py tests/test_cli.py
git commit -m "feat: detect speech from PCM energy"
```

### Task 3: One-Step Backpressure

**Files:**
- Modify: `src/cscall/streaming/session.py`
- Modify: `tests/test_streaming_session.py`

- [ ] **Step 1: Write a failing behavior test**

Create a session with `step_ms=100`, `decode_ms=150`, and a long active
utterance. Feed four 100 ms speech chunks and then enough silence to endpoint.
Assert:

- the first scheduled decode runs;
- the next scheduled decode is skipped;
- a later scheduled decode runs;
- endpoint finalization still runs when buffered audio remains.

The expected number of transcriber calls is three: first intermediate, delayed
intermediate, and final endpoint decode.

- [ ] **Step 2: Verify RED**

```bash
VIRTUAL_ENV="$PWD/.venv" .venv/bin/python -m pytest \
  tests/test_streaming_session.py::test_slow_decode_skips_one_intermediate_but_still_finalizes -v
```

Expected: the existing session decodes every interval, so the call count differs.

- [ ] **Step 3: Implement the minimum policy**

Add `_skip_next_decode = False`. After each `_decode`, set it when the current
metrics snapshot has `rtf > 1`. At a normal step boundary, consume the flag and
skip once; do not clear buffered audio. `_finalize` ignores the skip flag and
always decodes remaining buffered audio. Reset the flag in `_begin_utterance`
and after finalization.

- [ ] **Step 4: Verify GREEN**

```bash
VIRTUAL_ENV="$PWD/.venv" .venv/bin/python -m pytest tests/test_streaming_session.py -v
```

Expected: all session tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cscall/streaming/session.py tests/test_streaming_session.py
git commit -m "feat: skip excess streaming re-decodes"
```

### Task 4: Aggregate Metrics and Benchmark Command

**Files:**
- Modify: `src/cscall/streaming/metrics.py`
- Modify: `tests/test_streaming_metrics.py`
- Modify: `src/cscall/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_streaming_cli.py`

- [ ] **Step 1: Write failing percentile tests**

Add:

```python
from cscall.streaming.metrics import summarize_metrics


def test_summary_percentiles_are_deterministic():
    values = [
        StreamingMetrics(audio_ms=1000, decode_ms=100, first_partial_latency_ms=100, final_latency_ms=10),
        StreamingMetrics(audio_ms=1000, decode_ms=200, first_partial_latency_ms=200, final_latency_ms=20),
        StreamingMetrics(audio_ms=1000, decode_ms=300, first_partial_latency_ms=300, final_latency_ms=30),
    ]
    summary = summarize_metrics(values)
    assert summary["rtf"]["p50"] == 0.2
    assert summary["rtf"]["p99"] == 0.3
    assert summary["first_partial_ms"]["p50"] == 200
    assert summary["final_ms"]["p99"] == 30


def test_summary_uses_none_for_empty_samples():
    summary = summarize_metrics([])
    assert summary["rtf"] == {"p50": None, "p99": None}
```

Add parser and integration tests for:

```bash
cscall benchmark --audio a.wav b.wav --fake-transcript hello
```

The deterministic output contains a Markdown header and rows for `RTF`,
`first_partial_ms`, and `final_ms`.

- [ ] **Step 2: Verify RED**

```bash
VIRTUAL_ENV="$PWD/.venv" .venv/bin/python -m pytest \
  tests/test_streaming_metrics.py tests/test_streaming_cli.py tests/test_cli.py -v
```

Expected: failures because the summary and command do not exist.

- [ ] **Step 3: Implement summary and benchmark**

`summarize_metrics(metrics)` returns:

```python
{
    "rtf": {"p50": float | None, "p99": float | None},
    "first_partial_ms": {"p50": int | float | None, "p99": int | float | None},
    "final_ms": {"p50": int | float | None, "p99": int | float | None},
}
```

Use sorted values and nearest-rank p99 (`ceil(0.99 * n) - 1`); use
`statistics.median` for p50. Ignore `None` latency values.

Factor the current single-file stream loop into a helper that returns emitted
events. `benchmark` accepts one or more `--audio` paths and the same model,
device, compute type, language, chunk, agreement, threshold, and fake transcript
options as `stream`. Collect metrics events and render:

```text
| Metric | p50 | p99 |
|---|---:|---:|
| RTF | ... | ... |
| first_partial_ms | ... | ... |
| final_ms | ... | ... |
```

- [ ] **Step 4: Verify GREEN**

```bash
VIRTUAL_ENV="$PWD/.venv" .venv/bin/python -m pytest \
  tests/test_streaming_metrics.py tests/test_streaming_cli.py tests/test_cli.py -v
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cscall/streaming/metrics.py src/cscall/cli.py \
  tests/test_streaming_metrics.py tests/test_streaming_cli.py tests/test_cli.py
git commit -m "feat: benchmark streaming latency"
```

### Task 5: Explicit Language, Documentation, and Final Verification

**Files:**
- Modify: `src/cscall/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_streaming_cli.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing forwarding tests**

For `baseline`, `compare`, `stream`, and `benchmark`, parse `--language hi` and
assert `args.language == "hi"`. Add monkeypatched CLI tests asserting the
constructed `WhisperTranscriber` receives `language="hi"` for baseline, compare,
and stream. Benchmark shares stream construction and needs one forwarding test.

- [ ] **Step 2: Verify RED**

```bash
VIRTUAL_ENV="$PWD/.venv" .venv/bin/python -m pytest \
  tests/test_cli.py tests/test_streaming_cli.py -v
```

Expected: parser rejects `--language` or forwarding assertions fail.

- [ ] **Step 3: Implement language forwarding**

Add `--language`, default `None`, to all four commands. Pass it unchanged to
every `WhisperTranscriber` construction.

- [ ] **Step 4: Update README**

Document:

```bash
# Fast model-free smoke test
python -m cscall.cli stream --audio tests/fixtures/audio/a.wav \
  --fake-transcript "hello world"

# Native Mac demo; use tiny if small has RTF > 1
python -m cscall.cli stream --audio call.wav --model small

# Portable CPU benchmark/default intended for Docker
python -m cscall.cli benchmark --audio call.wav --model tiny \
  --device cpu --compute-type int8

# Reproducible forced-language experiment
python -m cscall.cli stream --audio call.wav --language hi
```

State that automatic detection remains the default and that Docker packaging is
Phase 5.

- [ ] **Step 5: Run all automated verification**

```bash
VIRTUAL_ENV="$PWD/.venv" .venv/bin/python -m pytest -q
git diff --check
```

Expected: all tests pass and `git diff --check` prints nothing.

- [ ] **Step 6: Run fake integration commands**

```bash
VIRTUAL_ENV="$PWD/.venv" .venv/bin/python -m cscall.cli stream \
  --audio tests/fixtures/audio/a.wav --fake-transcript "hello world"
VIRTUAL_ENV="$PWD/.venv" .venv/bin/python -m cscall.cli benchmark \
  --audio tests/fixtures/audio/a.wav --fake-transcript "hello world"
```

Expected: stream output has final text and metrics; benchmark output has a
Markdown p50/p99 table.

- [ ] **Step 7: Run the real-model smoke test**

```bash
VIRTUAL_ENV="$PWD/.venv" .venv/bin/python -m cscall.cli stream \
  --audio tests/fixtures/audio/a.wav --model tiny
```

Expected: command completes without an exception. Transcript may be empty for
the synthetic fixture.

- [ ] **Step 8: Commit**

```bash
git add README.md src/cscall/cli.py tests/test_cli.py tests/test_streaming_cli.py
git commit -m "docs: finish phase 2 streaming workflow"
```

## Plan Self-Review

- Spec coverage: energy VAD, validation, backpressure, p50/p99 benchmark,
  language regime, native/portable instructions, fake and real smoke checks are
  each mapped to a task.
- Scope: no diarization, UI, WebSocket, Docker image, resampling, or Silero work.
- Type consistency: benchmark consumes existing `StreamingMetrics`; all commands
  use the existing `WhisperTranscriber(language=...)` constructor.
- Placeholder scan: no deferred implementation instructions or unspecified test
  behavior remain.
