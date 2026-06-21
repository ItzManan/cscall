# Phase 3 Diarization and Speaker Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional two-speaker pyannote diarization, timestamped Whisper words, deterministic word-to-speaker fusion, RTTM/DER evaluation, and speaker-attributed CLI output.

**Architecture:** Keep optional third-party objects behind a lazy diarization adapter. Convert all external outputs into small immutable project records, then run pure fusion/rendering logic that remains fully testable without model downloads.

**Tech Stack:** Python 3.11, existing faster-whisper, optional `pyannote.audio>=4`, pytest.

---

## File Structure

```text
src/cscall/diarization.py       # SpeakerTurn, lazy Community-1 adapter, RTTM/DER
src/cscall/fusion.py            # TimedWord/SpeakerWord assignment and rendering
src/cscall/asr_baseline.py      # existing transcriber plus word timestamps
src/cscall/cli.py               # diarize and transcribe-speakers commands
tests/test_diarization.py       # fake-pipeline adapter and RTTM tests
tests/test_fusion.py            # pure overlap/tie/grouping tests
tests/test_asr_baseline.py      # word timestamp conversion
tests/test_speaker_cli.py       # model-free CLI integration
pyproject.toml                  # optional diarization dependency
README.md                       # token/model agreement and command docs
```

### Task 1: Speaker Data Models and Pure Fusion

**Files:**
- Create: `src/cscall/fusion.py`
- Create: `tests/test_fusion.py`

- [ ] **Step 1: Write failing model and assignment tests**

Cover:

```python
def test_assigns_word_to_turn_with_maximum_overlap():
    turns = [
        SpeakerTurn(0.0, 1.5, "SPEAKER_00"),
        SpeakerTurn(1.5, 3.0, "SPEAKER_01"),
    ]
    words = [TimedWord(1.25, 2.0, "hello")]
    assert fuse_words(words, turns)[0].speaker == "SPEAKER_01"


def test_equal_overlap_uses_earliest_turn_then_label():
    ...


def test_zero_overlap_uses_nearest_turn_to_word_midpoint():
    ...


def test_no_turns_assigns_unknown():
    ...


def test_grouping_renders_deterministic_lines():
    ...
```

Also test negative timestamps and `end < start` rejection for all three records.

- [ ] **Step 2: Verify RED**

```bash
VIRTUAL_ENV="$PWD/.venv" .venv/bin/python -m pytest tests/test_fusion.py -v
```

Expected: import failure because `cscall.fusion` does not exist.

- [ ] **Step 3: Implement minimum pure logic**

Implement immutable `SpeakerTurn`, `TimedWord`, and `SpeakerWord`; `fuse_words`;
`group_speaker_words`; and `render_speaker_transcript`.

Fusion is an O(words × turns) scan. Tie order is greatest overlap, earliest turn
start, then speaker label. For zero overlap use interval distance from the word
midpoint; if still tied use the same deterministic order.

Render times as `MM:SS.ss`:

```text
[00:01.20–00:03.40] SPEAKER_00: hello there
```

- [ ] **Step 4: Verify GREEN**

```bash
VIRTUAL_ENV="$PWD/.venv" .venv/bin/python -m pytest tests/test_fusion.py -v
```

Expected: all fusion tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cscall/fusion.py tests/test_fusion.py
git commit -m "feat: fuse timestamped words with speaker turns"
```

### Task 2: Timestamped Whisper Words

**Files:**
- Modify: `src/cscall/asr_baseline.py`
- Modify: `tests/test_asr_baseline.py`

- [ ] **Step 1: Write failing conversion tests**

Inject a fake model or construct the transcriber without running its normal
constructor. Verify `transcribe_words(path)`:

- calls model transcription with `word_timestamps=True` and configured language;
- flattens words across segments in order;
- strips whitespace;
- skips empty text and words missing start/end timestamps;
- returns `TimedWord` records.

- [ ] **Step 2: Verify RED**

```bash
VIRTUAL_ENV="$PWD/.venv" .venv/bin/python -m pytest tests/test_asr_baseline.py -v
```

Expected: failure because `transcribe_words` does not exist.

- [ ] **Step 3: Implement**

Add only:

```python
def transcribe_words(self, audio_path: str) -> list[TimedWord]:
    segments, _ = self._model.transcribe(
        audio_path,
        language=self._language,
        word_timestamps=True,
    )
    ...
```

Do not change existing `transcribe`.

- [ ] **Step 4: Verify GREEN and commit**

```bash
VIRTUAL_ENV="$PWD/.venv" .venv/bin/python -m pytest tests/test_asr_baseline.py -v
git add src/cscall/asr_baseline.py tests/test_asr_baseline.py
git commit -m "feat: expose Whisper word timestamps"
```

### Task 3: Lazy Community-1 Diarization Adapter

**Files:**
- Create: `src/cscall/diarization.py`
- Create: `tests/test_diarization.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write failing adapter tests**

Use fake pipeline/output/annotation objects. Cover:

- missing `HF_TOKEN` fails before loading;
- injected pipeline does not require a token or pyannote import;
- inference receives `num_speakers=2`;
- exclusive diarization is preferred;
- regular diarization is fallback;
- turns are converted and deterministically sorted;
- inaccessible model exceptions include Community-1 acceptance guidance;
- missing optional dependency includes `pip install -e ".[diarization]"`.

- [ ] **Step 2: Verify RED**

```bash
VIRTUAL_ENV="$PWD/.venv" .venv/bin/python -m pytest tests/test_diarization.py -v
```

Expected: import failure.

- [ ] **Step 3: Implement adapter**

Public API:

```python
COMMUNITY_MODEL = "pyannote/speaker-diarization-community-1"


class PyannoteDiarizer:
    def __init__(self, pipeline=None, token: str | None = None):
        ...

    def diarize(self, audio_path: str) -> list[SpeakerTurn]:
        ...
```

When no pipeline is injected, resolve token from explicit constructor value or
`HF_TOKEN`, lazily import `Pipeline`, and call
`Pipeline.from_pretrained(COMMUNITY_MODEL, token=token)`.

Support annotation iteration through `itertracks(yield_label=True)` and the
Community-1 iterable form `(turn, speaker)`.

Add:

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0.0"]
diarization = ["pyannote.audio>=4,<5"]
```

- [ ] **Step 4: Verify GREEN and dependency metadata**

```bash
VIRTUAL_ENV="$PWD/.venv" .venv/bin/python -m pytest tests/test_diarization.py -v
VIRTUAL_ENV="$PWD/.venv" .venv/bin/python -m pytest tests/test_smoke.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/cscall/diarization.py tests/test_diarization.py pyproject.toml
git commit -m "feat: add optional Community-1 diarization"
```

### Task 4: RTTM Parsing and DER

**Files:**
- Modify: `src/cscall/diarization.py`
- Modify: `tests/test_diarization.py`

- [ ] **Step 1: Write failing RTTM tests**

Cover:

- valid RTTM speaker rows become sorted `SpeakerTurn` values;
- blank/comment rows are ignored;
- invalid field count, non-numeric time, negative start, and non-positive duration
  raise `ValueError` with `path:line`;
- `diarization_error_rate(reference, hypothesis)` uses injected metric/annotation
  factories in unit tests and returns a float.

- [ ] **Step 2: Verify RED**

```bash
VIRTUAL_ENV="$PWD/.venv" .venv/bin/python -m pytest tests/test_diarization.py -v
```

- [ ] **Step 3: Implement**

Add:

```python
def load_rttm(path: str) -> list[SpeakerTurn]:
    ...


def diarization_error_rate(
    reference: list[SpeakerTurn],
    hypothesis: list[SpeakerTurn],
    annotation_factory=None,
    segment_factory=None,
    metric=None,
) -> float:
    ...
```

Production defaults lazily import `pyannote.core.Annotation`,
`pyannote.core.Segment`, and
`pyannote.metrics.diarization.DiarizationErrorRate(collar=0.0,
skip_overlap=False)`.

- [ ] **Step 4: Verify and commit**

```bash
VIRTUAL_ENV="$PWD/.venv" .venv/bin/python -m pytest tests/test_diarization.py -v
git add src/cscall/diarization.py tests/test_diarization.py
git commit -m "feat: evaluate diarization from RTTM"
```

### Task 5: Diarization and Speaker Transcript CLI

**Files:**
- Modify: `src/cscall/cli.py`
- Create: `tests/test_speaker_cli.py`

- [ ] **Step 1: Write failing parser/integration tests**

Cover:

- `diarize --audio call.wav [--reference-rttm ref.rttm]`;
- `transcribe-speakers --audio call.wav` with model/device/compute/language options;
- WAV validation happens before diarizer or ASR construction;
- fake/injected diarizer output prints deterministic tab-separated turns;
- supplied RTTM prints `DER: 12.34%`;
- timestamped ASR and turns are fused and rendered;
- silent/no-turn/no-word cases print no transcript and do not crash.

Monkeypatch project adapters; never import pyannote.

- [ ] **Step 2: Verify RED**

```bash
VIRTUAL_ENV="$PWD/.venv" .venv/bin/python -m pytest tests/test_speaker_cli.py -v
```

- [ ] **Step 3: Implement minimum CLI wiring**

Add parsers and runners. Both validate PCM WAV first.

`diarize` constructs `PyannoteDiarizer`, prints each turn:

```text
1.200	3.400	SPEAKER_00
```

When RTTM is supplied, load it and print DER with two decimals.

`transcribe-speakers` constructs one `WhisperTranscriber` and one
`PyannoteDiarizer`, calls `transcribe_words` and `diarize`, then prints
`render_speaker_transcript(...)`.

- [ ] **Step 4: Verify and commit**

```bash
VIRTUAL_ENV="$PWD/.venv" .venv/bin/python -m pytest \
  tests/test_speaker_cli.py tests/test_cli.py -v
git add src/cscall/cli.py tests/test_speaker_cli.py
git commit -m "feat: add speaker-attributed transcription CLI"
```

### Task 6: Documentation and Real Smoke Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document setup**

Add:

```bash
pip install -e ".[dev,diarization]"
export HF_TOKEN=hf_...
python -m cscall.cli diarize --audio call.wav
python -m cscall.cli transcribe-speakers --audio call.wav --model small
```

Link the Community-1 model page and state that its agreement must be accepted.
State that Phase 3 is file/utterance-level offline diarization; rolling online
speaker identity remains for the live service phase.

- [ ] **Step 2: Run complete automated verification**

```bash
VIRTUAL_ENV="$PWD/.venv" .venv/bin/python -m pytest -q
VIRTUAL_ENV="$PWD/.venv" .venv/bin/python -m compileall -q src data
git diff --check
```

- [ ] **Step 3: Run model-free CLI checks**

Use monkeypatch tests as the required automated integration. Confirm help works:

```bash
VIRTUAL_ENV="$PWD/.venv" .venv/bin/python -m cscall.cli diarize --help
VIRTUAL_ENV="$PWD/.venv" .venv/bin/python -m cscall.cli transcribe-speakers --help
```

- [ ] **Step 4: Attempt real Community-1 smoke**

Only if `HF_TOKEN` is available in the environment and optional dependencies are
installed:

```bash
VIRTUAL_ENV="$PWD/.venv" .venv/bin/python -m cscall.cli diarize \
  --audio "tests/fixtures/audio/New Recording.m4a"
```

This fixture is not a WAV, so use a real user-provided PCM WAV instead if one is
available. Do not convert or commit personal audio. If no suitable WAV/token or
dependency exists, report the exact external prerequisite rather than claiming
the smoke passed.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: explain speaker diarization workflow"
```

## Plan Self-Review

- Spec coverage: data records, Community-1 adapter, timestamped words, fusion,
  both CLI commands, RTTM/DER, docs, token handling, and smoke verification each
  map to a task.
- Scope: no online clustering, role identification, browser, service, or Docker.
- Security: token is environment-only and never printed or persisted.
- Optionality: default tests and existing commands do not import pyannote.
- Placeholder scan: every code task has a defined API, test behavior, command,
  and expected result.
