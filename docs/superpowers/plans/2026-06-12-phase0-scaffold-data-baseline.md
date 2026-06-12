# Phase 0: Scaffold + Data + Baseline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the project skeleton, a typed data/manifest layer, a code-switch-aware metrics harness, and a baseline (vanilla Whisper) evaluation runner that produces WER/CER numbers — the locked-in "before" measurement.

**Architecture:** A small Python package (`cscall`) with pure, unit-tested core modules (text normalization, manifest schema, metrics, eval runner with an injectable transcribe function) plus one integration module wrapping `faster-whisper`. Data acquisition (public benchmarks + self-recorded corpus) is documented with scripts; actual large downloads/recording are manual steps run by the human, while all code is tested against tiny committed fixtures.

**Tech Stack:** Python 3.11, `faster-whisper` (CTranslate2 Whisper runtime), `jiwer` (WER/CER), `indic-transliteration` (Devanagari→roman normalization), `pytest`. Inference on Apple Silicon (CPU/int8); no GPU needed for Phase 0.

---

## File Structure

```
pyproject.toml                     # package + deps + pytest config
src/cscall/__init__.py             # package marker, version
src/cscall/normalize.py            # code-switch text normalization (pure, TDD)
src/cscall/manifest.py             # Utterance dataclass + JSONL loader (pure, TDD)
src/cscall/metrics.py              # WER/CER over jiwer, post-normalization (pure, TDD)
src/cscall/eval_runner.py          # manifest + transcribe_fn -> EvalReport + markdown (TDD via fake)
src/cscall/asr_baseline.py         # faster-whisper wrapper (integration)
src/cscall/cli.py                  # `python -m cscall.cli baseline ...` entrypoint
data/README.md                     # how to obtain Svarah/MUCS (+fallback) and record the corpus
data/download_svarah.py            # download helper (arg-parsed, dry-run testable)
tests/test_normalize.py
tests/test_manifest.py
tests/test_metrics.py
tests/test_eval_runner.py
tests/test_download_svarah.py
tests/fixtures/mini_manifest.jsonl # 3-row fake manifest pointing at tiny wavs
tests/fixtures/audio/*.wav         # 2-3 ~1s wavs (silence/tone) committed for smoke tests
README.md                          # project framing + how to run baseline
```

**Responsibilities:** `normalize` = text canonicalization only. `manifest` = data schema + IO only. `metrics` = scoring only (depends on `normalize`). `eval_runner` = orchestration only (depends on `manifest`, `metrics`; ASR injected). `asr_baseline` = the only module that touches `faster-whisper`. `cli` = wiring for humans.

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `src/cscall/__init__.py`
- Create: `tests/test_smoke.py`
- Create: `README.md`
- Create: `.python-version`

- [ ] **Step 1: Write the failing test**

`tests/test_smoke.py`:
```python
def test_package_imports():
    import cscall

    assert cscall.__version__ == "0.0.1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_smoke.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cscall'`

- [ ] **Step 3: Create the package and config**

`pyproject.toml`:
```toml
[project]
name = "cscall"
version = "0.0.1"
description = "Real-time speaker-attributed code-switch ASR for support calls"
requires-python = ">=3.11"
dependencies = [
    "faster-whisper>=1.0.0",
    "jiwer>=3.0.0",
    "indic-transliteration>=2.3.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0.0"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

`src/cscall/__init__.py`:
```python
__version__ = "0.0.1"
```

`.python-version`:
```
3.11
```

`README.md`:
```markdown
# cscall — Real-Time Speaker-Attributed Code-Switch ASR

Streaming, speaker-attributed transcription for Indian-accented, Hindi-English
code-switched support calls. See `docs/superpowers/specs/` for the design.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Phase 0: baseline

See `data/README.md` to obtain evaluation audio, then:

```bash
python -m cscall.cli baseline --manifest data/manifests/svarah.jsonl --model small
```
```

- [ ] **Step 4: Install and run the test**

Run: `pip install -e ".[dev]" && pytest tests/test_smoke.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/cscall/__init__.py tests/test_smoke.py README.md .python-version
git commit -m "feat: project scaffold for cscall package"
```

---

### Task 2: Code-switch text normalization

**Files:**
- Create: `src/cscall/normalize.py`
- Create: `tests/test_normalize.py`

- [ ] **Step 1: Write the failing test**

`tests/test_normalize.py`:
```python
from cscall.normalize import normalize_text, romanize_devanagari


def test_lowercases_and_strips_punctuation():
    assert normalize_text("Hello, World!!") == "hello world"


def test_collapses_whitespace():
    assert normalize_text("  too   many\tspaces  ") == "too many spaces"


def test_romanizes_devanagari_word():
    # मेरा -> "meraa" in ITRANS-style romanization (lowercased)
    out = romanize_devanagari("मेरा")
    assert "mer" in out
    assert out == out.lower()


def test_normalize_text_romanizes_mixed_script():
    # "order kahan hai" written half in Devanagari should romanize then normalize
    out = normalize_text("order कहां hai")
    assert "order" in out
    assert "hai" in out
    # the Devanagari word becomes ascii letters only
    assert all(ord(c) < 128 for c in out)


def test_empty_string():
    assert normalize_text("") == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_normalize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cscall.normalize'`

- [ ] **Step 3: Write minimal implementation**

`src/cscall/normalize.py`:
```python
"""Text normalization for code-switch WER scoring.

The eval set mixes romanized Hindi, Devanagari, and English. To score fairly we
canonicalize everything to a single lowercase, punctuation-free, romanized form
BEFORE computing WER/CER, so "कहां" and "kahan" are not counted as errors.
"""
import re
import unicodedata

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")


def romanize_devanagari(text: str) -> str:
    """Transliterate any Devanagari runs to ASCII (ITRANS), leaving other text."""
    if not _DEVANAGARI_RE.search(text):
        return text.lower()
    romanized = transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
    return romanized.lower()


def normalize_text(text: str) -> str:
    """Canonicalize text for WER/CER: romanize, lowercase, strip punctuation, collapse ws."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = romanize_devanagari(text)
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_normalize.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/cscall/normalize.py tests/test_normalize.py
git commit -m "feat: code-switch text normalization for fair WER scoring"
```

---

### Task 3: Manifest schema + loader

**Files:**
- Create: `src/cscall/manifest.py`
- Create: `tests/test_manifest.py`
- Create: `tests/fixtures/mini_manifest.jsonl`

- [ ] **Step 1: Write the failing test**

`tests/fixtures/mini_manifest.jsonl`:
```json
{"id": "u1", "audio_path": "tests/fixtures/audio/a.wav", "text": "order kahan hai", "speaker": "customer", "lang": "hi-en", "accent": "north", "cs_density": 0.5}
{"id": "u2", "audio_path": "tests/fixtures/audio/b.wav", "text": "your refund is processed", "speaker": "agent", "lang": "en", "accent": "north", "cs_density": 0.0}
{"id": "u3", "audio_path": "tests/fixtures/audio/c.wav", "text": "thoda wait karo please", "speaker": "agent", "lang": "hi-en", "accent": "south", "cs_density": 0.6}
```

`tests/test_manifest.py`:
```python
import pytest

from cscall.manifest import Utterance, load_manifest


def test_loads_all_rows():
    utts = load_manifest("tests/fixtures/mini_manifest.jsonl")
    assert len(utts) == 3
    assert all(isinstance(u, Utterance) for u in utts)


def test_fields_parsed():
    utts = load_manifest("tests/fixtures/mini_manifest.jsonl")
    u = utts[0]
    assert u.id == "u1"
    assert u.text == "order kahan hai"
    assert u.speaker == "customer"
    assert u.cs_density == 0.5


def test_missing_required_field_raises():
    import tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write('{"id": "x", "text": "no audio path"}\n')
        path = f.name
    try:
        with pytest.raises(ValueError, match="audio_path"):
            load_manifest(path)
    finally:
        os.unlink(path)


def test_optional_fields_default():
    import tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write('{"id": "y", "audio_path": "p.wav", "text": "hello"}\n')
        path = f.name
    try:
        u = load_manifest(path)[0]
        assert u.speaker is None
        assert u.cs_density is None
    finally:
        os.unlink(path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_manifest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cscall.manifest'`

- [ ] **Step 3: Write minimal implementation**

`src/cscall/manifest.py`:
```python
"""Evaluation manifest: one JSONL row per utterance."""
import json
from dataclasses import dataclass
from typing import Optional

_REQUIRED = ("id", "audio_path", "text")


@dataclass
class Utterance:
    id: str
    audio_path: str
    text: str
    speaker: Optional[str] = None
    lang: Optional[str] = None
    accent: Optional[str] = None
    cs_density: Optional[float] = None


def load_manifest(path: str) -> list[Utterance]:
    """Load a JSONL manifest into Utterance objects, validating required fields."""
    utts: list[Utterance] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for key in _REQUIRED:
                if key not in row:
                    raise ValueError(f"{path}:{lineno} missing required field '{key}'")
            utts.append(
                Utterance(
                    id=row["id"],
                    audio_path=row["audio_path"],
                    text=row["text"],
                    speaker=row.get("speaker"),
                    lang=row.get("lang"),
                    accent=row.get("accent"),
                    cs_density=row.get("cs_density"),
                )
            )
    return utts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_manifest.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/cscall/manifest.py tests/test_manifest.py tests/fixtures/mini_manifest.jsonl
git commit -m "feat: typed JSONL manifest loader with validation"
```

---

### Task 4: Metrics (WER/CER)

**Files:**
- Create: `src/cscall/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

`tests/test_metrics.py`:
```python
from cscall.metrics import score


def test_perfect_match_zero_wer():
    r = score(["hello world"], ["hello world"])
    assert r["wer"] == 0.0
    assert r["cer"] == 0.0
    assert r["n"] == 1


def test_one_substitution():
    # 2 words, 1 wrong -> WER 0.5
    r = score(["hello world"], ["hello there"])
    assert r["wer"] == 0.5


def test_normalization_ignores_script_and_case():
    # reference in Devanagari, hyp romanized + capitalized -> should match
    r = score(["कहां hai"], ["Kahan hai"])
    assert r["wer"] == 0.0


def test_multiple_utterances_aggregate():
    r = score(["a b", "c d"], ["a b", "c x"])
    assert r["n"] == 2
    assert r["wer"] == 0.25  # 1 error / 4 words


def test_mismatched_lengths_raises():
    import pytest
    with pytest.raises(ValueError):
        score(["a"], ["a", "b"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cscall.metrics'`

- [ ] **Step 3: Write minimal implementation**

`src/cscall/metrics.py`:
```python
"""WER/CER scoring, applied AFTER code-switch normalization."""
import jiwer

from cscall.normalize import normalize_text


def score(references: list[str], hypotheses: list[str]) -> dict:
    """Compute corpus WER/CER over normalized text.

    Returns {"wer": float, "cer": float, "n": int}.
    """
    if len(references) != len(hypotheses):
        raise ValueError(
            f"reference/hypothesis count mismatch: {len(references)} vs {len(hypotheses)}"
        )
    refs = [normalize_text(r) for r in references]
    hyps = [normalize_text(h) for h in hypotheses]
    return {
        "wer": jiwer.wer(refs, hyps),
        "cer": jiwer.cer(refs, hyps),
        "n": len(refs),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_metrics.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/cscall/metrics.py tests/test_metrics.py
git commit -m "feat: WER/CER scoring over normalized code-switch text"
```

---

### Task 5: Eval runner (with injectable transcriber)

**Files:**
- Create: `src/cscall/eval_runner.py`
- Create: `tests/test_eval_runner.py`

- [ ] **Step 1: Write the failing test**

`tests/test_eval_runner.py`:
```python
from cscall.eval_runner import run_eval, render_markdown
from cscall.manifest import load_manifest


def fake_transcriber(audio_path: str) -> str:
    # Deterministic fake keyed by file name; pretends to mis-hear u3.
    return {
        "tests/fixtures/audio/a.wav": "order kahan hai",
        "tests/fixtures/audio/b.wav": "your refund is processed",
        "tests/fixtures/audio/c.wav": "thoda wait karo",  # dropped "please"
    }[audio_path]


def test_run_eval_overall_metrics():
    utts = load_manifest("tests/fixtures/mini_manifest.jsonl")
    report = run_eval(utts, fake_transcriber)
    assert report["overall"]["n"] == 3
    # 1 deleted word out of 11 total reference words
    assert report["overall"]["wer"] > 0.0


def test_run_eval_groups_by_accent():
    utts = load_manifest("tests/fixtures/mini_manifest.jsonl")
    report = run_eval(utts, fake_transcriber, group_by="accent")
    assert set(report["groups"].keys()) == {"north", "south"}
    assert report["groups"]["north"]["wer"] == 0.0  # north utts transcribed perfectly


def test_render_markdown_contains_table():
    utts = load_manifest("tests/fixtures/mini_manifest.jsonl")
    report = run_eval(utts, fake_transcriber, group_by="accent")
    md = render_markdown(report)
    assert "| WER |" in md
    assert "north" in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cscall.eval_runner'`

- [ ] **Step 3: Write minimal implementation**

`src/cscall/eval_runner.py`:
```python
"""Run a transcriber over a manifest and compute baseline metrics.

The transcriber is injected as a callable (audio_path -> text) so this module is
testable without loading a model and so the same runner serves baseline and
fine-tuned models in later phases.
"""
from collections import defaultdict
from typing import Callable, Optional

from cscall.manifest import Utterance
from cscall.metrics import score

Transcriber = Callable[[str], str]


def run_eval(
    utterances: list[Utterance],
    transcribe: Transcriber,
    group_by: Optional[str] = None,
) -> dict:
    """Transcribe every utterance, return overall (and optionally grouped) metrics."""
    refs = [u.text for u in utterances]
    hyps = [transcribe(u.audio_path) for u in utterances]
    report = {"overall": score(refs, hyps)}

    if group_by:
        buckets: dict[str, list[int]] = defaultdict(list)
        for i, u in enumerate(utterances):
            key = getattr(u, group_by)
            buckets[str(key)].append(i)
        report["groups"] = {
            key: score([refs[i] for i in idxs], [hyps[i] for i in idxs])
            for key, idxs in buckets.items()
        }
    return report


def render_markdown(report: dict) -> str:
    """Render a report dict as a markdown results table."""
    lines = ["| Group | WER | CER | N |", "|---|---|---|---|"]
    o = report["overall"]
    lines.append(f"| **overall** | {o['wer']:.3f} | {o['cer']:.3f} | {o['n']} |")
    for key, m in sorted(report.get("groups", {}).items()):
        lines.append(f"| {key} | {m['wer']:.3f} | {m['cer']:.3f} | {m['n']} |")
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_eval_runner.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/cscall/eval_runner.py tests/test_eval_runner.py
git commit -m "feat: eval runner with injectable transcriber and grouped metrics"
```

---

### Task 6: Baseline ASR wrapper (faster-whisper)

**Files:**
- Create: `src/cscall/asr_baseline.py`
- Create: `tests/fixtures/audio/a.wav` (and `b.wav`, `c.wav`)
- Create: `tests/test_asr_baseline.py`

- [ ] **Step 1: Generate tiny audio fixtures**

Run (creates three ~1s 16kHz mono wavs so smoke tests have real audio):
```bash
python - <<'PY'
import wave, struct, math, os
os.makedirs("tests/fixtures/audio", exist_ok=True)
def tone(path, freq):
    sr=16000; n=sr
    with wave.open(path,"w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        for i in range(n):
            w.writeframes(struct.pack("<h", int(3000*math.sin(2*math.pi*freq*i/sr))))
for name,f in [("a",220),("b",330),("c",440)]:
    tone(f"tests/fixtures/audio/{name}.wav", f)
print("wrote fixtures")
PY
```

- [ ] **Step 2: Write the failing test**

`tests/test_asr_baseline.py`:
```python
import pytest

from cscall.asr_baseline import WhisperTranscriber


def test_transcriber_constructs_callable():
    # Build with the tiny model; should expose a transcribe(path)->str callable.
    t = WhisperTranscriber(model_size="tiny", compute_type="int8")
    assert callable(t.transcribe)


@pytest.mark.slow
def test_transcribe_returns_string_on_real_audio():
    t = WhisperTranscriber(model_size="tiny", compute_type="int8")
    out = t.transcribe("tests/fixtures/audio/a.wav")
    assert isinstance(out, str)  # a pure tone yields "" or noise text; type is the contract
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_asr_baseline.py -v -m "not slow"`
Expected: FAIL with `ModuleNotFoundError: No module named 'cscall.asr_baseline'`

- [ ] **Step 4: Write minimal implementation**

`src/cscall/asr_baseline.py`:
```python
"""Baseline ASR using faster-whisper (the only module touching the model runtime)."""
from faster_whisper import WhisperModel


class WhisperTranscriber:
    """Wraps a faster-whisper model behind a transcribe(path) -> str callable."""

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = None,
    ):
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self._language = language

    def transcribe(self, audio_path: str) -> str:
        segments, _info = self._model.transcribe(audio_path, language=self._language)
        return " ".join(seg.text.strip() for seg in segments).strip()
```

- [ ] **Step 5: Register the `slow` marker**

Append to `pyproject.toml` under `[tool.pytest.ini_options]`:
```toml
markers = ["slow: tests that download/run a real model"]
```

- [ ] **Step 6: Run the fast test to verify it passes**

Run: `pytest tests/test_asr_baseline.py::test_transcriber_constructs_callable -v`
Expected: PASS (downloads the `tiny` model once; subsequent runs cached)

- [ ] **Step 7: Commit**

```bash
git add src/cscall/asr_baseline.py tests/test_asr_baseline.py tests/fixtures/audio pyproject.toml
git commit -m "feat: faster-whisper baseline transcriber + audio fixtures"
```

---

### Task 7: CLI entrypoint

**Files:**
- Create: `src/cscall/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

`tests/test_cli.py`:
```python
from cscall.cli import build_parser


def test_baseline_subcommand_parses():
    parser = build_parser()
    args = parser.parse_args(
        ["baseline", "--manifest", "m.jsonl", "--model", "small", "--group-by", "accent"]
    )
    assert args.command == "baseline"
    assert args.manifest == "m.jsonl"
    assert args.model == "small"
    assert args.group_by == "accent"


def test_model_defaults_to_small():
    parser = build_parser()
    args = parser.parse_args(["baseline", "--manifest", "m.jsonl"])
    assert args.model == "small"
    assert args.group_by is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cscall.cli'`

- [ ] **Step 3: Write minimal implementation**

`src/cscall/cli.py`:
```python
"""CLI: `python -m cscall.cli baseline --manifest <path> [--model small] [--group-by accent]`."""
import argparse
import json

from cscall.asr_baseline import WhisperTranscriber
from cscall.eval_runner import render_markdown, run_eval
from cscall.manifest import load_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cscall")
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("baseline", help="run baseline ASR eval over a manifest")
    b.add_argument("--manifest", required=True)
    b.add_argument("--model", default="small")
    b.add_argument("--group-by", dest="group_by", default=None)
    b.add_argument("--compute-type", dest="compute_type", default="int8")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "baseline":
        utts = load_manifest(args.manifest)
        transcriber = WhisperTranscriber(
            model_size=args.model, compute_type=args.compute_type
        )
        report = run_eval(utts, transcriber.transcribe, group_by=args.group_by)
        print(render_markdown(report))
        print("\nJSON:\n" + json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/cscall/cli.py tests/test_cli.py
git commit -m "feat: baseline CLI entrypoint"
```

---

### Task 8: Data acquisition docs + Svarah download helper

**Files:**
- Create: `data/README.md`
- Create: `data/download_svarah.py`
- Create: `tests/test_download_svarah.py`

- [ ] **Step 1: Write the failing test**

`tests/test_download_svarah.py`:
```python
from data.download_svarah import build_parser


def test_dry_run_flag_parses():
    args = build_parser().parse_args(["--out", "data/raw/svarah", "--dry-run"])
    assert args.out == "data/raw/svarah"
    assert args.dry_run is True


def test_out_is_required():
    import pytest
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_download_svarah.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.download_svarah'`

(If import path errors: add `conftest.py` at repo root containing `import sys, os; sys.path.insert(0, os.path.dirname(__file__))` so `data/` is importable in tests.)

- [ ] **Step 3: Write minimal implementation**

`data/download_svarah.py`:
```python
"""Helper to fetch the Svarah Indian-accented English eval set.

Svarah is published by AI4Bharat. This script only prints/setups the steps and
verifies the target dir; the actual large download is gated behind --execute so
CI/tests never pull gigabytes. Verify the current URL in data/README.md before use.
"""
import argparse
import os

SVARAH_INFO_URL = "https://github.com/AI4Bharat/Svarah"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fetch the Svarah eval set")
    p.add_argument("--out", required=True, help="target directory")
    p.add_argument("--dry-run", dest="dry_run", action="store_true")
    p.add_argument("--execute", action="store_true", help="actually download")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    if args.dry_run or not args.execute:
        print(f"[dry-run] Would download Svarah into {args.out}")
        print(f"See {SVARAH_INFO_URL} for the current dataset link and license.")
        return
    raise SystemExit(
        "Automated download not implemented: follow data/README.md to obtain "
        "Svarah (license acceptance required), then place files in " + args.out
    )


if __name__ == "__main__":
    main()
```

`data/README.md`:
```markdown
# Evaluation Data

Three assets feed the baseline (see the design spec, Section 3).

## A. Public benchmarks

### Svarah (Indian-accented English)
- Source: AI4Bharat — https://github.com/AI4Bharat/Svarah
- Accept the license, download, and place audio + transcripts under `data/raw/svarah/`.
- Then build a manifest at `data/manifests/svarah.jsonl` (one JSON row per the
  `Utterance` schema in `src/cscall/manifest.py`).
- Helper: `python data/download_svarah.py --out data/raw/svarah --dry-run`

### Hindi-English code-switch
- Source options: MUCS 2021 (Hindi-English) or the Microsoft Interspeech 2021
  code-switching challenge set.
- **Fallback** (if access is delayed): any public Hinglish ASR set on Hugging Face
  Hub. Document whichever you use here so results are reproducible.
- Build `data/manifests/codeswitch.jsonl` the same way.

## B. Self-recorded support mini-corpus (~1-3 hrs)
- Record agent<->customer Hinglish role-plays (refunds, late orders), 2 speakers.
- Save 16kHz mono wavs under `data/recorded/`, transcribe each, and label speaker
  turns. Build `data/manifests/recorded.jsonl`.
- This set is REQUIRED for diarization (DER) in later phases and is the live demo.

## Manifest format
Each line of a `*.jsonl` manifest:
```json
{"id": "u1", "audio_path": "data/raw/svarah/x.wav", "text": "reference here", "speaker": "customer", "lang": "hi-en", "accent": "north", "cs_density": 0.4}
```
`id`, `audio_path`, `text` are required; the rest are optional grouping keys.

## Run the baseline
```bash
python -m cscall.cli baseline --manifest data/manifests/svarah.jsonl --model small --group-by accent
```
Record the printed WER/CER table in the project README — this is the "before" number.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_download_svarah.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the full suite (excluding slow)**

Run: `pytest -m "not slow" -v`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add data/README.md data/download_svarah.py tests/test_download_svarah.py conftest.py
git commit -m "docs: data acquisition guide + Svarah download helper"
```

---

## Manual Steps (human, not subagent)

These are explicitly out of scope for subagents and tracked for the human:
1. Obtain Svarah + a Hindi-English code-switch set (license acceptance), build their manifests.
2. Record + transcribe + speaker-label the ~1-3 hr self-recorded corpus, build its manifest.
3. Run `python -m cscall.cli baseline ...` on each real manifest and paste the WER/CER tables into `README.md` as the locked-in baseline.

---

## Self-Review

- **Spec coverage:** Normalization scheme (§3) → Task 2. WER/CER + DER metrics (§3) → Task 4 covers WER/CER; DER is a later phase (diarization), correctly deferred. Manifest/data assets (§3) → Tasks 3 & 8. Baseline experiment (§3, Phase 0) → Tasks 5–7 + Manual Steps. Repo structure (§8) → Task 1 + per-task files. Grouped breakdowns (§3) → Task 5 `group_by`.
- **Placeholder scan:** No TBD/TODO; all steps contain runnable code/commands.
- **Type consistency:** `Utterance` fields used identically in Tasks 3/5/8; `score()` return keys (`wer`/`cer`/`n`) consistent across Tasks 4/5; `transcribe(path)->str` contract consistent across Tasks 5/6/7.
- **Deferred intentionally:** DER, fine-tuning, streaming, UI, Docker — separate phase plans.
```
