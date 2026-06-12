# Phase 1: LoRA Fine-Tune + Comparison — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the (locally testable) code scaffolding to LoRA-fine-tune Whisper for Hindi-English code-switch, export it to CTranslate2, and produce a baseline-vs-fine-tuned WER/CER comparison — reusing the Phase 0 eval harness — so the actual Colab/T4 training run and the headline "before→after" table can be produced by the human.

**Architecture:** New modules under `src/cscall/finetune/` plus a thin extension to the eval layer. Pure/logic-heavy pieces (dataset manifest → HF-style records, code-switch-density computation, training-arg config, CT2 export wiring, baseline-vs-fine-tuned comparison + delta table) are built TDD against tiny fixtures and run on CPU. The GPU-bound training itself lives in a Colab notebook that imports these modules; subagents build and test everything except the actual GPU run. The fine-tuned model is consumed in production via the **existing** `WhisperTranscriber` (CTranslate2), so no inference code changes — the spec's "live service runs the model the offline harness scored" property holds.

**Tech Stack:** Python 3.11, Hugging Face `transformers` + `peft` (LoRA) + `datasets`, `ctranslate2` converter (`ct2-transformers-converter`), reuse of Phase 0 `faster-whisper`/`jiwer`/`indic-transliteration`. Training on Colab/Kaggle free T4; all unit tests CPU-only with no model download.

**Depends on Phase 0:** `cscall.manifest`, `cscall.metrics`, `cscall.eval_runner`, `cscall.normalize`, `cscall.asr_baseline` (all complete, 25 tests passing).

---

## Human-Only Prerequisites (NOT subagent tasks)

These gate the *real* numbers but not the code build. Tracked explicitly:
1. Obtain datasets per `data/README.md` (Svarah + a Hindi-English code-switch set) and build their manifests. **Define a strict train/test split** — never fine-tune on the official test split you will report WER on.
2. Run the Colab notebook (Task 7) on a T4 to produce LoRA weights, then the CT2 export.
3. Run the comparison CLI (Task 6) on the held-out test manifest and paste the before→after table into `README.md`.

---

## File Structure

```
src/cscall/finetune/__init__.py
src/cscall/finetune/cs_density.py        # code-switch density metric (pure, TDD)
src/cscall/finetune/dataset.py           # manifest -> training records + split helpers (TDD)
src/cscall/finetune/config.py            # LoRA + training hyperparams dataclass (TDD)
src/cscall/finetune/export_ct2.py        # wrap ct2-transformers-converter (arg/build, TDD on cmd)
src/cscall/compare.py                    # baseline vs fine-tuned: run both, delta table (TDD via fakes)
src/cscall/cli.py                        # MODIFY: add `compare` subcommand
notebooks/phase1_finetune_colab.ipynb    # Colab orchestration (human-run; not unit-tested)
tests/test_cs_density.py
tests/test_finetune_dataset.py
tests/test_finetune_config.py
tests/test_export_ct2.py
tests/test_compare.py
```

**Responsibilities:** `cs_density` = one metric. `dataset` = manifest↔training-record conversion + split. `config` = hyperparameter container only. `export_ct2` = build the converter command only (no training). `compare` = orchestrate two transcribers through `eval_runner` and diff. The notebook wires `dataset`+`config` into an actual `peft` training loop (the only GPU code).

---

### Task 1: Code-switch density metric

**Files:**
- Create: `src/cscall/finetune/__init__.py` (empty)
- Create: `src/cscall/finetune/cs_density.py`
- Create: `tests/test_cs_density.py`

- [ ] **Step 1: Write the failing test**

`tests/test_cs_density.py`:
```python
from cscall.finetune.cs_density import code_switch_density


def test_all_english_is_zero():
    assert code_switch_density("please wait a moment") == 0.0


def test_all_devanagari_is_one():
    # every token is Devanagari script
    assert code_switch_density("नमस्ते खाना है") == 1.0


def test_half_and_half():
    # 2 of 4 tokens Devanagari (खाना, है)
    assert code_switch_density("order खाना है ready") == 0.5


def test_empty_is_zero():
    assert code_switch_density("") == 0.0


def test_romanized_hindi_counts_as_english_script():
    # script-based metric: romanized Hindi is Latin script, so density 0.0
    # (documents the known limitation — this measures SCRIPT mixing, not language)
    assert code_switch_density("thoda wait karo") == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cs_density.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cscall.finetune'`

- [ ] **Step 3: Write minimal implementation**

`src/cscall/finetune/__init__.py`: (empty file)

`src/cscall/finetune/cs_density.py`:
```python
"""Code-switch density: fraction of whitespace tokens written in Devanagari script.

This is a SCRIPT-based proxy (not language ID): romanized Hindi counts as 0.
Used to bucket eval utterances by how heavily code-switched they are.
"""
import re

_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")


def code_switch_density(text: str) -> float:
    """Return the fraction of tokens that contain at least one Devanagari char."""
    tokens = text.split()
    if not tokens:
        return 0.0
    deva = sum(1 for t in tokens if _DEVANAGARI_RE.search(t))
    return deva / len(tokens)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_cs_density.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/cscall/finetune/__init__.py src/cscall/finetune/cs_density.py tests/test_cs_density.py
git commit -m "feat: code-switch density metric for eval bucketing"
```

---

### Task 2: Training dataset records + split

**Files:**
- Create: `src/cscall/finetune/dataset.py`
- Create: `tests/test_finetune_dataset.py`

- [ ] **Step 1: Write the failing test**

`tests/test_finetune_dataset.py`:
```python
from cscall.finetune.dataset import to_training_records, split_manifest
from cscall.manifest import load_manifest


def test_to_training_records_shape():
    utts = load_manifest("tests/fixtures/mini_manifest.jsonl")
    records = to_training_records(utts)
    assert len(records) == 3
    assert records[0] == {"audio_path": "tests/fixtures/audio/a.wav", "text": "order kahan hai"}


def test_split_is_deterministic_and_disjoint():
    utts = load_manifest("tests/fixtures/mini_manifest.jsonl")
    train1, test1 = split_manifest(utts, test_frac=0.34, seed=0)
    train2, test2 = split_manifest(utts, test_frac=0.34, seed=0)
    # deterministic
    assert [u.id for u in train1] == [u.id for u in train2]
    assert [u.id for u in test1] == [u.id for u in test2]
    # disjoint and complete
    ids_train = {u.id for u in train1}
    ids_test = {u.id for u in test1}
    assert ids_train.isdisjoint(ids_test)
    assert ids_train | ids_test == {"u1", "u2", "u3"}


def test_split_test_frac_size():
    utts = load_manifest("tests/fixtures/mini_manifest.jsonl")
    _, test = split_manifest(utts, test_frac=0.34, seed=0)
    assert len(test) == 1  # round(3 * 0.34) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_finetune_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cscall.finetune.dataset'`

- [ ] **Step 3: Write minimal implementation**

`src/cscall/finetune/dataset.py`:
```python
"""Convert eval manifests into training records and deterministic train/test splits.

A training record is the minimal {audio_path, text} the Colab training loop needs;
keeping it tiny avoids coupling the GPU loop to the full Utterance schema.
"""
import random

from cscall.manifest import Utterance


def to_training_records(utterances: list[Utterance]) -> list[dict]:
    """Project Utterances to minimal training records."""
    return [{"audio_path": u.audio_path, "text": u.text} for u in utterances]


def split_manifest(
    utterances: list[Utterance],
    test_frac: float = 0.2,
    seed: int = 0,
) -> tuple[list[Utterance], list[Utterance]]:
    """Deterministically split into (train, test) by a shuffled copy.

    Seeded so the split is reproducible; test set is round(n * test_frac) items.
    """
    if not 0.0 <= test_frac <= 1.0:
        raise ValueError(f"test_frac must be in [0,1], got {test_frac}")
    items = list(utterances)
    random.Random(seed).shuffle(items)
    n_test = round(len(items) * test_frac)
    test = items[:n_test]
    train = items[n_test:]
    return train, test
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_finetune_dataset.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/cscall/finetune/dataset.py tests/test_finetune_dataset.py
git commit -m "feat: training records + deterministic train/test split"
```

---

### Task 3: LoRA + training config

**Files:**
- Create: `src/cscall/finetune/config.py`
- Create: `tests/test_finetune_config.py`

- [ ] **Step 1: Write the failing test**

`tests/test_finetune_config.py`:
```python
from cscall.finetune.config import FineTuneConfig, default_lora_target_modules


def test_defaults_are_t4_friendly():
    cfg = FineTuneConfig()
    assert cfg.base_model == "openai/whisper-small"
    assert cfg.lora_r == 16
    assert cfg.lora_alpha == 32
    assert cfg.batch_size <= 16  # fits a T4
    assert cfg.language == "hi"


def test_target_modules_are_attention_projections():
    mods = default_lora_target_modules()
    assert "q_proj" in mods
    assert "v_proj" in mods


def test_to_dict_roundtrip():
    cfg = FineTuneConfig(lora_r=8)
    d = cfg.to_dict()
    assert d["lora_r"] == 8
    assert "base_model" in d
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_finetune_config.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

`src/cscall/finetune/config.py`:
```python
"""Hyperparameters for LoRA fine-tuning Whisper. Pure data — no training here."""
from dataclasses import asdict, dataclass, field


def default_lora_target_modules() -> list[str]:
    """Attention projection layers LoRA adapts in Whisper."""
    return ["q_proj", "v_proj"]


@dataclass
class FineTuneConfig:
    base_model: str = "openai/whisper-small"
    language: str = "hi"          # Whisper task language tag for decoding
    task: str = "transcribe"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=default_lora_target_modules)
    batch_size: int = 8           # per-device; T4-friendly with fp16 + grad checkpointing
    grad_accum: int = 2
    learning_rate: float = 1e-3   # LoRA tolerates higher LR than full fine-tune
    num_epochs: int = 3
    warmup_steps: int = 50

    def to_dict(self) -> dict:
        return asdict(self)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_finetune_config.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/cscall/finetune/config.py tests/test_finetune_config.py
git commit -m "feat: LoRA/training config dataclass (T4 defaults)"
```

---

### Task 4: CTranslate2 export command builder

**Files:**
- Create: `src/cscall/finetune/export_ct2.py`
- Create: `tests/test_export_ct2.py`

- [ ] **Step 1: Write the failing test**

`tests/test_export_ct2.py`:
```python
from cscall.finetune.export_ct2 import build_convert_command


def test_command_targets_merged_model_dir():
    cmd = build_convert_command(
        merged_model_dir="out/merged", output_dir="out/ct2", quantization="int8"
    )
    assert cmd[0] == "ct2-transformers-converter"
    assert "--model" in cmd
    assert "out/merged" in cmd
    assert "--output_dir" in cmd
    assert "out/ct2" in cmd
    assert "--quantization" in cmd
    assert "int8" in cmd


def test_force_flag_included():
    cmd = build_convert_command("m", "o", "int8", force=True)
    assert "--force" in cmd


def test_invalid_quantization_raises():
    import pytest
    with pytest.raises(ValueError, match="quantization"):
        build_convert_command("m", "o", "float64")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_export_ct2.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

`src/cscall/finetune/export_ct2.py`:
```python
"""Build the ct2-transformers-converter command to export a merged Whisper to CT2.

Fine-tuning produces LoRA adapters; before export they must be MERGED into the base
model (peft `merge_and_unload`) in the Colab notebook. This module only constructs
the converter command so it is unit-testable without running the heavy conversion.
"""
_VALID_QUANT = {"int8", "int8_float16", "float16", "float32"}


def build_convert_command(
    merged_model_dir: str,
    output_dir: str,
    quantization: str = "int8",
    force: bool = False,
) -> list[str]:
    """Return argv for ct2-transformers-converter. Faster-whisper loads the result."""
    if quantization not in _VALID_QUANT:
        raise ValueError(
            f"quantization must be one of {sorted(_VALID_QUANT)}, got {quantization!r}"
        )
    cmd = [
        "ct2-transformers-converter",
        "--model", merged_model_dir,
        "--output_dir", output_dir,
        "--quantization", quantization,
    ]
    if force:
        cmd.append("--force")
    return cmd
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_export_ct2.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/cscall/finetune/export_ct2.py tests/test_export_ct2.py
git commit -m "feat: CTranslate2 export command builder"
```

---

### Task 5: Baseline-vs-fine-tuned comparison

**Files:**
- Create: `src/cscall/compare.py`
- Create: `tests/test_compare.py`

- [ ] **Step 1: Write the failing test**

`tests/test_compare.py`:
```python
from cscall.compare import compare_models, render_comparison_markdown
from cscall.manifest import load_manifest


def baseline_fake(audio_path: str) -> str:
    return {
        "tests/fixtures/audio/a.wav": "order kahan",          # drops "hai"
        "tests/fixtures/audio/b.wav": "your refund is processed",
        "tests/fixtures/audio/c.wav": "thoda wait karo",       # drops "please"
    }[audio_path]


def finetuned_fake(audio_path: str) -> str:
    # perfect — simulates the fine-tuned model fixing the errors
    return {
        "tests/fixtures/audio/a.wav": "order kahan hai",
        "tests/fixtures/audio/b.wav": "your refund is processed",
        "tests/fixtures/audio/c.wav": "thoda wait karo please",
    }[audio_path]


def test_compare_reports_both_and_delta():
    utts = load_manifest("tests/fixtures/mini_manifest.jsonl")
    result = compare_models(utts, baseline_fake, finetuned_fake)
    assert result["baseline"]["overall"]["wer"] > 0.0
    assert result["finetuned"]["overall"]["wer"] == 0.0
    # delta = baseline - finetuned (positive == improvement)
    assert result["delta_wer"] == result["baseline"]["overall"]["wer"]


def test_render_comparison_has_both_columns():
    utts = load_manifest("tests/fixtures/mini_manifest.jsonl")
    result = compare_models(utts, baseline_fake, finetuned_fake, group_by="accent")
    md = render_comparison_markdown(result)
    assert "Baseline WER" in md
    assert "Fine-tuned WER" in md
    assert "north" in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_compare.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cscall.compare'`

- [ ] **Step 3: Write minimal implementation**

`src/cscall/compare.py`:
```python
"""Run two transcribers over the same manifest and report the WER/CER delta.

Reuses the Phase 0 eval_runner so the comparison is apples-to-apples with the
baseline numbers. Positive delta == fine-tuned is better.
"""
from typing import Optional

from cscall.eval_runner import Transcriber, run_eval
from cscall.manifest import Utterance


def compare_models(
    utterances: list[Utterance],
    baseline: Transcriber,
    finetuned: Transcriber,
    group_by: Optional[str] = None,
) -> dict:
    """Return {baseline, finetuned, delta_wer, delta_cer} reports."""
    base = run_eval(utterances, baseline, group_by=group_by)
    fine = run_eval(utterances, finetuned, group_by=group_by)
    return {
        "baseline": base,
        "finetuned": fine,
        "delta_wer": base["overall"]["wer"] - fine["overall"]["wer"],
        "delta_cer": base["overall"]["cer"] - fine["overall"]["cer"],
    }


def render_comparison_markdown(result: dict) -> str:
    """Render a before/after table with per-group rows."""
    lines = [
        "| Group | Baseline WER | Fine-tuned WER | Δ WER |",
        "|---|---|---|---|",
    ]
    bo = result["baseline"]["overall"]
    fo = result["finetuned"]["overall"]
    lines.append(
        f"| **overall** | {bo['wer']:.3f} | {fo['wer']:.3f} | {bo['wer'] - fo['wer']:+.3f} |"
    )
    bgroups = result["baseline"].get("groups", {})
    fgroups = result["finetuned"].get("groups", {})
    for key in sorted(bgroups):
        bw = bgroups[key]["wer"]
        fw = fgroups.get(key, {}).get("wer", float("nan"))
        lines.append(f"| {key} | {bw:.3f} | {fw:.3f} | {bw - fw:+.3f} |")
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_compare.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/cscall/compare.py tests/test_compare.py
git commit -m "feat: baseline-vs-fine-tuned comparison with delta table"
```

---

### Task 6: `compare` CLI subcommand

**Files:**
- Modify: `src/cscall/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test (append to tests/test_cli.py)**

```python
def test_compare_subcommand_parses():
    parser = build_parser()
    args = parser.parse_args(
        [
            "compare",
            "--manifest", "m.jsonl",
            "--baseline-model", "small",
            "--finetuned-ct2", "out/ct2",
            "--group-by", "accent",
        ]
    )
    assert args.command == "compare"
    assert args.manifest == "m.jsonl"
    assert args.baseline_model == "small"
    assert args.finetuned_ct2 == "out/ct2"
    assert args.group_by == "accent"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cli.py::test_compare_subcommand_parses -v`
Expected: FAIL with `AttributeError`/`SystemExit` (no `compare` subcommand yet)

- [ ] **Step 3: Add the subcommand**

In `src/cscall/cli.py`, add these imports near the top (with the existing imports):
```python
from cscall.compare import compare_models, render_comparison_markdown
```

In `build_parser()`, after the existing `baseline` subparser block and before `return parser`, add:
```python
    c = sub.add_parser("compare", help="baseline vs fine-tuned WER on a manifest")
    c.add_argument("--manifest", required=True)
    c.add_argument("--baseline-model", dest="baseline_model", default="small")
    c.add_argument("--finetuned-ct2", dest="finetuned_ct2", required=True)
    c.add_argument("--group-by", dest="group_by", default=None)
    c.add_argument("--compute-type", dest="compute_type", default="int8")
```

In `main()`, after the existing `if args.command == "baseline":` block, add:
```python
    elif args.command == "compare":
        utts = load_manifest(args.manifest)
        baseline = WhisperTranscriber(
            model_size=args.baseline_model, compute_type=args.compute_type
        )
        finetuned = WhisperTranscriber(
            model_size=args.finetuned_ct2, compute_type=args.compute_type
        )
        result = compare_models(
            utts, baseline.transcribe, finetuned.transcribe, group_by=args.group_by
        )
        print(render_comparison_markdown(result))
```
(Note: `WhisperTranscriber` accepts either a model size name OR a local CT2 directory path as `model_size` — faster-whisper resolves both. The fine-tuned model is a CT2 directory from Task 4's export.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_cli.py -v`
Expected: PASS (all cli tests, including the new one)

- [ ] **Step 5: Commit**

```bash
git add src/cscall/cli.py tests/test_cli.py
git commit -m "feat: compare CLI subcommand (baseline vs fine-tuned)"
```

---

### Task 7: Colab training notebook (human-run; structural test only)

**Files:**
- Create: `notebooks/phase1_finetune_colab.ipynb`
- Create: `tests/test_notebook_structure.py`

This task creates the GPU-run notebook and a light test that it is valid JSON with the expected sections. The notebook is NOT executed in CI (no GPU); the human runs it on Colab.

- [ ] **Step 1: Write the failing test**

`tests/test_notebook_structure.py`:
```python
import json


def test_notebook_is_valid_and_has_sections():
    with open("notebooks/phase1_finetune_colab.ipynb", encoding="utf-8") as f:
        nb = json.load(f)
    assert nb["nbformat"] >= 4
    sources = "\n".join(
        "".join(cell.get("source", [])) for cell in nb["cells"]
    )
    # references the project modules it must wire together
    assert "cscall.finetune.config" in sources
    assert "cscall.finetune.dataset" in sources
    assert "peft" in sources
    assert "merge_and_unload" in sources
    assert "ct2-transformers-converter" in sources or "build_convert_command" in sources
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_notebook_structure.py -v`
Expected: FAIL with `FileNotFoundError`

- [ ] **Step 3: Create the notebook**

Create `notebooks/phase1_finetune_colab.ipynb` as a valid Jupyter notebook (nbformat 4) with markdown + code cells implementing, in order:
1. **Markdown intro** — purpose, expects a T4 runtime, links to this plan.
2. **Install** cell: `!pip install -q transformers peft datasets accelerate jiwer indic-transliteration ctranslate2 faster-whisper` and `!git clone` of this repo (or `%pip install -e .`).
3. **Imports** cell: `from cscall.finetune.config import FineTuneConfig` and `from cscall.finetune.dataset import to_training_records, split_manifest` and `from cscall.manifest import load_manifest`.
4. **Load data** cell: `utts = load_manifest("data/manifests/codeswitch.jsonl")`, then `train, test = split_manifest(utts, test_frac=0.2, seed=0)` and `records = to_training_records(train)`. Markdown note: never include the test split in training.
5. **Build HF dataset** cell: load audio with `datasets.Audio(sampling_rate=16000)`, map to log-mel features via `WhisperProcessor`, labels via the tokenizer.
6. **Model + LoRA** cell: `cfg = FineTuneConfig()`; load `WhisperForConditionalGeneration.from_pretrained(cfg.base_model)`; wrap with `peft.LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout, target_modules=cfg.target_modules)` → `get_peft_model`.
7. **Train** cell: `Seq2SeqTrainer` with `Seq2SeqTrainingArguments(per_device_train_batch_size=cfg.batch_size, gradient_accumulation_steps=cfg.grad_accum, learning_rate=cfg.learning_rate, num_train_epochs=cfg.num_epochs, warmup_steps=cfg.warmup_steps, fp16=True, gradient_checkpointing=True)`.
8. **Merge + export** cell: `merged = model.merge_and_unload(); merged.save_pretrained("out/merged")`; then build and run the CT2 command:
   ```python
   from cscall.finetune.export_ct2 import build_convert_command
   import subprocess
   subprocess.run(build_convert_command("out/merged", "out/ct2", "int8", force=True), check=True)
   ```
9. **Quick eval** cell (markdown + code): note that the human downloads `out/ct2`, then locally runs `python -m cscall.cli compare --manifest data/manifests/codeswitch_test.jsonl --finetuned-ct2 out/ct2 --group-by cs_density` and pastes the table into `README.md`.

Ensure the file is valid JSON conforming to nbformat 4 (cells list, each with `cell_type`, `source`, and for code cells `outputs: []`, `execution_count: null`, plus top-level `metadata: {}`, `nbformat: 4`, `nbformat_minor: 5`).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_notebook_structure.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Run the full fast suite**

Run: `.venv/bin/python -m pytest -m "not slow" -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add notebooks/phase1_finetune_colab.ipynb tests/test_notebook_structure.py
git commit -m "feat: Colab LoRA fine-tune notebook + structure test"
```

---

## Manual Steps (human, not subagent)

1. Build `data/manifests/codeswitch.jsonl` (+ a held-out `codeswitch_test.jsonl`) from a real Hindi-English code-switch dataset; ensure train/test are disjoint.
2. Open `notebooks/phase1_finetune_colab.ipynb` in Colab on a T4, run all cells, download `out/ct2`.
3. Locally: `python -m cscall.cli baseline --manifest data/manifests/codeswitch_test.jsonl --model small --group-by cs_density` (the "before"), then `python -m cscall.cli compare --manifest data/manifests/codeswitch_test.jsonl --finetuned-ct2 out/ct2 --group-by cs_density` (the before→after delta). Paste both into `README.md`.

---

## Self-Review

- **Spec coverage (§3 + Phase 1 milestone):** LoRA fine-tune → Tasks 3 + 7. Code-switch normalization already in Phase 0 (reused by metrics). Fine-tuned WER / breakdown tables → Tasks 5 + 6 (compare) + cs_density bucketing (Task 1). CT2 export → Task 4 + notebook cell 8. Train/test discipline → Task 2 + manual steps. "Live service runs the scored model" → fine-tuned model consumed via existing `WhisperTranscriber` (no inference fork), satisfied by Task 6 using a CT2 dir.
- **Placeholder scan:** No TBD/TODO; every code step has runnable code. The notebook task gives concrete cell-by-cell content (the one task that is human-run by nature; its test asserts structure, not GPU execution).
- **Type consistency:** `Transcriber = Callable[[str], str]` reused from eval_runner in compare.py; `WhisperTranscriber(model_size=...)` signature matches Phase 0; `run_eval(..., group_by=...)` matches Phase 0 (and `cs_density` is added to GROUPABLE_FIELDS — NOTE: Task 1 adds the metric, but grouping by it requires each Utterance to carry a `cs_density` value; manifests already include `cs_density`, and Phase 0's GROUPABLE_FIELDS already lists `"cs_density"`). `compare_models`/`render_comparison_markdown` names consistent across Tasks 5 and 6.
- **Deferred to later phases:** streaming, VAD, diarization, fusion, UI, Docker.
```
