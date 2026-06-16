# Phase 1 LoRA Fine-Tune — Findings & Handoff (2026-06-15)

> **Purpose:** Full context for an agent picking up this project fresh. Covers what was
> built, the Phase 1 fine-tune experiment, its **negative result**, the diagnosis, and the
> open decision on how to proceed. Read alongside:
> - `docs/superpowers/specs/2026-06-12-streaming-codeswitch-asr-design.md` (overall design)
> - `docs/superpowers/plans/2026-06-12-phase0-scaffold-data-baseline.md`
> - `docs/superpowers/plans/2026-06-12-phase1-lora-finetune.md`

## Project in one line

Real-time, speaker-attributed transcription for **Indian-accented, Hindi-English
code-switched** support calls. Three capabilities being fused: (1) accent/code-switch-robust
ASR, (2) low-latency streaming, (3) speaker diarization. This is a resume/portfolio project
for MLE/DS roles — the bar is **production depth + real modeling rigor**, not flashy demos.
Owner curates and understands the code deeply; implementation is subagent-driven (Sonnet
writes, Opus/owner reviews).

## What exists and works (Phase 0 + Phase 1 code)

All TDD'd, reviewed, committed. Package `cscall` (src-layout, `pyproject.toml`, Python 3.11,
**uv** — the `.venv` has no `pip`; run `VIRTUAL_ENV="$PWD/.venv" .venv/bin/python ...`).

- `src/cscall/normalize.py` — `normalize_text()`, `romanize_devanagari()` (ITRANS) for fair WER.
- `src/cscall/manifest.py` — `Utterance` dataclass + `load_manifest()`.
- `src/cscall/metrics.py` — `score(refs, hyps) -> {wer, cer, n}` (jiwer; empty-corpus guard).
- `src/cscall/eval_runner.py` — `run_eval(utts, transcribe, group_by=None)`, `render_markdown()`.
- `src/cscall/asr_baseline.py` — `WhisperTranscriber(model_size, device, compute_type, language)`
  wrapping faster-whisper. **NOTE:** `transcribe()` passes `language=self._language`
  (default `None` = auto-detect). The CLI does **not** expose `--language` yet.
- `src/cscall/compare.py` — `compare_models()`, `render_comparison_markdown()` (before/after table).
- `src/cscall/cli.py` — `baseline` and `compare` subcommands. Has `--device` (cpu/cuda),
  `--compute-type` (default int8), `--group-by`. **Does NOT have `--language`.**
- `src/cscall/finetune/` — `cs_density.py` (`code_switch_density`, `cs_bucket`: none/low/mid/high),
  `dataset.py` (`to_training_records`, `split_manifest`, `write_manifest`), `config.py`
  (`FineTuneConfig`), `export_ct2.py` (`build_convert_command`).
- `notebooks/phase1_finetune_colab.ipynb` — runnable Colab LoRA pipeline + a GPU-eval section.

### Data assets (gitignored; live under `data/raw/`, `data/manifests/`)

- **HiACC** (Hindi-English code-switch, Zenodo DOI 10.5281/zenodo.15551669, CC BY-NC 4.0,
  16kHz mono). Official splits → `codeswitch_train.jsonl` (2322), `codeswitch_val.jsonl`,
  `codeswitch_test.jsonl` (664). Built by `data/build_hiacc_manifests.py`. References are
  **mixed-script**: Hindi in Devanagari, English loanwords kept in **Latin** ("better की hometown").
- **Svarah** (AI4Bharat Indian-accented English, 6656 clips) → `data/manifests/svarah.jsonl`.
  Built by `data/build_svarah_manifest.py`. Baseline NOT yet run (slow on CPU).

### Colab bundles (built on the Mac, in `~/`)

- `~/cscall_colab_bundle.zip` (~214 MB) — train audio + manifests + code, for fine-tuning.
- `~/cscall_eval_bundle.zip` (~263 MB) — **test** audio + manifests + code + the fine-tuned
  `output_model/ct2`, for GPU eval.
- `output_model/` (downloaded from Colab) — `ct2/` (fine-tuned CTranslate2 model, ~248 MB
  `model.bin`) and `merged/`. This is the trained artifact under analysis.

## The Phase 1 experiment

Fine-tuned `openai/whisper-small` with LoRA on HiACC train (2322 clips), exported merged
model to CTranslate2 int8, evaluated on the 664-clip held-out test split.

**Training recipe (`FineTuneConfig` defaults):** `lora_r=16`, `lora_alpha=32`,
`target_modules=["q_proj","v_proj"]`, `lr=1e-3`, `num_epochs=3`, `batch_size=8`,
`grad_accum=2`, `language="hi"`, `task="transcribe"`, `save_strategy="no"` (no checkpoint
selection — final model taken regardless of val). Trained on Colab T4.

### Result: the fine-tune did NOT improve code-switch WER (negative/null result)

**Full GPU compare (664 test clips, faster-whisper float16, `language=None`):**

| Group | Baseline WER | Fine-tuned WER | Δ WER |
|---|---|---|---|
| **overall** | 0.485 | 0.502 | **−0.017** |
| high | 0.593 | 0.591 | +0.002 |
| low  | 0.634 | 0.746 | −0.113 |
| mid  | 0.814 | 0.818 | −0.004 |
| none | 0.155 | 0.186 | −0.030 |

(Negative Δ = fine-tuned worse. Note: this base column, GPU/float16, differs from the
README baseline table which was CPU/int8: overall 0.515 there vs 0.485 here — compute-backend
variation, see below.)

## Diagnosis (this is the important part)

Ran two local CPU diagnostics (`scratch_diag.py`, `scratch_regime.py` in repo root —
throwaway, safe to delete) dumping base-vs-ft predictions and per-regime WER.

**60-clip subset WER (CPU int8; composition none=34, high=19, mid=4, low=3):**

```
                overall   high(19)   mid(4)    none(34)
base(auto)      0.407     0.486      0.863     0.238
base(hi)        0.843     0.482      0.863     1.206
ft(auto)        0.380     0.498      0.863     0.146
ft(hi)          0.695     0.490      0.863     0.872
```

### Findings

1. **The fine-tune's effect is within noise.** On the full GPU set ft was *worse* on `none`
   (0.155→0.186); on the CPU subset ft was *better* on `none` (0.238→0.146). Same bucket,
   opposite sign — driven by float16-GPU-vs-int8-CPU differences + Whisper's temperature
   fallback (default `temperature=[0.0,0.2,...]` retries on low logprob/high compression
   ratio → non-reproducible garbled outputs). **No robust improvement or regression.**

2. **No movement on the buckets that matter.** `high` ≈ 0.49 and `mid` = 0.863 in *every*
   condition. The whole point of the project — better code-switch transcription — shows
   nothing. (`mid`=0.863 identical across all conditions is just 4 hard clips, small-n.)

3. **The adapter DID learn something, but only under forced `language="hi"`.** With
   `language=None` (how both the compare and default CLI run), ft output ≈ base output (often
   byte-identical). Forced to `hi`, base Whisper collapses on English ("the picture is
   presenting..." → "में ब़।म्ले आखसा...") while ft recovers the correct English. So the LoRA
   merged in and is real, but it's dormant in the eval regime.

4. **Train/eval regime mismatch.** Trained 100% under `language="hi"` prompt; evaluated under
   `language=None`. BUT — forcing `hi` at eval is a **net loss** on this test set (lots of
   pure-English clips; forcing Hindi decoding wrecks them). So "just eval with hi" is NOT the
   fix. (This hypothesis was tested and refuted.)

5. **Root cause of high code-switch WER = script-convention mismatch, not acoustics.** Base
   Whisper-small is already decent on the Hindi audio. The WER is dominated by: (a) English
   loanwords — HiACC keeps them Latin ("better"), Whisper Devanagari-izes them ("बेटर" →
   romanized "betara") → counted as errors; (b) diacritics (पहुचे vs पहुँचे). The fine-tune
   did **not** learn HiACC's Latin-for-English convention (high bucket unchanged), which is
   the single biggest learnable lever.

### Why the recipe was too weak

Only `q_proj`/`v_proj` adapted (low capacity), `lr=1e-3` (aggressive, some English
forgetting), 3 epochs, and `save_strategy="no"` (no best-checkpoint selection). Thin LoRA
against a hard stylistic target. The model *can* learn this (it protected English under
hi-forcing) — the recipe just didn't have the capacity/schedule to instill the script convention.

## Bugs fixed while getting the notebook to run (Colab gotchas — keep these)

- `datasets>=4` forces `torchcodec` for audio decode → pin `datasets<4` (+ explicit `soundfile`).
- Colab preinstalls `torchao 0.10.0` which newer transformers rejects at import → `pip uninstall -y torchao` (we don't use it).
- `Seq2SeqTrainer(tokenizer=...)` removed in newer transformers → use `processing_class=`.
- **`LoraConfig(task_type=SEQ_2_SEQ_LM)` breaks Whisper** ("got multiple values for keyword
  argument 'input_ids'") because PEFT injects `input_ids` and Whisper's encoder takes
  `input_features`. **Omit `task_type`** — canonical HF Whisper-LoRA recipe.
- faster-whisper on CUDA needs cuDNN on `LD_LIBRARY_PATH`; if "Unable to load libcudnn_ops",
  `pip install nvidia-cudnn-cu12==9.*`. The GPU-eval cell exports the cudnn lib path.

## Open decision (owner is switching agents here — pick up from this fork)

The owner has NOT chosen a direction. Options presented:

1. **Retrain, stronger recipe** (recommended if goal is a real win): adapt all attention+MLP
   modules (add `k_proj`, `out_proj`, `fc1`, `fc2`), `lr≈2e-4` cosine, 6–10 epochs, add
   eval-on-val + best-checkpoint selection. Explicitly target the Latin-script convention for
   English words. ~30–45 min Colab run. Real shot at moving `high`/`mid`. Then re-run GPU compare.
2. **Diagnose data/metric first**: decompose WER into script-convention vs diacritics vs real
   acoustic error to quantify the achievable ceiling and whether the eval is fair. ~20 min, no GPU.
3. **Reframe as honest finding**: accept the null result; narrative becomes "Whisper-small is
   already strong on Indian Hinglish; naive LoRA doesn't help — rigorous eval harness + error
   analysis showing why." Defensible, senior-signaling, no more training.
4. **Move to Phase 2 streaming** (VAD + LocalAgreement): park the fine-tune, build breadth,
   revisit later.

### Recommended next step regardless of path

Add a `--language` flag to the CLI/`WhisperTranscriber` (currently missing) so eval regime is
explicit and reproducible — the `--device` flag was just added the same way (see commit
`5728f9e`, TDD in `tests/test_cli.py`).

## Reproduce the eval

```bash
# CPU (slow, ~1 hr for 664×2):
VIRTUAL_ENV="$PWD/.venv" .venv/bin/python -m cscall.cli compare \
  --manifest data/manifests/codeswitch_test.jsonl \
  --finetuned-ct2 output_model/ct2 --group-by cs_bucket

# GPU (Colab T4, ~few min): upload ~/cscall_eval_bundle.zip to Drive, run the
# "GPU eval" cell in notebooks/phase1_finetune_colab.ipynb (adds --device cuda --compute-type float16).
```

## Conventions / preferences (from owner)

- Git: commit identity `Manan <manishjainjain197220@gmail.com>`, **no Co-Authored-By trailer**.
- Subagent-driven dev: Sonnet implementers by default; Haiku only for trivially mechanical tasks.
- No LLM-centric framing for the project (voice/ASR niche chosen deliberately).
- Never commit HF tokens or any secret to the repo.
