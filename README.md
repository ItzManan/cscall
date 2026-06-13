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

### Baseline results — HiACC Hindi-English code-switch (whisper-small, 664-clip test set)

Word/Character Error Rate by code-switch density bucket (`--group-by cs_bucket`):

| code-switch bucket | WER | CER | N |
|---|---|---|---|
| **overall** | **0.515** | 0.295 | 664 |
| none (pure English) | 0.174 | 0.107 | 274 |
| low | 0.673 | 0.438 | 40 |
| mid | 0.839 | 0.518 | 87 |
| high | 0.633 | 0.337 | 263 |

The off-the-shelf model handles pure-English utterances well (17% WER) but degrades
sharply on code-switched speech (63–84% WER) — quantifying the gap the Phase 1 LoRA
fine-tune targets. Reproduce with:

```bash
python -m cscall.cli baseline --manifest data/manifests/codeswitch_test.jsonl \
    --model small --group-by cs_bucket
```
