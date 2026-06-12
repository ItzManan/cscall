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
