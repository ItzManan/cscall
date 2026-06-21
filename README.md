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

## Phase 2: streaming demo

The Phase 2 CLI adds a local streaming smoke test over a WAV file. It chunks the
audio, runs the existing streaming session, and prints `stable`, `partial`,
`final`, and metrics lines.

Language auto-detection remains the default; `--language` is only for forced
runs when you want reproducible language-specific behavior. For a native Mac
demo, start with `--model small` and drop to `tiny` if the small model's real-
time factor is still above 1. For the portable benchmark path, use
`--device cpu --compute-type int8`. Docker packaging is intentionally deferred
to Phase 5.

`benchmark` accepts either one or more WAV paths via `--audio` or a JSONL
manifest via `--manifest`.

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

`--fake-transcript` keeps the command model-free for demos and tests. When it is
omitted, the CLI wires through `WhisperTranscriber` for the real transcription
path. Diarization and a UI sit in later phases and are not part of this step.

## Phase 3: speaker diarization workflow

Phase 3 adds file-level, offline speaker diarization for saved audio. It is
useful for post-call analysis and evaluation, but it does not provide rolling
online speaker identities for the live service. Stable speaker identities remain
part of the live streaming phase.

Install the extra dependencies with:

```bash
pip install -e ".[dev,diarization]"
```

Before running diarization, set `HF_TOKEN` in your shell to a valid Hugging Face
access token. Do not paste a real token into docs or commands. For example:

```bash
export HF_TOKEN=hf_your_token_here
```

The Community-1 model is available here:
[pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)

You must accept the model agreement on Hugging Face before the CLI can use it.

```bash
# File-level diarization for a local audio file
python -m cscall.cli diarize --audio call.wav

# Compare against a ground-truth RTTM file when you have one
python -m cscall.cli diarize --audio call.wav --reference-rttm call.rttm

# Transcribe and attribute speakers for a local audio file
python -m cscall.cli transcribe-speakers --audio call.wav --model small
```
