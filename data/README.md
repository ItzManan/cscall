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
