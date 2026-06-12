# Design: Real-Time Speaker-Attributed Code-Switch ASR for Support Calls

**Date:** 2026-06-12
**Status:** Approved (design phase)
**Author:** (portfolio / flagship project)

## 1. Purpose & Context

A portfolio flagship for MLE / Data Scientist roles, deliberately **outside the LLM crowd**, in the higher-scarcity **voice/ASR** space. It fuses three sub-goals into one coherent product:

1. **Accuracy on hard audio** — robustness to Indian-accented English and Hindi-English code-switching (the differentiated, domain-backed research story).
2. **Streaming / low latency** — real-time transcription (the production-systems flex).
3. **Speaker attribution** — online diarization ("who spoke when").

**Headline framing:** *Real-time, speaker-attributed transcription for Indian-accented, Hindi-English code-switched support calls.*

**Primary use case:** **Live agent-assist** — transcribe + label speakers in real time *during* a call. This commits the whole pipeline to streaming, including online (incremental) diarization.

### Signals this project is built to demonstrate
- Production systems depth (streaming, latency budget, backpressure, containerized service).
- Research / modeling chops (LoRA fine-tuning, rigorous eval, code-switch normalization, accuracy↔latency analysis).
- Breadth across the stack (data → model → fusion → API → web UI → Docker).

### Scope boundaries (YAGNI)
- Languages limited to **Hindi-English code-switch + Indian-accented English**. Not all 22 Indian languages.
- Speakers capped at **2** (agent + customer).
- No real customer/call data (privacy/legal); see data plan.

## 2. Architecture & Components

Two paths that **share the same ASR model and eval code** — the live service runs exactly the model the offline harness scored.

```
                          ┌──────────── LIVE PATH (real-time) ────────────┐
Browser mic
 (AudioWorklet,16kHz) ──WS──▶ [1] Ingest ──▶ [2] Silero VAD / endpointing
                                                   │
                                    ┌──────────────┴───────────────┐
                                    ▼                              ▼
                      [3] Streaming Whisper            [4] diart online
                      (faster-whisper + CT2,           diarization
                       LocalAgreement partials)        (who-spoke-when)
                                    │                              │
                                    └──────────┬───────────────────┘
                                               ▼
                                   [5] Fusion (align word ts ↔ speaker turns)
                                               │
                                    ──WS──▶ [6] Web UI
                                   (live captions, speaker colors, p50/p99)

                          ┌──────────── OFFLINE PATH (the research) ──────────┐
[8] Fine-tune (Colab/T4): data prep → LoRA Whisper → export to CTranslate2
[7] Eval harness: WER/CER (jiwer), DER (pyannote.metrics), RTF → tables/plots
     baseline (vanilla Whisper) vs fine-tuned — the headline numbers
```

**Components**
1. **Ingest** — FastAPI WebSocket server receiving 16 kHz PCM frames.
2. **VAD / endpointing** — Silero VAD gates speech, detects utterance ends (~600 ms trailing silence); suppresses silence hallucination.
3. **Streaming ASR** — fine-tuned Whisper via `faster-whisper` (CTranslate2), wrapped in a **LocalAgreement-2** policy emitting stable partials + finals with word timestamps.
4. **Online diarization** — `diart` incremental pipeline (segmentation → embeddings → online clustering), capped at 2 speakers.
5. **Fusion** — aligns ASR word timestamps to diarization turns (max-overlap) → speaker-attributed transcript.
6. **Web UI** — live captions, colored speaker labels, partial/final state, latency readout.
7. **Eval harness** — WER/CER/DER/RTF, baseline-vs-fine-tuned tables and plots. *The resume artifact.*
8. **Fine-tune pipeline** — Colab notebook: data prep → LoRA → eval → CT2 export.

### Chosen approach (and the rejected alternatives)
**Approach A (chosen): Whisper + chunked LocalAgreement streaming + `diart`.**
- Whisper is the easiest model to *measurably move WER on* for code-switch (mature ecosystem, HF Trainer), fits T4 + Apple Silicon, and `diart` is a mature online diarizer.
- **Known tradeoff:** streaming is chunk-*simulated*, not a native streaming architecture. This is intentional and defensible; the "when would I switch to a true streaming transducer" discussion (Approach B) is a deliberate interview talking point.

**Approach B (rejected): NeMo Conformer-RNNT cache-aware streaming.** Natively streaming, strongest pure-systems story, but much harder to fine-tune for code-switch on a T4, weaker code-switch resources, heavier at runtime, awkward on Mac — it weakens the most differentiated (research) signal and raises stall risk.

**Approach C (rejected): two ASR models (accuracy + true-streaming).** Over-engineered; YAGNI.

## 3. Data & Evaluation Design

Three data assets:

**A. Public benchmarks (recognizable headline numbers)**
- **Svarah** (AI4Bharat) — ~9.6 hrs Indian-accented English, 117 speakers → accented-English WER.
- **Hindi-English code-switch corpus** — **MUCS 2021** / **Microsoft Interspeech code-switching challenge** (Hindi-English) → code-switch WER.
- **Risk + mitigation:** some require a free access request and availability shifts. Phase 0 verifies download access and selects a **fallback public Hinglish set** so the project never blocks on one dataset.

**B. Self-recorded support mini-corpus (~1–3 hrs)** — role-played agent↔customer Hinglish support calls (refunds, late orders), **2 speakers, speaker-labeled turns**. Triple duty:
- **Required for DER** (public ASR sets are not diarized/conversational).
- The live-demo audio.
- The "I built and labeled a dataset" interview story.

**C. Metrics & methodology**
- **WER/CER** via `jiwer`, with an **explicit code-switch normalization scheme**: transliterate to one canonical script (romanized) before scoring, to avoid Devanagari-vs-romanized mismatch inflating WER. (Methodological rigor that signals seniority.)
- **DER** via `pyannote.metrics` on corpus B.
- **RTF + p50/p99 latency** on the streaming path.
- **Breakdowns** by accent region and by **code-switch density** — the memorable tables.

**Headline experiment:** vanilla Whisper (baseline WER) → LoRA fine-tuned (improved WER). Baseline measured in **Phase 0**, before fine-tuning, to lock the "before" number and de-risk.

**Expectation-setting:** the fine-tuning delta may be large or modest — that is research. The artifact is the *rigor* (real eval set, normalization, breakdowns), impressive regardless of the exact delta.

## 4. Streaming & Latency Design

- **Capture → transport:** Browser `AudioWorklet` captures mic at 48 kHz → downsample to 16 kHz mono PCM16 → ~100–200 ms frames over WebSocket (raw Web Audio API, no audio libs).
- **VAD + endpointing:** Silero VAD per-frame speech/non-speech; ~600 ms trailing-silence timer marks utterance ends (triggers finalize + suppresses hallucination).
- **Streaming ASR — LocalAgreement-2:** rolling buffer of the in-progress utterance; re-run faster-whisper on the *growing* buffer every ~0.5–1 s; **commit the longest agreed prefix as stable (final)**, show the unstable tail as volatile partial; flush on endpoint. (Macháček `whisper-streaming` policy.)
- **Online diarization:** `diart` on a rolling ~5 s window / 500 ms step; incremental clustering into stable IDs; capped at 2.
- **Fusion across two async streams:** shared sample-aligned timeline; each committed word assigned to the max-overlap speaker turn; a reconciliation buffer holds a word until both transcript and speaker label exist, so words never visibly jump speakers.
- **Latency targets (to measure, not promise):** partial-commit **p50 < ~1 s, p99 < ~2 s**; **RTF < 1**.
- **Backpressure:** if RTF > 1, widen step / drop intermediate re-decodes; surface live RTF in UI.
- **Tunable knobs → deliverable plot:** chunk/step size, LocalAgreement *n*, VAD aggressiveness, beam size, model size (small vs medium) → an **accuracy↔latency curve**.

## 5. Error Handling

- **WebSocket drop/reconnect:** per-connection session state; on disconnect, flush in-progress partial as final and tear down ASR/diart state; client auto-reconnects with a fresh session.
- **Silence / hallucination:** VAD gating + Whisper `no_speech_threshold` + repeated-n-gram suppression.
- **RTF > 1 backpressure:** widen step / drop intermediate re-decodes; live RTF in UI.
- **>2 speakers / speaker drift:** hard cap at 2, deterministic fallback labeling.
- **Dataset access failure:** Phase-0 fallback dataset.
- **Bad audio (wrong rate, empty, garbage):** validate + defensively resample at ingest.
- **Cold start:** warm model on startup; `/health` endpoint for the container.

## 6. Testing

- **Unit (TDD-friendly, pure logic):** LocalAgreement commit function (synthetic hypothesis sequences → asserted committed prefix), fusion word↔speaker alignment, VAD state machine, code-switch normalization scheme.
- **Integration:** push a known WAV through the full streaming pipeline in simulated real-time; assert transcript converges to expected text + speaker labels.
- **Regression / eval gate:** WER/DER thresholds in CI; golden test asserting fine-tuned ≥ baseline on a held-out micro-set.
- **Latency test:** assert RTF < 1 on representative audio.

## 7. Build Milestones

Ordered so a defensible resume artifact exists as early as Phase 1, before the live system. Each phase likely becomes its own implementation plan.

- **Phase 0 — Scaffold + data + baseline:** repo/env, download benchmarks (+fallback), record/label mini-corpus, **baseline vanilla-Whisper WER + DER** (locks the "before" number).
- **Phase 1 — Fine-tune:** Colab LoRA pipeline, normalization, **fine-tuned WER**, CT2 export, breakdown tables (the research artifact).
- **Phase 2 — Streaming ASR:** VAD + LocalAgreement + latency harness + accuracy↔latency curve.
- **Phase 3 — Diarization + fusion:** `diart` integration, DER, speaker-attributed transcript.
- **Phase 4 — Web UI:** live captions, speaker colors, latency readout.
- **Phase 5 — Package:** Dockerize + WebSocket API + README writeup with all result tables.

## 8. Proposed Repo Structure

```
.
├── README.md                  # the writeup: framing + result tables/plots
├── data/                      # download scripts, manifests, self-recorded corpus
├── finetune/                  # Colab notebook + LoRA training/export scripts
├── asr/                       # streaming Whisper wrapper, LocalAgreement
├── diarization/               # diart wrapper
├── fusion/                    # word↔speaker alignment
├── server/                    # FastAPI WebSocket service, /health
├── web/                       # browser UI (AudioWorklet, live captions)
├── eval/                      # WER/CER/DER/RTF harness, results/
├── docker/                    # Dockerfile, docker-compose
└── docs/superpowers/specs/    # this design + future plans
```

## 9. Tech Stack Summary

| Concern | Choice |
|---|---|
| ASR model | Whisper small/medium, LoRA fine-tuned |
| ASR runtime | `faster-whisper` (CTranslate2) |
| Streaming policy | LocalAgreement-2 (`whisper-streaming`) |
| VAD | Silero VAD |
| Diarization | `diart` (online) |
| Metrics | `jiwer` (WER/CER), `pyannote.metrics` (DER) |
| Service | FastAPI + WebSockets |
| Frontend | Web Audio API / AudioWorklet (no framework required) |
| Packaging | Docker + docker-compose |
| Training compute | Colab/Kaggle free T4; inference on Apple Silicon |
