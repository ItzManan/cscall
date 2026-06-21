# Phase 2 Streaming Completion Design

**Date:** 2026-06-21  
**Status:** Approved for specification review

## Goal

Finish the existing Phase 2 streaming ASR implementation as a measurable local
demo that runs quickly on Apple Silicon and remains usable on CPU-only Mac/Linux
machines and in Docker.

## Scope

Phase 2 will:

- stream a WAV file through the existing `StreamingSession`;
- detect speech using PCM energy instead of treating every non-zero byte as speech;
- stabilize partial transcripts with the existing LocalAgreement policy;
- report RTF plus first-partial and final latency percentiles;
- skip intermediate decodes when inference is slower than real time;
- expose Whisper's language setting explicitly in all CLI workflows;
- provide a repeatable benchmark command and documented native/Docker defaults.

Phase 2 will not add microphone capture, WebSockets, diarization, speaker fusion,
or a browser UI. Those remain separate phases so this milestone stays small and
independently testable.

## Runtime Targets

- Native Apple Silicon is the primary development and demo path.
- CPU-only Mac/Linux is supported without platform-specific code.
- Docker defaults to Whisper `tiny` with CPU `int8`.
- Native execution may use Whisper `small` when measured RTF remains below 1.
- Model size, device, compute type, language, chunk size, agreement count, and
  energy threshold remain CLI knobs because audio hardware and inference speed
  vary in practice.

## Architecture

### Audio input

The CLI validates WAV input before streaming. Supported input is PCM WAV with a
positive sample rate, channel count, and sample width. Resampling and arbitrary
media decoding are deferred to the live ingest phase; the Phase 2 command fails
clearly on unsupported input.

### Speech detection

Each PCM chunk is classified using root-mean-square amplitude from Python's
standard library. A configurable threshold handles different recording levels.
This avoids adding PyTorch solely for Silero VAD. The detector remains a small
callable so Silero can replace it later if real recordings show materially worse
endpointing.

### Streaming session

The existing `EndpointDetector`, `LocalAgreement`, and `StreamingSession` remain
the core pipeline:

1. WAV frames become timestamped `AudioChunk` values.
2. Energy classification sets `is_speech`.
3. Endpointing opens and closes utterances.
4. The growing speech buffer is decoded at the configured step.
5. LocalAgreement emits stable and partial text.
6. Endpointing emits final text and metrics.

No second streaming abstraction will be introduced.

### Backpressure

After each decode, the session compares cumulative decode time with cumulative
audio time. When RTF exceeds 1, it waits one additional normal decode interval
before decoding again. Final endpoint decoding is never skipped.

This deliberately simple policy bounds wasted re-decodes without threads,
queues, or a scheduler. A more adaptive policy is justified only if benchmark
results show this one cannot keep the demo responsive.

### Metrics and benchmark

Per-utterance metrics continue to report:

- audio duration;
- cumulative decode duration;
- real-time factor;
- first-partial latency;
- endpoint-to-final latency.

A benchmark accumulator will summarize multiple utterances with p50 and p99 for
RTF, first-partial latency, and final latency. Empty samples render as `n/a`.
The CLI benchmark command runs a manifest or explicit WAV inputs and prints one
small Markdown table suitable for copying into the README.

### Explicit language regime

`baseline`, `compare`, `stream`, and `benchmark` accept `--language`. The value
is passed directly to `WhisperTranscriber`; omission preserves automatic
language detection. This makes the evaluation regime reproducible without
changing the default established by prior results.

## Error Handling

- Missing or unreadable input: fail with the file path and operating-system error.
- Invalid or unsupported WAV: fail before model loading.
- Invalid chunk size, agreement count, or energy threshold: reject during CLI
  parsing or object construction.
- Empty/silent WAV: produce no transcript and a valid zero/`n/a` metrics summary.
- Model failure: propagate the original exception after temporary WAV cleanup.

## Testing

Tests use synthetic PCM and fake transcribers so the suite remains fast and does
not download models.

Required checks:

- energy detection distinguishes silence from speech and honors its threshold;
- WAV validation rejects malformed input;
- backpressure skips only intermediate decodes and still finalizes;
- percentile summaries handle odd, even, and empty samples;
- every CLI workflow parses and forwards `--language`;
- benchmark output is deterministic;
- the existing streaming tests remain green.

One manual smoke command will exercise a real model on a committed WAV fixture.
It is documented but not part of the default test suite.

## Completion Criteria

Phase 2 is complete when:

1. the full automated test suite passes;
2. a fake-transcriber stream command emits partial/final text and metrics;
3. a real-model native smoke command completes on the local machine;
4. benchmark output includes RTF and p50/p99 latency;
5. README instructions describe native and CPU/Docker model defaults;
6. all Phase 2 source, tests, and documentation are committed together.

## Deferred Work

- Silero VAD: add only if the energy detector fails on representative calls.
- Audio resampling and browser microphone input: Phase 4/live ingest.
- Online diarization and word-speaker fusion: Phase 3.
- FastAPI, WebSockets, health checks, and Docker image: Phase 5.
