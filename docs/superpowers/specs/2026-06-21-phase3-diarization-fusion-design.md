# Phase 3 Diarization and Speaker Fusion Design

**Date:** 2026-06-21  
**Status:** Approved for specification review

## Goal

Add local two-speaker diarization and fuse speaker turns with timestamped Whisper
words, producing a speaker-attributed transcript that can later be reused by the
browser service.

## Scope

Phase 3 will:

- load `pyannote/speaker-diarization-community-1` using `HF_TOKEN`;
- diarize a WAV file with exactly two expected speakers;
- convert pyannote output into small project-owned speaker-turn records;
- transcribe with faster-whisper word timestamps;
- assign each word to the speaker turn with maximum temporal overlap;
- expose `diarize` and `transcribe-speakers` CLI commands;
- compute diarization error rate when an RTTM reference is supplied;
- test all project logic without downloading models.

Phase 3 will not:

- add true incremental/online diarization;
- identify real speaker names or roles automatically;
- add browser capture, WebSockets, or Docker;
- require a Hugging Face token for normal imports or the existing test suite.

## Dependency Strategy

`pyannote.audio` is an optional `diarization` dependency. The core package and
existing commands remain usable without it.

The wrapper imports pyannote lazily when diarization is requested. Missing
dependency and missing token errors must say how to install/configure them.
`HF_TOKEN` is read from the environment and is never accepted as a CLI argument,
printed, persisted, or committed.

The selected model is `pyannote/speaker-diarization-community-1`. The user must
accept its Hugging Face conditions before first download. The wrapper passes
`num_speakers=2`.

`diart` is not used because its released dependency stack targets an older
pyannote generation. A hand-built embedding/clustering pipeline is also out of
scope.

## Data Model

Project-owned immutable records keep pyannote types out of fusion and CLI code:

```python
@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class TimedWord:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class SpeakerWord:
    start: float
    end: float
    text: str
    speaker: str
```

Times are seconds from the beginning of the audio. Constructors reject negative
times and intervals where `end < start`. Empty word text is ignored at adapter
boundaries rather than stored.

## Diarization Adapter

`PyannoteDiarizer` owns model loading and inference:

1. Read `HF_TOKEN`.
2. Lazily import `pyannote.audio.Pipeline`.
3. Load Community-1 with the token.
4. Run the pipeline on the WAV with `num_speakers=2`.
5. Prefer `exclusive_speaker_diarization`; fall back to
   `speaker_diarization` for compatible output objects.
6. Convert each segment to `SpeakerTurn`.
7. Sort turns by start time, end time, and label.

The adapter accepts an injected pipeline in tests so tests never import PyTorch
or download models.

Speaker labels remain model-generated stable IDs such as `SPEAKER_00` and
`SPEAKER_01`. Role mapping to `Agent`/`Customer` is deferred because automatic
role inference is not justified by the available data.

## Timestamped ASR Adapter

The existing `WhisperTranscriber.transcribe(path) -> str` API remains unchanged.
A new `transcribe_words(path) -> list[TimedWord]` method invokes faster-whisper
with `word_timestamps=True` and converts segment words.

Words missing a start or end timestamp are skipped. Leading/trailing whitespace
is stripped; empty results are skipped. The existing model instance and language
setting are reused.

## Fusion

For every word:

1. Compute overlap with every speaker turn:
   `max(0, min(word.end, turn.end) - max(word.start, turn.start))`.
2. Choose the turn with greatest overlap.
3. Break equal-overlap ties by earliest turn start, then speaker label.
4. If every overlap is zero, choose the speaker turn whose interval is nearest
   to the word midpoint.
5. If there are no turns, assign `UNKNOWN`.

This is intentionally an O(words × turns) scan. Calls are short and two-speaker;
an interval index is unnecessary until profiling proves otherwise.

Adjacent words with the same speaker are grouped into transcript lines. Output
is deterministic:

```text
[00:01.20–00:03.40] SPEAKER_00: hello how can I help
[00:03.45–00:06.10] SPEAKER_01: mera order late hai
```

## CLI

### `diarize`

```bash
HF_TOKEN=... python -m cscall.cli diarize --audio call.wav
```

Prints:

```text
00:01.20	00:03.40	SPEAKER_00
00:03.45	00:06.10	SPEAKER_01
```

Optional `--reference-rttm path.rttm` prints a DER summary after turns.

### `transcribe-speakers`

```bash
HF_TOKEN=... python -m cscall.cli transcribe-speakers \
  --audio call.wav --model small
```

Supports the existing model, device, compute-type, and language options. It
runs timestamped ASR and diarization over the same validated WAV, fuses the
results, and prints grouped speaker-attributed lines.

Both commands validate the WAV before model construction.

## DER Evaluation

DER support uses pyannote's metric objects only inside the optional adapter.
The project parses RTTM rows into reference speaker turns, converts reference and
hypothesis turns to annotations, and computes DER with no collar and overlap
included. The CLI reports a single percentage with the reference path.

DER is reported only when a real reference RTTM is supplied. No synthetic DER
number will be presented as a project result.

## Error Handling

- Missing/unreadable WAV: preserve the path and operating-system error.
- Malformed/unsupported WAV: use the existing supported-PCM `ValueError`.
- Missing `pyannote.audio`: raise a concise error with
  `pip install -e ".[diarization]"`.
- Missing `HF_TOKEN`: fail before attempting model download.
- Gated model access not accepted or token rejected: preserve the underlying
  exception while adding the Community-1 model name and acceptance instruction.
- Empty/silent audio: zero turns and zero words produce no transcript lines and
  do not crash.
- Invalid RTTM: include file path and line number in the error.

## Testing

Default tests remain CPU-only and model-free.

Required tests:

- adapter prefers exclusive diarization and falls back correctly;
- adapter passes `num_speakers=2`;
- token and optional-dependency failures are actionable;
- faster-whisper word conversion handles missing timestamps and whitespace;
- overlap assignment, tie breaking, nearest-turn fallback, and `UNKNOWN`;
- grouping produces deterministic timestamped speaker lines;
- RTTM parsing validates malformed rows;
- CLI validates input before model construction and prints stable output;
- existing 124 tests remain green.

One manual smoke command downloads/runs Community-1 after the model agreement is
accepted. It is not part of the default suite.

## Completion Criteria

Phase 3 is complete when:

1. all automated tests pass without requiring `HF_TOKEN`;
2. `diarize` runs against a real WAV with Community-1 and prints two-speaker turns;
3. `transcribe-speakers` prints timestamped, speaker-attributed text;
4. a supplied RTTM reference produces DER;
5. README documents token setup, model agreement, commands, and the offline
   diarization limitation;
6. no token or model cache artifact is tracked by Git.

## Deferred Work

- Online diarization over rolling windows.
- Stable speaker identity reconciliation across WebSocket utterances.
- Agent/customer role identification.
- Browser UI, service transport, and container packaging.
