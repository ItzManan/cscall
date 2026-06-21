from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from cscall.streaming.endpointing import EndpointDetector
from cscall.streaming.local_agreement import AgreementUpdate, LocalAgreement
from cscall.streaming.metrics import MetricsTracker


@dataclass(frozen=True)
class AudioChunk:
    timestamp_ms: int
    duration_ms: int
    data: bytes
    is_speech: bool


@dataclass(frozen=True)
class StreamingEvent:
    type: str
    timestamp_ms: int
    text: str = ""
    metrics: object | None = None


class StreamingSession:
    def __init__(
        self,
        transcribe: Callable[[bytes], str],
        step_ms: int = 500,
        agreement: int = 2,
        endpoint_detector: EndpointDetector | None = None,
        decode_ms: int | Callable[[bytes], int] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ):
        if step_ms < 1:
            raise ValueError("step_ms must be at least 1")

        self._transcribe = transcribe
        self._step_ms = step_ms
        self._decode_ms = decode_ms
        self._clock = clock
        self._endpoint_detector = endpoint_detector or EndpointDetector()
        self._agreement_size = agreement
        self._agreement = LocalAgreement(agreement=agreement)
        self._metrics = MetricsTracker()
        self._buffer = bytearray()
        self._buffer_ms = 0
        self._audio_since_decode_ms = 0
        self._skip_next_decode = False
        self._skip_decode_debt = False
        self._in_utterance = False
        self._final_text = ""

    def update(self, chunk: AudioChunk) -> list[StreamingEvent]:
        events: list[StreamingEvent] = []
        endpoint_events = self._endpoint_detector.update(
            chunk.timestamp_ms, chunk.is_speech
        )

        started = any(event.type == "start" for event in endpoint_events)
        if started:
            self._begin_utterance(chunk.timestamp_ms)

        if chunk.is_speech and (self._in_utterance or started):
            self._buffer.extend(chunk.data)
            self._buffer_ms += chunk.duration_ms
            self._audio_since_decode_ms += chunk.duration_ms

            if self._audio_since_decode_ms >= self._step_ms:
                if self._skip_next_decode:
                    self._skip_next_decode = False
                    self._skip_decode_debt = True
                    self._audio_since_decode_ms -= self._step_ms
                else:
                    events.extend(self._decode(chunk.timestamp_ms))
                    self._audio_since_decode_ms -= self._step_ms
                    if self._skip_decode_debt:
                        self._audio_since_decode_ms -= self._step_ms

        if any(event.type == "endpoint" for event in endpoint_events):
            events.extend(self._finalize(chunk.timestamp_ms))

        return events

    def _begin_utterance(self, timestamp_ms: int) -> None:
        self._agreement = LocalAgreement(agreement=self._agreement_size)
        self._buffer.clear()
        self._buffer_ms = 0
        self._audio_since_decode_ms = 0
        self._skip_next_decode = False
        self._skip_decode_debt = False
        self._in_utterance = True
        self._final_text = ""
        self._metrics.mark_utterance_start(timestamp_ms)

    def _decode(self, timestamp_ms: int) -> list[StreamingEvent]:
        audio = bytes(self._buffer)
        decode_started_at = self._clock() if self._decode_ms is None else None
        hypothesis = self._transcribe(audio)
        decode_ms = self._resolve_decode_ms(audio, decode_started_at)
        self._metrics.add_decode(audio_ms=self._buffer_ms, decode_ms=decode_ms)

        update: AgreementUpdate = self._agreement.update(hypothesis)
        events: list[StreamingEvent] = []

        if update.committed:
            self._final_text = _append_text(self._final_text, update.committed)
            self._metrics.mark_first_partial(timestamp_ms)
            events.append(
                StreamingEvent(
                    type="stable",
                    timestamp_ms=timestamp_ms,
                    text=update.committed,
                )
            )

        if update.unstable:
            self._metrics.mark_first_partial(timestamp_ms)
            events.append(
                StreamingEvent(
                    type="partial",
                    timestamp_ms=timestamp_ms,
                    text=update.unstable,
                )
            )

        if self._metrics.snapshot().rtf > 1:
            self._skip_next_decode = True

        return events

    def _finalize(self, timestamp_ms: int) -> list[StreamingEvent]:
        events: list[StreamingEvent] = []
        if self._buffer and (self._audio_since_decode_ms > 0 or self._skip_decode_debt):
            events.extend(self._decode(timestamp_ms))
            self._audio_since_decode_ms = 0

        self._metrics.mark_endpoint(timestamp_ms)
        self._metrics.mark_final(timestamp_ms)

        remaining_text = self._agreement.final_flush()
        final_text = _append_text(self._final_text, remaining_text)
        if final_text:
            events.append(
                StreamingEvent(
                    type="final",
                    timestamp_ms=timestamp_ms,
                    text=final_text,
                )
            )

        events.append(
            StreamingEvent(
                type="metrics",
                timestamp_ms=timestamp_ms,
                metrics=self._metrics.snapshot(),
            )
        )

        self._buffer.clear()
        self._buffer_ms = 0
        self._audio_since_decode_ms = 0
        self._skip_next_decode = False
        self._skip_decode_debt = False
        self._in_utterance = False
        self._final_text = ""
        return events

    def _resolve_decode_ms(self, audio: bytes, started_at: float | None) -> int:
        if self._decode_ms is None:
            assert started_at is not None
            return round((self._clock() - started_at) * 1000)
        if callable(self._decode_ms):
            return int(self._decode_ms(audio))
        return int(self._decode_ms)


def _append_text(existing: str, addition: str) -> str:
    if not addition:
        return existing
    return f"{existing}{addition}"
