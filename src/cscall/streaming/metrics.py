"""Deterministic streaming latency and throughput metrics."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from statistics import median
from typing import Iterable


@dataclass(frozen=True)
class StreamingMetrics:
    audio_ms: int = 0
    decode_ms: int = 0
    first_partial_latency_ms: int | None = None
    final_latency_ms: int | None = None

    @property
    def rtf(self) -> float:
        if self.audio_ms == 0:
            return 0.0
        return self.decode_ms / self.audio_ms

    def render(self) -> str:
        first_partial = (
            f"{self.first_partial_latency_ms} ms"
            if self.first_partial_latency_ms is not None
            else "n/a"
        )
        final_latency = (
            f"{self.final_latency_ms} ms" if self.final_latency_ms is not None else "n/a"
        )
        return (
            "Streaming metrics | "
            f"audio={self.audio_ms} ms | "
            f"decode={self.decode_ms} ms | "
            f"RTF={self.rtf:.3f} | "
            f"first_partial={first_partial} | "
            f"final={final_latency}"
        )


class MetricsTracker:
    def __init__(self):
        self._audio_ms = 0
        self._decode_ms = 0
        self._utterance_start_ms: int | None = None
        self._endpoint_ms: int | None = None
        self._first_partial_latency_ms: int | None = None
        self._final_latency_ms: int | None = None

    def add_decode(self, audio_ms: int, decode_ms: int) -> None:
        self._audio_ms += audio_ms
        self._decode_ms += decode_ms

    def mark_utterance_start(self, timestamp_ms: int) -> None:
        self._utterance_start_ms = timestamp_ms
        self._endpoint_ms = None
        self._first_partial_latency_ms = None
        self._final_latency_ms = None

    def mark_first_partial(self, timestamp_ms: int) -> None:
        if self._first_partial_latency_ms is not None:
            return
        if self._utterance_start_ms is None:
            return
        self._first_partial_latency_ms = timestamp_ms - self._utterance_start_ms

    def mark_endpoint(self, timestamp_ms: int) -> None:
        self._endpoint_ms = timestamp_ms

    def mark_final(self, timestamp_ms: int) -> None:
        if self._final_latency_ms is not None:
            return
        if self._endpoint_ms is None:
            return
        self._final_latency_ms = timestamp_ms - self._endpoint_ms

    def snapshot(self) -> StreamingMetrics:
        return StreamingMetrics(
            audio_ms=self._audio_ms,
            decode_ms=self._decode_ms,
            first_partial_latency_ms=self._first_partial_latency_ms,
            final_latency_ms=self._final_latency_ms,
        )


def summarize_metrics(metrics: Iterable[StreamingMetrics]) -> dict[str, dict[str, int | float | None]]:
    items = list(metrics)
    return {
        "rtf": _summarize_values([item.rtf for item in items]),
        "first_partial_ms": _summarize_values(
            [item.first_partial_latency_ms for item in items]
        ),
        "final_ms": _summarize_values([item.final_latency_ms for item in items]),
    }


def _summarize_values(values: Iterable[int | float | None]) -> dict[str, int | float | None]:
    samples = [value for value in values if value is not None]
    if not samples:
        return {"p50": None, "p99": None}

    sorted_samples = sorted(samples)
    return {
        "p50": median(sorted_samples),
        "p99": sorted_samples[_nearest_rank_index(len(sorted_samples))],
    }


def _nearest_rank_index(count: int) -> int:
    return max(0, ceil(0.99 * count) - 1)
