import pytest

from cscall.streaming.metrics import MetricsTracker, StreamingMetrics, summarize_metrics


def test_rtf_is_deterministic_and_zero_without_audio():
    metrics = StreamingMetrics()

    assert metrics.audio_ms == 0
    assert metrics.decode_ms == 0
    assert metrics.rtf == 0.0


def test_rtf_uses_total_decode_over_total_audio():
    tracker = MetricsTracker()

    tracker.add_decode(audio_ms=1200, decode_ms=300)
    tracker.add_decode(audio_ms=800, decode_ms=200)

    metrics = tracker.snapshot()

    assert metrics.audio_ms == 2000
    assert metrics.decode_ms == 500
    assert metrics.rtf == 0.25


def test_rtf_p50_averages_even_sample_pair():
    summary = summarize_metrics(
        [
            StreamingMetrics(audio_ms=1000, decode_ms=100),
            StreamingMetrics(audio_ms=1000, decode_ms=300),
        ]
    )

    assert summary["rtf"]["p50"] == 0.2


def test_first_partial_latency_is_recorded_only_once():
    tracker = MetricsTracker()

    tracker.mark_utterance_start(timestamp_ms=1000)
    tracker.mark_first_partial(timestamp_ms=1150)
    tracker.mark_first_partial(timestamp_ms=1300)

    metrics = tracker.snapshot()

    assert metrics.first_partial_latency_ms == 150


def test_final_latency_is_endpoint_to_final():
    tracker = MetricsTracker()

    tracker.mark_utterance_start(timestamp_ms=1000)
    tracker.mark_endpoint(timestamp_ms=2400)
    tracker.mark_final(timestamp_ms=2600)

    metrics = tracker.snapshot()

    assert metrics.final_latency_ms == 200


def test_render_contains_summary_fields():
    metrics = StreamingMetrics(
        audio_ms=2500,
        decode_ms=625,
        first_partial_latency_ms=180,
        final_latency_ms=240,
    )

    rendered = metrics.render()

    assert "RTF" in rendered
    assert "2500" in rendered
    assert "625" in rendered
    assert "180" in rendered
    assert "240" in rendered


def test_summary_percentiles_are_deterministic():
    values = [
        StreamingMetrics(
            audio_ms=1000,
            decode_ms=100,
            first_partial_latency_ms=100,
            final_latency_ms=10,
        ),
        StreamingMetrics(
            audio_ms=1000,
            decode_ms=200,
            first_partial_latency_ms=200,
            final_latency_ms=20,
        ),
        StreamingMetrics(
            audio_ms=1000,
            decode_ms=300,
            first_partial_latency_ms=300,
            final_latency_ms=30,
        ),
    ]

    summary = summarize_metrics(values)

    assert summary["rtf"]["p50"] == 0.2
    assert summary["rtf"]["p99"] == 0.3
    assert summary["first_partial_ms"]["p50"] == 200
    assert summary["first_partial_ms"]["p99"] == 300
    assert summary["final_ms"]["p50"] == 20
    assert summary["final_ms"]["p99"] == 30


def test_summary_ignores_none_latency_values():
    values = [
        StreamingMetrics(audio_ms=1000, decode_ms=100),
        StreamingMetrics(
            audio_ms=1000,
            decode_ms=200,
            first_partial_latency_ms=120,
            final_latency_ms=24,
        ),
    ]

    summary = summarize_metrics(values)

    assert summary["rtf"]["p50"] == pytest.approx(0.15)
    assert summary["first_partial_ms"] == {"p50": 120, "p99": 120}
    assert summary["final_ms"] == {"p50": 24, "p99": 24}


def test_summary_uses_none_for_empty_samples():
    summary = summarize_metrics([])

    assert summary["rtf"] == {"p50": None, "p99": None}
    assert summary["first_partial_ms"] == {"p50": None, "p99": None}
    assert summary["final_ms"] == {"p50": None, "p99": None}
