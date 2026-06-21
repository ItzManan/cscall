from cscall.streaming.metrics import MetricsTracker, StreamingMetrics


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
