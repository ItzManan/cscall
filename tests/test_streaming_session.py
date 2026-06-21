from cscall.streaming.endpointing import EndpointConfig, EndpointDetector
from cscall.streaming.session import AudioChunk, StreamingSession


def _collect_types(events):
    return [event.type for event in events]


def test_streaming_session_emits_partial_stable_final_and_metrics_in_order():
    hypotheses = iter(["hello wor", "hello wor", "hello world"])

    def transcribe(_audio: bytes) -> str:
        return next(hypotheses)

    session = StreamingSession(
        transcribe=transcribe,
        step_ms=200,
        agreement=2,
        endpoint_detector=EndpointDetector(
            EndpointConfig(frame_ms=100, min_speech_ms=100, trailing_silence_ms=200)
        ),
        decode_ms=17,
    )

    events = []
    chunks = [
        AudioChunk(timestamp_ms=100, duration_ms=100, data=b"a", is_speech=True),
        AudioChunk(timestamp_ms=200, duration_ms=100, data=b"b", is_speech=True),
        AudioChunk(timestamp_ms=300, duration_ms=100, data=b"c", is_speech=True),
        AudioChunk(timestamp_ms=400, duration_ms=100, data=b"d", is_speech=True),
        AudioChunk(timestamp_ms=500, duration_ms=100, data=b"e", is_speech=True),
        AudioChunk(timestamp_ms=600, duration_ms=100, data=b"f", is_speech=True),
        AudioChunk(timestamp_ms=700, duration_ms=100, data=b"", is_speech=False),
        AudioChunk(timestamp_ms=800, duration_ms=100, data=b"", is_speech=False),
    ]

    for chunk in chunks:
        events.extend(session.update(chunk))

    assert _collect_types(events) == ["partial", "stable", "partial", "final", "metrics"]
    assert [event.text for event in events[:4]] == [
        "hello wor",
        "hello wor",
        "ld",
        "hello world",
    ]
    assert events[4].metrics.audio_ms > 0
    assert events[4].metrics.decode_ms > 0


def test_streaming_session_decodes_only_on_step_boundaries():
    calls = []

    def transcribe(audio: bytes) -> str:
        calls.append(audio)
        return "hello"

    session = StreamingSession(
        transcribe=transcribe,
        step_ms=300,
        agreement=2,
        endpoint_detector=EndpointDetector(
            EndpointConfig(frame_ms=100, min_speech_ms=100, trailing_silence_ms=1000)
        ),
    )

    for index in range(4):
        session.update(
            AudioChunk(
                timestamp_ms=(index + 1) * 100,
                duration_ms=100,
                data=bytes([index]),
                is_speech=True,
            )
        )

    assert len(calls) == 1

    session.update(AudioChunk(timestamp_ms=500, duration_ms=100, data=b"4", is_speech=True))
    session.update(AudioChunk(timestamp_ms=600, duration_ms=100, data=b"5", is_speech=True))

    assert len(calls) == 2


def test_streaming_session_resets_after_endpoint_and_handles_second_utterance():
    hypotheses = iter(["alpha", "beta"])

    def transcribe(_audio: bytes) -> str:
        return next(hypotheses)

    session = StreamingSession(
        transcribe=transcribe,
        step_ms=200,
        agreement=2,
        endpoint_detector=EndpointDetector(
            EndpointConfig(frame_ms=100, min_speech_ms=100, trailing_silence_ms=200)
        ),
    )

    utterance_1 = [
        AudioChunk(timestamp_ms=100, duration_ms=100, data=b"a", is_speech=True),
        AudioChunk(timestamp_ms=200, duration_ms=100, data=b"b", is_speech=True),
        AudioChunk(timestamp_ms=300, duration_ms=100, data=b"", is_speech=False),
        AudioChunk(timestamp_ms=400, duration_ms=100, data=b"", is_speech=False),
    ]
    utterance_2 = [
        AudioChunk(timestamp_ms=500, duration_ms=100, data=b"c", is_speech=True),
        AudioChunk(timestamp_ms=600, duration_ms=100, data=b"d", is_speech=True),
        AudioChunk(timestamp_ms=700, duration_ms=100, data=b"", is_speech=False),
        AudioChunk(timestamp_ms=800, duration_ms=100, data=b"", is_speech=False),
    ]

    events = []
    for chunk in utterance_1 + utterance_2:
        events.extend(session.update(chunk))

    finals = [event.text for event in events if event.type == "final"]
    assert finals == ["alpha", "beta"]


def test_streaming_session_metrics_event_uses_configured_decode_ms():
    def transcribe(_audio: bytes) -> str:
        return "metric"

    session = StreamingSession(
        transcribe=transcribe,
        step_ms=200,
        agreement=2,
        endpoint_detector=EndpointDetector(
            EndpointConfig(frame_ms=100, min_speech_ms=100, trailing_silence_ms=200)
        ),
        decode_ms=17,
    )

    events = []
    for chunk in [
        AudioChunk(timestamp_ms=100, duration_ms=100, data=b"a", is_speech=True),
        AudioChunk(timestamp_ms=200, duration_ms=100, data=b"b", is_speech=True),
        AudioChunk(timestamp_ms=300, duration_ms=100, data=b"", is_speech=False),
        AudioChunk(timestamp_ms=400, duration_ms=100, data=b"", is_speech=False),
    ]:
        events.extend(session.update(chunk))

    metrics_events = [event for event in events if event.type == "metrics"]
    assert len(metrics_events) == 1
    assert metrics_events[0].metrics.audio_ms > 0
    assert metrics_events[0].metrics.decode_ms == 17


def test_streaming_session_decodes_short_utterance_at_endpoint():
    calls = []

    def transcribe(audio: bytes) -> str:
        calls.append(audio)
        return "short"

    session = StreamingSession(
        transcribe=transcribe,
        step_ms=500,
        agreement=2,
        endpoint_detector=EndpointDetector(
            EndpointConfig(frame_ms=100, min_speech_ms=100, trailing_silence_ms=200)
        ),
    )

    events = []
    for chunk in [
        AudioChunk(timestamp_ms=100, duration_ms=100, data=b"a", is_speech=True),
        AudioChunk(timestamp_ms=200, duration_ms=100, data=b"", is_speech=False),
        AudioChunk(timestamp_ms=300, duration_ms=100, data=b"", is_speech=False),
    ]:
        events.extend(session.update(chunk))

    assert calls == [b"a"]
    assert [event.text for event in events if event.type == "final"] == ["short"]


def test_streaming_session_final_includes_already_stable_text():
    def transcribe(_audio: bytes) -> str:
        return "done"

    session = StreamingSession(
        transcribe=transcribe,
        step_ms=100,
        agreement=2,
        endpoint_detector=EndpointDetector(
            EndpointConfig(frame_ms=100, min_speech_ms=100, trailing_silence_ms=200)
        ),
    )

    events = []
    for chunk in [
        AudioChunk(timestamp_ms=100, duration_ms=100, data=b"a", is_speech=True),
        AudioChunk(timestamp_ms=200, duration_ms=100, data=b"b", is_speech=True),
        AudioChunk(timestamp_ms=300, duration_ms=100, data=b"", is_speech=False),
        AudioChunk(timestamp_ms=400, duration_ms=100, data=b"", is_speech=False),
    ]:
        events.extend(session.update(chunk))

    assert [event.text for event in events if event.type == "stable"] == ["done"]
    assert [event.text for event in events if event.type == "final"] == ["done"]


def test_streaming_session_measures_decode_time_by_default():
    clock_values = iter([1.0, 1.025])

    session = StreamingSession(
        transcribe=lambda _audio: "timed",
        step_ms=100,
        endpoint_detector=EndpointDetector(
            EndpointConfig(frame_ms=100, min_speech_ms=100, trailing_silence_ms=200)
        ),
        clock=lambda: next(clock_values),
    )

    events = []
    for chunk in [
        AudioChunk(timestamp_ms=100, duration_ms=100, data=b"a", is_speech=True),
        AudioChunk(timestamp_ms=200, duration_ms=100, data=b"", is_speech=False),
        AudioChunk(timestamp_ms=300, duration_ms=100, data=b"", is_speech=False),
    ]:
        events.extend(session.update(chunk))

    metrics = next(event.metrics for event in events if event.type == "metrics")
    assert metrics.decode_ms == 25
