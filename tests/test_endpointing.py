from cscall.streaming.endpointing import EndpointConfig, EndpointDetector


def _types(events):
    return [event.type for event in events]


def test_starts_after_min_speech_ms_worth_of_speech_frames():
    detector = EndpointDetector(
        EndpointConfig(frame_ms=100, min_speech_ms=200, trailing_silence_ms=600)
    )

    assert detector.update(100, True) == []
    events = detector.update(200, True)

    assert _types(events) == ["start", "speech"]
    assert [event.timestamp_ms for event in events] == [200, 200]


def test_emits_endpoint_after_trailing_silence_threshold():
    detector = EndpointDetector(
        EndpointConfig(frame_ms=100, min_speech_ms=200, trailing_silence_ms=300)
    )

    detector.update(100, True)
    detector.update(200, True)
    detector.update(300, False)
    assert detector.update(400, False) == []

    events = detector.update(500, False)

    assert _types(events) == ["endpoint"]
    assert events[0].timestamp_ms == 500


def test_ignores_speech_shorter_than_min_speech_ms():
    detector = EndpointDetector(
        EndpointConfig(frame_ms=100, min_speech_ms=300, trailing_silence_ms=600)
    )

    assert detector.update(100, True) == []
    assert detector.update(200, True) == []
    assert detector.update(300, False) == []
    assert detector.update(400, False) == []


def test_can_start_another_utterance_after_endpoint():
    detector = EndpointDetector(
        EndpointConfig(frame_ms=100, min_speech_ms=200, trailing_silence_ms=200)
    )

    detector.update(100, True)
    detector.update(200, True)
    detector.update(300, False)
    endpoint_events = detector.update(400, False)
    assert _types(endpoint_events) == ["endpoint"]

    assert detector.update(500, True) == []
    events = detector.update(600, True)

    assert _types(events) == ["start", "speech"]


def test_config_values_are_honored():
    detector = EndpointDetector(
        EndpointConfig(frame_ms=50, min_speech_ms=100, trailing_silence_ms=150)
    )

    assert detector.update(50, True) == []
    events = detector.update(100, True)
    assert _types(events) == ["start", "speech"]

    detector.update(150, False)
    assert detector.update(200, False) == []
    endpoint_events = detector.update(250, False)
    assert _types(endpoint_events) == ["endpoint"]
