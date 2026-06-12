import pytest

from cscall.asr_baseline import WhisperTranscriber


def test_transcriber_constructs_callable():
    # Build with the tiny model; should expose a transcribe(path)->str callable.
    t = WhisperTranscriber(model_size="tiny", compute_type="int8")
    assert callable(t.transcribe)


@pytest.mark.slow
def test_transcribe_returns_string_on_real_audio():
    t = WhisperTranscriber(model_size="tiny", compute_type="int8")
    out = t.transcribe("tests/fixtures/audio/a.wav")
    assert isinstance(out, str)  # a pure tone yields "" or noise text; type is the contract
