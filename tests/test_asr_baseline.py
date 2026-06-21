from types import SimpleNamespace

from cscall.asr_baseline import WhisperTranscriber
from cscall.fusion import TimedWord


class FakeModel:
    def __init__(self, segments_factory):
        self._segments_factory = segments_factory
        self.calls = []

    def transcribe(self, audio_path, **kwargs):
        self.calls.append((audio_path, kwargs))
        return self._segments_factory(), SimpleNamespace(language="hi")


def build_transcriber(model, language="hi"):
    transcriber = object.__new__(WhisperTranscriber)
    transcriber._model = model
    transcriber._language = language
    return transcriber


def test_transcribe_still_joins_segment_text_from_lazy_model():
    def segments():
        yield SimpleNamespace(text="  hello ")
        yield SimpleNamespace(text="world  ")

    model = FakeModel(segments)
    transcriber = build_transcriber(model)

    assert transcriber.transcribe("clip.wav") == "hello world"
    assert model.calls == [("clip.wav", {"language": "hi"})]


def test_transcribe_words_flattens_lazy_segments_and_skips_bad_words():
    def segments():
        yield SimpleNamespace(
            words=(
                SimpleNamespace(word="  hi ", start=0.0, end=0.2),
                SimpleNamespace(word="   ", start=0.2, end=0.4),
                SimpleNamespace(word="skip start", start=None, end=0.5),
            )
        )
        yield SimpleNamespace(
            words=(
                SimpleNamespace(word="there", start=0.2, end=0.5),
                SimpleNamespace(word="skip end", start=0.6, end=None),
                SimpleNamespace(word="  friend  ", start=0.5, end=0.9),
            )
        )

    model = FakeModel(segments)
    transcriber = build_transcriber(model)

    assert transcriber.transcribe_words("clip.wav") == [
        TimedWord(0.0, 0.2, "hi"),
        TimedWord(0.2, 0.5, "there"),
        TimedWord(0.5, 0.9, "friend"),
    ]
    assert model.calls == [
        (
            "clip.wav",
            {"language": "hi", "word_timestamps": True},
        )
    ]
