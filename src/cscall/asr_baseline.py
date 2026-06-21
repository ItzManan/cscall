"""Baseline ASR using faster-whisper (the only module touching the model runtime)."""
from faster_whisper import WhisperModel

from cscall.fusion import TimedWord


class WhisperTranscriber:
    """Wraps a faster-whisper model behind a transcribe(path) -> str callable."""

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = None,
    ):
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self._language = language

    def transcribe(self, audio_path: str) -> str:
        segments, _info = self._model.transcribe(audio_path, language=self._language)
        return " ".join(seg.text.strip() for seg in segments).strip()

    def transcribe_words(self, audio_path: str) -> list[TimedWord]:
        segments, _info = self._model.transcribe(
            audio_path,
            language=self._language,
            word_timestamps=True,
        )
        words: list[TimedWord] = []
        for segment in segments:
            for word in getattr(segment, "words", None) or []:
                text = (getattr(word, "word", "") or "").strip()
                start = getattr(word, "start", None)
                end = getattr(word, "end", None)
                if not text or start is None or end is None:
                    continue
                words.append(TimedWord(start, end, text))
        return words
