from __future__ import annotations

from collections.abc import Callable
import threading
import time
import wave

from cscall.fusion import fuse_words, group_speaker_words
from cscall.streaming.audio import validate_pcm_wav


def _default_transcriber_factory():
    from cscall.asr_baseline import WhisperTranscriber

    return WhisperTranscriber()


def _default_diarizer_factory():
    from cscall.diarization import PyannoteDiarizer

    return PyannoteDiarizer()


class SpeakerTranscriptionService:
    def __init__(
        self,
        *,
        transcriber=None,
        diarizer=None,
        transcriber_factory: Callable[[], object] | None = None,
        diarizer_factory: Callable[[], object] | None = None,
        clock: Callable[[], float] | None = None,
        lock=None,
    ):
        self._transcriber = transcriber
        self._diarizer = diarizer
        self._transcriber_factory = transcriber_factory or _default_transcriber_factory
        self._diarizer_factory = diarizer_factory or _default_diarizer_factory
        self._clock = clock or time.perf_counter
        self._lock = lock or threading.Lock()

    def transcribe_wav(self, path):
        path_str = str(path)
        started = self._clock()
        validate_pcm_wav(path_str)
        audio_seconds = self._audio_seconds(path_str)

        with self._lock:
            turns = self._get_diarizer().diarize(path_str)
            words = self._get_transcriber().transcribe_words(path_str)

        finished = self._clock()
        segments = []
        for group in group_speaker_words(fuse_words(words, turns)):
            segments.append(
                {
                    "start": group[0].start,
                    "end": group[-1].end,
                    "speaker": group[0].speaker,
                    "text": " ".join(word.text for word in group),
                }
            )

        processing_ms = int(round((finished - started) * 1000))
        rtf = 0.0 if audio_seconds == 0 else (processing_ms / 1000) / audio_seconds

        return {
            "segments": segments,
            "processing_ms": processing_ms,
            "audio_seconds": audio_seconds,
            "rtf": rtf,
        }

    def _get_transcriber(self):
        if self._transcriber is None:
            self._transcriber = self._transcriber_factory()
        return self._transcriber

    def _get_diarizer(self):
        if self._diarizer is None:
            self._diarizer = self._diarizer_factory()
        return self._diarizer

    def _audio_seconds(self, path_str: str) -> float:
        with wave.open(path_str, "rb") as wav:
            return wav.getnframes() / wav.getframerate()
