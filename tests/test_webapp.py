from __future__ import annotations

import wave
from pathlib import Path

import pytest

from cscall.fusion import SpeakerTurn, TimedWord
from cscall.webapp import SpeakerTranscriptionService


def _write_pcm_wav(
    path: Path,
    *,
    frames: int = 8000,
    sample_rate: int = 8000,
    channels: int = 1,
    sample_width: int = 2,
) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00" * frames * channels * sample_width)


class RecordingClock:
    def __init__(self, values: list[float]):
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


class RecordingLock:
    def __init__(self, log: list[object]):
        self.log = log
        self.depth = 0

    def __enter__(self):
        self.log.append("lock_enter")
        self.depth += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        self.depth -= 1
        self.log.append("lock_exit")
        return False


def test_transcribe_wav_orders_diarization_before_asr_and_groups_words(
    tmp_path: Path,
):
    audio_path = tmp_path / "sample.wav"
    _write_pcm_wav(audio_path)

    log: list[object] = []
    lock = RecordingLock(log)

    class FakeDiarizer:
        def diarize(self, path: str):
            log.append(("diarize", path, lock.depth))
            assert lock.depth == 1
            return [
                SpeakerTurn(0.0, 0.25, "SPEAKER_00"),
                SpeakerTurn(0.25, 1.0, "SPEAKER_01"),
            ]

    class FakeTranscriber:
        def transcribe_words(self, path: str):
            log.append(("transcribe", path, lock.depth))
            assert lock.depth == 1
            return [
                TimedWord(0.0, 0.1, "hello"),
                TimedWord(0.1, 0.2, "how"),
                TimedWord(0.3, 0.4, "there"),
            ]

    service = SpeakerTranscriptionService(
        transcriber=FakeTranscriber(),
        diarizer=FakeDiarizer(),
        clock=RecordingClock([100.0, 100.842]),
        lock=lock,
    )

    result = service.transcribe_wav(audio_path)

    assert log == [
        "lock_enter",
        ("diarize", str(audio_path), 1),
        ("transcribe", str(audio_path), 1),
        "lock_exit",
    ]
    assert result == {
        "segments": [
            {
                "start": 0.0,
                "end": 0.2,
                "speaker": "SPEAKER_00",
                "text": "hello how",
            },
            {
                "start": 0.3,
                "end": 0.4,
                "speaker": "SPEAKER_01",
                "text": "there",
            },
        ],
        "processing_ms": 842,
        "audio_seconds": pytest.approx(1.0),
        "rtf": pytest.approx(0.842),
    }


def test_transcribe_wav_uses_lazy_factories_once_and_reuses_objects(tmp_path: Path):
    audio_path = tmp_path / "sample.wav"
    _write_pcm_wav(audio_path)

    log: list[object] = []
    lock = RecordingLock(log)
    transcriber_factory_calls = 0
    diarizer_factory_calls = 0

    class FactoryDiarizer:
        def __init__(self):
            self.calls: list[str] = []

        def diarize(self, path: str):
            self.calls.append(path)
            return [SpeakerTurn(0.0, 1.0, "SPEAKER_00")]

    class FactoryTranscriber:
        def __init__(self):
            self.calls: list[str] = []

        def transcribe_words(self, path: str):
            self.calls.append(path)
            return [TimedWord(0.0, 0.5, "hello")]

    diarizer_instance: FactoryDiarizer | None = None
    transcriber_instance: FactoryTranscriber | None = None

    def diarizer_factory():
        nonlocal diarizer_factory_calls, diarizer_instance
        diarizer_factory_calls += 1
        diarizer_instance = FactoryDiarizer()
        return diarizer_instance

    def transcriber_factory():
        nonlocal transcriber_factory_calls, transcriber_instance
        transcriber_factory_calls += 1
        transcriber_instance = FactoryTranscriber()
        return transcriber_instance

    service = SpeakerTranscriptionService(
        transcriber_factory=transcriber_factory,
        diarizer_factory=diarizer_factory,
        clock=RecordingClock([10.0, 10.5, 20.0, 20.5]),
        lock=lock,
    )

    first = service.transcribe_wav(audio_path)
    second = service.transcribe_wav(audio_path)

    assert diarizer_factory_calls == 1
    assert transcriber_factory_calls == 1
    assert diarizer_instance is not None
    assert transcriber_instance is not None
    assert diarizer_instance.calls == [str(audio_path), str(audio_path)]
    assert transcriber_instance.calls == [str(audio_path), str(audio_path)]
    assert first == second


def test_transcribe_wav_rejects_non_pcm_wav_before_model_work(
    tmp_path: Path,
):
    audio_path = tmp_path / "bad.wav"
    audio_path.write_text("not a wav")

    lock_log: list[object] = []
    lock = RecordingLock(lock_log)

    def transcriber_factory():
        raise AssertionError("transcriber factory should not be called")

    def diarizer_factory():
        raise AssertionError("diarizer factory should not be called")

    service = SpeakerTranscriptionService(
        transcriber_factory=transcriber_factory,
        diarizer_factory=diarizer_factory,
        clock=RecordingClock([0.0, 0.1]),
        lock=lock,
    )

    with pytest.raises(ValueError, match="supported PCM WAV"):
        service.transcribe_wav(audio_path)

    assert lock_log == []


def test_transcribe_wav_allows_empty_words_and_segments(tmp_path: Path):
    audio_path = tmp_path / "silent.wav"
    _write_pcm_wav(audio_path)

    class EmptyDiarizer:
        def diarize(self, path: str):
            return []

    class EmptyTranscriber:
        def transcribe_words(self, path: str):
            return []

    service = SpeakerTranscriptionService(
        transcriber=EmptyTranscriber(),
        diarizer=EmptyDiarizer(),
        clock=RecordingClock([2.0, 2.25]),
        lock=RecordingLock([]),
    )

    result = service.transcribe_wav(audio_path)

    assert result == {
        "segments": [],
        "processing_ms": 250,
        "audio_seconds": pytest.approx(1.0),
        "rtf": pytest.approx(0.25),
    }


def test_transcribe_wav_holds_the_lock_around_both_model_calls(tmp_path: Path):
    audio_path = tmp_path / "locked.wav"
    _write_pcm_wav(audio_path)

    log: list[object] = []
    lock = RecordingLock(log)

    class GuardedDiarizer:
        def diarize(self, path: str):
            log.append(("diarize", lock.depth))
            assert lock.depth == 1
            return [SpeakerTurn(0.0, 1.0, "SPEAKER_00")]

    class GuardedTranscriber:
        def transcribe_words(self, path: str):
            log.append(("transcribe", lock.depth))
            assert lock.depth == 1
            return [TimedWord(0.0, 0.5, "hello")]

    service = SpeakerTranscriptionService(
        transcriber=GuardedTranscriber(),
        diarizer=GuardedDiarizer(),
        clock=RecordingClock([50.0, 50.5]),
        lock=lock,
    )

    service.transcribe_wav(audio_path)

    assert log == [
        "lock_enter",
        ("diarize", 1),
        ("transcribe", 1),
        "lock_exit",
    ]
