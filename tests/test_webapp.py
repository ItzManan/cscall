from __future__ import annotations

import wave
from pathlib import Path

import pytest

from cscall.fusion import SpeakerTurn, TimedWord
import cscall.webapp as webapp
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


def test_transcribe_wav_reads_final_clock_after_grouping(
    monkeypatch, tmp_path: Path
):
    audio_path = tmp_path / "timed.wav"
    _write_pcm_wav(audio_path)

    grouped = {"value": False}

    class FinalClock:
        def __init__(self):
            self.calls = 0

        def __call__(self) -> float:
            self.calls += 1
            if self.calls == 1:
                return 1.0
            assert grouped["value"], "grouping should happen before the final clock"
            return 1.25

    class FakeDiarizer:
        def diarize(self, path: str):
            return [SpeakerTurn(0.0, 1.0, "SPEAKER_00")]

    class FakeTranscriber:
        def transcribe_words(self, path: str):
            return [TimedWord(0.0, 0.5, "hello")]

    def fake_group_speaker_words(words):
        grouped["value"] = True
        return [words]

    monkeypatch.setattr(webapp, "group_speaker_words", fake_group_speaker_words)

    service = SpeakerTranscriptionService(
        transcriber=FakeTranscriber(),
        diarizer=FakeDiarizer(),
        clock=FinalClock(),
        lock=RecordingLock([]),
    )

    result = service.transcribe_wav(audio_path)

    assert result["processing_ms"] == 250
    assert grouped["value"] is True


def test_transcribe_wav_retains_falsey_injected_factories_clock_and_lock(
    monkeypatch, tmp_path: Path
):
    audio_path = tmp_path / "falsey.wav"
    _write_pcm_wav(audio_path)

    log: list[str] = []

    class FalseyClock:
        def __init__(self):
            self.values = iter([2.0, 2.25])

        def __bool__(self):
            return False

        def __call__(self) -> float:
            log.append("clock")
            return next(self.values)

    class FalseyLock:
        def __bool__(self):
            return False

        def __enter__(self):
            log.append("lock_enter")
            return self

        def __exit__(self, exc_type, exc, tb):
            log.append("lock_exit")
            return False

    class FakeDiarizer:
        def diarize(self, path: str):
            log.append("diarize")
            return [SpeakerTurn(0.0, 1.0, "SPEAKER_00")]

    class FakeTranscriber:
        def transcribe_words(self, path: str):
            log.append("transcribe")
            return [TimedWord(0.0, 0.5, "hello")]

    class FalseyFactory:
        def __init__(self, name: str, value):
            self.name = name
            self.value = value

        def __bool__(self):
            return False

        def __call__(self):
            log.append(self.name)
            return self.value

    monkeypatch.setattr(
        webapp,
        "_default_transcriber_factory",
        lambda: (_ for _ in ()).throw(AssertionError("default transcriber factory")),
    )
    monkeypatch.setattr(
        webapp,
        "_default_diarizer_factory",
        lambda: (_ for _ in ()).throw(AssertionError("default diarizer factory")),
    )

    service = SpeakerTranscriptionService(
        transcriber_factory=FalseyFactory("transcriber_factory", FakeTranscriber()),
        diarizer_factory=FalseyFactory("diarizer_factory", FakeDiarizer()),
        clock=FalseyClock(),
        lock=FalseyLock(),
    )

    result = service.transcribe_wav(audio_path)

    assert log == [
        "clock",
        "lock_enter",
        "diarizer_factory",
        "diarize",
        "transcriber_factory",
        "transcribe",
        "lock_exit",
        "clock",
    ]
    assert result["processing_ms"] == 250
