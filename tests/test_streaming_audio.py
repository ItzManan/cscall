import re
import struct
import wave

import pytest

from cscall.streaming import audio


def test_is_speech_pcm_treats_unsigned_8bit_center_as_silence():
    silence = bytes([0x80, 0x80, 0x80, 0x80])

    assert not audio.is_speech_pcm(silence, sample_width=1, threshold=1)


def test_is_speech_pcm_respects_16bit_threshold():
    speech = struct.pack("<4h", 100, -100, 100, -100)

    assert audio.is_speech_pcm(speech, sample_width=2, threshold=99)
    assert not audio.is_speech_pcm(speech, sample_width=2, threshold=100)


def test_is_speech_pcm_treats_alternating_8bit_pcm_as_speech_at_default_threshold():
    speech = bytes([0x00, 0xFF, 0x00, 0xFF])

    assert audio.is_speech_pcm(speech, sample_width=1, threshold=200)


def test_is_speech_pcm_rejects_unsupported_sample_width():
    with pytest.raises(ValueError, match="sample_width"):
        audio.is_speech_pcm(b"\x00\x00\x00", sample_width=0, threshold=200)


def test_is_speech_pcm_rejects_incomplete_sample_frames():
    with pytest.raises(ValueError, match="whole number of frames"):
        audio.is_speech_pcm(b"\x00\x01\x02", sample_width=2, threshold=200)


def test_validate_pcm_wav_returns_wavinfo_for_pcm_audio(tmp_path):
    path = tmp_path / "sample.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(struct.pack("<4h", 0, 100, -100, 0))

    assert audio.validate_pcm_wav(path) == audio.WavInfo(
        sample_rate=8000, channels=1, sample_width=2
    )


class _FakeWave:
    def __init__(self, *, comptype="NONE", framerate=8000, channels=1, sampwidth=2):
        self._comptype = comptype
        self._framerate = framerate
        self._channels = channels
        self._sampwidth = sampwidth

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def getcomptype(self):
        return self._comptype

    def getframerate(self):
        return self._framerate

    def getnchannels(self):
        return self._channels

    def getsampwidth(self):
        return self._sampwidth


@pytest.mark.parametrize(
    "fake_wave",
    [
        _FakeWave(comptype="ULAW"),
        _FakeWave(framerate=0),
        _FakeWave(channels=0),
        _FakeWave(sampwidth=0),
    ],
)
def test_validate_pcm_wav_rejects_unsupported_pcm_metadata(monkeypatch, tmp_path, fake_wave):
    path = tmp_path / "bad.wav"
    path.write_bytes(b"placeholder")
    monkeypatch.setattr(wave, "open", lambda *args, **kwargs: fake_wave)

    with pytest.raises(
        ValueError, match=rf"{re.escape(str(path))} is not a supported PCM WAV"
    ):
        audio.validate_pcm_wav(path)


def test_validate_pcm_wav_rejects_unsupported_sample_width(monkeypatch, tmp_path):
    path = tmp_path / "wide.wav"
    path.write_bytes(b"placeholder")
    monkeypatch.setattr(wave, "open", lambda *args, **kwargs: _FakeWave(sampwidth=5))

    with pytest.raises(
        ValueError, match=rf"{re.escape(str(path))} is not a supported PCM WAV"
    ):
        audio.validate_pcm_wav(path)


def test_validate_pcm_wav_wraps_wave_errors(tmp_path):
    path = tmp_path / "bad.wav"
    path.write_bytes(b"not a wav")

    with pytest.raises(
        ValueError, match=rf"{re.escape(str(path))} is not a supported PCM WAV"
    ):
        audio.validate_pcm_wav(path)


def test_validate_pcm_wav_preserves_file_not_found(monkeypatch, tmp_path):
    path = tmp_path / "missing.wav"
    monkeypatch.setattr(wave, "open", lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError(str(path))))

    with pytest.raises(FileNotFoundError):
        audio.validate_pcm_wav(path)
