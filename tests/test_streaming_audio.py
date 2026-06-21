import struct
import pytest

from cscall.streaming.audio import is_speech_pcm


def test_is_speech_pcm_treats_unsigned_8bit_center_as_silence():
    silence = bytes([0x80, 0x80, 0x80, 0x80])

    assert not is_speech_pcm(silence, sample_width=1, threshold=1)


def test_is_speech_pcm_respects_16bit_threshold():
    speech = struct.pack("<4h", 100, -100, 100, -100)

    assert is_speech_pcm(speech, sample_width=2, threshold=99)
    assert not is_speech_pcm(speech, sample_width=2, threshold=100)


def test_is_speech_pcm_treats_alternating_8bit_pcm_as_speech_at_default_threshold():
    speech = bytes([0x00, 0xFF, 0x00, 0xFF])

    assert is_speech_pcm(speech, sample_width=1, threshold=200)


def test_is_speech_pcm_rejects_unsupported_sample_width():
    with pytest.raises(ValueError, match="sample_width"):
        is_speech_pcm(b"\x00\x00\x00", sample_width=0, threshold=200)


def test_is_speech_pcm_rejects_incomplete_sample_frames():
    with pytest.raises(ValueError, match="whole number of frames"):
        is_speech_pcm(b"\x00\x01\x02", sample_width=2, threshold=200)
