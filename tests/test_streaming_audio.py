import struct

from cscall.streaming.audio import is_speech_pcm


def test_is_speech_pcm_treats_unsigned_8bit_center_as_silence():
    silence = bytes([0x80, 0x80, 0x80, 0x80])

    assert not is_speech_pcm(silence, sample_width=1, threshold=1)


def test_is_speech_pcm_respects_16bit_threshold():
    speech = struct.pack("<4h", 100, -100, 100, -100)

    assert is_speech_pcm(speech, sample_width=2, threshold=99)
    assert not is_speech_pcm(speech, sample_width=2, threshold=100)
