from __future__ import annotations

import math


def is_speech_pcm(data: bytes, sample_width: int, threshold: int) -> bool:
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    if sample_width not in (1, 2, 3, 4):
        raise ValueError("sample_width must be between 1 and 4 bytes")
    if not data:
        return False
    if len(data) % sample_width != 0:
        raise ValueError("PCM data must contain a whole number of frames")
    return _pcm_rms_16bit(data, sample_width) > threshold


def _pcm_rms_16bit(data: bytes, sample_width: int) -> int:
    total = 0
    samples = 0

    if sample_width == 1:
        for sample in data:
            centered = sample - 128
            normalized = centered * 256
            total += normalized * normalized
            samples += 1
    else:
        shift = 8 * (sample_width - 2)
        for offset in range(0, len(data), sample_width):
            sample = int.from_bytes(
                data[offset : offset + sample_width], "little", signed=True
            )
            normalized = sample if shift == 0 else sample >> shift
            total += normalized * normalized
            samples += 1

    return int(math.sqrt(total / samples)) if samples else 0
