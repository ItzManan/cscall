from __future__ import annotations

import audioop


def is_speech_pcm(data: bytes, sample_width: int, threshold: int) -> bool:
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    if not data:
        return False
    if sample_width == 1:
        data = audioop.bias(data, 1, -128)
    return audioop.rms(data, sample_width) > threshold
