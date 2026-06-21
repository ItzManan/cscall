from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EndpointConfig:
    frame_ms: int = 100
    min_speech_ms: int = 200
    trailing_silence_ms: int = 600


@dataclass(frozen=True)
class EndpointEvent:
    type: str
    timestamp_ms: int


class EndpointDetector:
    def __init__(self, config: EndpointConfig | None = None):
        self.config = config or EndpointConfig()
        self._speech_ms = 0
        self._silence_ms = 0
        self._active = False

    def update(self, timestamp_ms: int, is_speech: bool) -> list[EndpointEvent]:
        frame_ms = self.config.frame_ms
        events: list[EndpointEvent] = []

        if is_speech:
            self._silence_ms = 0
            if self._active:
                events.append(EndpointEvent(type="speech", timestamp_ms=timestamp_ms))
                return events

            self._speech_ms += frame_ms
            if self._speech_ms >= self.config.min_speech_ms:
                self._active = True
                events.append(EndpointEvent(type="start", timestamp_ms=timestamp_ms))
                events.append(EndpointEvent(type="speech", timestamp_ms=timestamp_ms))
            return events

        if not self._active:
            self._speech_ms = 0
            self._silence_ms = 0
            return events

        self._silence_ms += frame_ms
        if self._silence_ms >= self.config.trailing_silence_ms:
            events.append(EndpointEvent(type="endpoint", timestamp_ms=timestamp_ms))
            self._reset()
        return events

    def _reset(self) -> None:
        self._speech_ms = 0
        self._silence_ms = 0
        self._active = False
