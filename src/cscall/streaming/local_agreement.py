from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

from cscall.normalize import normalize_text


@dataclass
class AgreementUpdate:
    committed: str
    unstable: str


def _common_prefix(values: list[str]) -> str:
    if not values:
        return ""

    prefix = values[0]
    for value in values[1:]:
        limit = min(len(prefix), len(value))
        index = 0
        while index < limit and prefix[index] == value[index]:
            index += 1
        prefix = prefix[:index]
        if not prefix:
            break
    return prefix


class LocalAgreement:
    def __init__(self, agreement: int = 2):
        if agreement < 1:
            raise ValueError("agreement must be at least 1")
        self._agreement = agreement
        self._history: Deque[str] = deque(maxlen=agreement)
        self._committed = ""
        self._unstable = ""

    def update(self, hypothesis: str) -> AgreementUpdate:
        normalized = normalize_text(hypothesis)
        if not normalized:
            self._history.clear()
            return AgreementUpdate(committed="", unstable=self._unstable)

        self._history.append(normalized)
        previous_committed = self._committed
        stable_prefix = previous_committed
        if len(self._history) == self._agreement:
            candidate = _common_prefix(list(self._history))
            if len(candidate) > len(previous_committed) and candidate.startswith(
                previous_committed
            ):
                stable_prefix = candidate
                self._committed = candidate

        if normalized.startswith(stable_prefix):
            self._unstable = normalized[len(stable_prefix):]
        else:
            self._unstable = normalized

        if stable_prefix.startswith(previous_committed):
            committed_delta = stable_prefix[len(previous_committed):]
        else:
            committed_delta = ""
        return AgreementUpdate(committed=committed_delta, unstable=self._unstable)

    def final_flush(self) -> str:
        remaining = self._unstable
        self._history.clear()
        self._committed = ""
        self._unstable = ""
        return remaining
