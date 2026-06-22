from __future__ import annotations

from dataclasses import dataclass
from math import inf, isfinite


UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str

    def __post_init__(self) -> None:
        _validate_interval(self.start, self.end)


@dataclass(frozen=True, slots=True)
class TimedWord:
    start: float
    end: float
    text: str

    def __post_init__(self) -> None:
        _validate_interval(self.start, self.end)


@dataclass(frozen=True, slots=True)
class SpeakerWord:
    start: float
    end: float
    text: str
    speaker: str

    def __post_init__(self) -> None:
        _validate_interval(self.start, self.end)


def fuse_words(words: list[TimedWord], turns: list[SpeakerTurn]) -> list[SpeakerWord]:
    fused: list[SpeakerWord] = []
    for word in words:
        fused.append(
            SpeakerWord(
                word.start,
                word.end,
                word.text,
                _pick_speaker(word, turns),
            )
        )
    return fused


def group_speaker_words(words: list[SpeakerWord]) -> list[list[SpeakerWord]]:
    groups: list[list[SpeakerWord]] = []
    for word in words:
        if groups and groups[-1][-1].speaker == word.speaker:
            groups[-1].append(word)
        else:
            groups.append([word])
    return groups


def render_speaker_transcript(words: list[SpeakerWord]) -> str:
    lines = []
    for group in group_speaker_words(words):
        if not group:
            continue
        start = _format_time(group[0].start)
        end = _format_time(group[-1].end)
        speaker = group[0].speaker
        text = " ".join(word.text for word in group)
        lines.append(f"[{start}–{end}] {speaker}: {text}")
    return "\n".join(lines)


def _pick_speaker(word: TimedWord, turns: list[SpeakerTurn]) -> str:
    if not turns:
        return UNKNOWN

    best_turn: SpeakerTurn | None = None
    best_overlap = 0.0
    found_positive_overlap = False

    for turn in turns:
        overlap = _overlap(word.start, word.end, turn.start, turn.end)
        if overlap <= 0:
            continue
        if (
            not found_positive_overlap
            or overlap > best_overlap
            or (
                overlap == best_overlap
                and _turn_key(turn) < _turn_key(best_turn)
            )
        ):
            best_turn = turn
            best_overlap = overlap
            found_positive_overlap = True

    if found_positive_overlap:
        assert best_turn is not None
        return best_turn.speaker

    best_turn = None
    best_distance = inf
    for turn in turns:
        distance = _interval_distance(
            _midpoint(word.start, word.end),
            turn.start,
            turn.end,
        )
        if (
            best_turn is None
            or distance < best_distance
            or (
                distance == best_distance
                and _turn_key(turn) < _turn_key(best_turn)
            )
        ):
            best_turn = turn
            best_distance = distance

    assert best_turn is not None
    return best_turn.speaker


def _turn_key(turn: SpeakerTurn | None) -> tuple[float, str]:
    if turn is None:
        return (inf, "")
    return (turn.start, turn.speaker)


def _overlap(word_start: float, word_end: float, turn_start: float, turn_end: float) -> float:
    return max(0.0, min(word_end, turn_end) - max(word_start, turn_start))


def _midpoint(start: float, end: float) -> float:
    return (start + end) / 2


def _interval_distance(point: float, start: float, end: float) -> float:
    if point < start:
        return start - point
    if point > end:
        return point - end
    return 0.0


def _format_time(seconds: float) -> str:
    centiseconds = round(seconds * 100)
    minutes, remainder = divmod(centiseconds, 60 * 100)
    whole_seconds, fractional = divmod(remainder, 100)
    return f"{minutes:02d}:{whole_seconds:02d}.{fractional:02d}"


def _validate_interval(start: float, end: float) -> None:
    if not isfinite(start) or not isfinite(end):
        raise ValueError("timestamps must be finite")
    if start < 0 or end < 0:
        raise ValueError("timestamps must be non-negative")
    if end < start:
        raise ValueError("end must be greater than or equal to start")
