from dataclasses import FrozenInstanceError
from math import inf, nan

import pytest

from cscall.fusion import (
    SpeakerTurn,
    SpeakerWord,
    TimedWord,
    fuse_words,
    group_speaker_words,
    render_speaker_transcript,
)


@pytest.mark.parametrize(
    "factory, valid_kwargs, invalid_kwargs",
    [
        (
            SpeakerTurn,
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
            {"start": -0.1, "end": 0.0, "speaker": "SPEAKER_00"},
        ),
        (
            SpeakerTurn,
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 0.5, "speaker": "SPEAKER_00"},
        ),
        (
            TimedWord,
            {"start": 0.0, "end": 1.0, "text": "hello"},
            {"start": -0.1, "end": 0.0, "text": "hello"},
        ),
        (
            TimedWord,
            {"start": 0.0, "end": 1.0, "text": "hello"},
            {"start": 1.0, "end": 0.5, "text": "hello"},
        ),
        (
            SpeakerWord,
            {"start": 0.0, "end": 1.0, "text": "hello", "speaker": "SPEAKER_00"},
            {"start": -0.1, "end": 0.0, "text": "hello", "speaker": "SPEAKER_00"},
        ),
        (
            SpeakerWord,
            {"start": 0.0, "end": 1.0, "text": "hello", "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 0.5, "text": "hello", "speaker": "SPEAKER_00"},
        ),
    ],
)
def test_records_are_frozen_and_validate_time_bounds(
    factory, valid_kwargs, invalid_kwargs
):
    value = factory(**valid_kwargs)
    with pytest.raises(FrozenInstanceError):
        value.start = 2.0

    with pytest.raises(ValueError):
        factory(**invalid_kwargs)


@pytest.mark.parametrize(
    "factory, field, bad_value, kwargs",
    [
        (
            SpeakerTurn,
            "start",
            nan,
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
        ),
        (
            SpeakerTurn,
            "start",
            inf,
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
        ),
        (
            SpeakerTurn,
            "start",
            -inf,
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
        ),
        (
            SpeakerTurn,
            "end",
            nan,
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
        ),
        (
            SpeakerTurn,
            "end",
            inf,
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
        ),
        (
            SpeakerTurn,
            "end",
            -inf,
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
        ),
        (TimedWord, "start", nan, {"start": 0.0, "end": 1.0, "text": "hello"}),
        (TimedWord, "start", inf, {"start": 0.0, "end": 1.0, "text": "hello"}),
        (TimedWord, "start", -inf, {"start": 0.0, "end": 1.0, "text": "hello"}),
        (TimedWord, "end", nan, {"start": 0.0, "end": 1.0, "text": "hello"}),
        (TimedWord, "end", inf, {"start": 0.0, "end": 1.0, "text": "hello"}),
        (TimedWord, "end", -inf, {"start": 0.0, "end": 1.0, "text": "hello"}),
        (
            SpeakerWord,
            "start",
            nan,
            {"start": 0.0, "end": 1.0, "text": "hello", "speaker": "SPEAKER_00"},
        ),
        (
            SpeakerWord,
            "start",
            inf,
            {"start": 0.0, "end": 1.0, "text": "hello", "speaker": "SPEAKER_00"},
        ),
        (
            SpeakerWord,
            "start",
            -inf,
            {"start": 0.0, "end": 1.0, "text": "hello", "speaker": "SPEAKER_00"},
        ),
        (
            SpeakerWord,
            "end",
            nan,
            {"start": 0.0, "end": 1.0, "text": "hello", "speaker": "SPEAKER_00"},
        ),
        (
            SpeakerWord,
            "end",
            inf,
            {"start": 0.0, "end": 1.0, "text": "hello", "speaker": "SPEAKER_00"},
        ),
        (
            SpeakerWord,
            "end",
            -inf,
            {"start": 0.0, "end": 1.0, "text": "hello", "speaker": "SPEAKER_00"},
        ),
    ],
)
def test_records_reject_non_finite_timestamps(factory, field, bad_value, kwargs):
    kwargs = dict(kwargs)
    kwargs[field] = bad_value

    with pytest.raises(ValueError):
        factory(**kwargs)


def test_assigns_word_to_turn_with_maximum_overlap():
    turns = [
        SpeakerTurn(0.0, 1.5, "SPEAKER_00"),
        SpeakerTurn(1.5, 3.0, "SPEAKER_01"),
    ]
    words = [TimedWord(1.25, 2.0, "hello")]

    fused = fuse_words(words, turns)

    assert fused == [
        SpeakerWord(1.25, 2.0, "hello", "SPEAKER_01"),
    ]


def test_equal_overlap_prefers_earliest_turn_start():
    turns = [
        SpeakerTurn(0.0, 1.0, "SPEAKER_00"),
        SpeakerTurn(0.5, 1.5, "SPEAKER_01"),
    ]
    words = [TimedWord(0.5, 1.0, "hello")]

    fused = fuse_words(words, turns)

    assert fused[0].speaker == "SPEAKER_00"


def test_equal_overlap_uses_speaker_label_when_turn_starts_match():
    turns = [
        SpeakerTurn(0.0, 1.0, "SPEAKER_02"),
        SpeakerTurn(0.0, 1.0, "SPEAKER_01"),
    ]
    words = [TimedWord(0.25, 0.75, "hello")]

    fused = fuse_words(words, turns)

    assert fused[0].speaker == "SPEAKER_01"


def test_zero_overlap_uses_nearest_turn_to_word_midpoint():
    turns = [
        SpeakerTurn(0.0, 1.0, "SPEAKER_00"),
        SpeakerTurn(3.0, 4.0, "SPEAKER_01"),
    ]
    words = [TimedWord(2.7, 2.8, "hello")]

    fused = fuse_words(words, turns)

    assert fused == [SpeakerWord(2.7, 2.8, "hello", "SPEAKER_01")]


def test_no_turns_assigns_unknown():
    fused = fuse_words([TimedWord(0.1, 0.2, "hello")], [])

    assert fused == [SpeakerWord(0.1, 0.2, "hello", "UNKNOWN")]


def test_groups_adjacent_speakers_into_runs():
    words = [
        SpeakerWord(0.0, 0.5, "hello", "SPEAKER_00"),
        SpeakerWord(0.5, 1.0, "there", "SPEAKER_00"),
        SpeakerWord(1.0, 1.5, "hi", "SPEAKER_01"),
        SpeakerWord(1.5, 2.0, "again", "SPEAKER_00"),
    ]

    groups = group_speaker_words(words)

    assert groups == [
        words[:2],
        words[2:3],
        words[3:],
    ]


def test_render_speaker_transcript_formats_mm_ss_ss():
    words = [
        SpeakerWord(1.2, 3.4, "hello", "SPEAKER_00"),
        SpeakerWord(1.2, 3.4, "there", "SPEAKER_00"),
        SpeakerWord(3.45, 6.1, "friend", "SPEAKER_01"),
    ]

    rendered = render_speaker_transcript(words)

    assert rendered == (
        "[00:01.20–00:03.40] SPEAKER_00: hello there\n"
        "[00:03.45–00:06.10] SPEAKER_01: friend"
    )
