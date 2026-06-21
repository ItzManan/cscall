from __future__ import annotations

import importlib
import os
from collections.abc import Iterable

from cscall.fusion import SpeakerTurn

COMMUNITY_MODEL = "pyannote/speaker-diarization-community-1"


class PyannoteDiarizer:
    def __init__(self, pipeline=None, token: str | None = None):
        self._pipeline = pipeline
        self._token = token

    def diarize(self, audio_path: str) -> list[SpeakerTurn]:
        pipeline = self._pipeline if self._pipeline is not None else self._load_pipeline()
        try:
            annotation = pipeline(audio_path, num_speakers=2)
        except Exception as exc:  # pragma: no cover - exercised by tests
            raise RuntimeError(
                f"Community-1 diarization failed for {COMMUNITY_MODEL}. "
                "Accept the model conditions on Hugging Face and try again."
            ) from exc
        turns = _annotation_to_turns(annotation)
        turns.sort(key=lambda turn: (turn.start, turn.end, turn.speaker))
        return turns

    def _load_pipeline(self):
        token = self._token if self._token is not None else os.getenv("HF_TOKEN")
        if not token:
            raise RuntimeError(
                "HF_TOKEN is required to load Community-1 diarization from "
                f"{COMMUNITY_MODEL}."
            )
        try:
            module = importlib.import_module("pyannote.audio")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                'pyannote.audio is required; install it with '
                'pip install -e ".[diarization]".'
            ) from exc

        try:
            return module.Pipeline.from_pretrained(COMMUNITY_MODEL, token=token)
        except Exception as exc:  # pragma: no cover - exercised by tests
            raise RuntimeError(
                f"Community-1 diarization failed for {COMMUNITY_MODEL}. "
                "Accept the model conditions on Hugging Face and try again."
            ) from exc


def _annotation_to_turns(annotation) -> list[SpeakerTurn]:
    source = getattr(annotation, "exclusive_speaker_diarization", None)
    if source is None:
        source = getattr(annotation, "speaker_diarization", annotation)

    turns: list[SpeakerTurn] = []
    for item in _iter_annotation_items(source):
        turn, speaker = _coerce_turn_item(item)
        turns.append(SpeakerTurn(turn.start, turn.end, speaker))
    return turns


def _iter_annotation_items(annotation) -> Iterable[object]:
    itertracks = getattr(annotation, "itertracks", None)
    if callable(itertracks):
        yield from itertracks(yield_label=True)
        return
    yield from annotation


def _coerce_turn_item(item) -> tuple[object, str]:
    if isinstance(item, tuple) or isinstance(item, list):
        if len(item) == 2:
            turn, speaker = item
            return turn, speaker
        if len(item) >= 3:
            turn, _track, speaker = item[:3]
            return turn, speaker
    raise TypeError("Unsupported diarization item")
