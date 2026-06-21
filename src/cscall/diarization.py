from __future__ import annotations

import importlib
import os
from math import isfinite
from collections.abc import Iterable

from cscall.fusion import SpeakerTurn

COMMUNITY_MODEL = "pyannote/speaker-diarization-community-1"
_OPTIONAL_DIARIZATION_HINT = 'pip install -e ".[diarization]".'


def load_rttm(path: str) -> list[SpeakerTurn]:
    turns: list[SpeakerTurn] = []
    path_str = os.fspath(path)

    with open(path_str, encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            fields = stripped.split()
            if fields[0] != "SPEAKER":
                continue
            if len(fields) < 9:
                raise ValueError(
                    f"{path_str}:{line_number}: expected at least 9 fields"
                )

            start = _parse_rttm_float(
                fields[3], path_str, line_number, "start and duration must be numeric"
            )
            duration = _parse_rttm_float(
                fields[4], path_str, line_number, "start and duration must be numeric"
            )

            if not isfinite(start) or not isfinite(duration):
                raise ValueError(
                    f"{path_str}:{line_number}: start and duration must be finite"
                )
            if start < 0:
                raise ValueError(f"{path_str}:{line_number}: start must be >= 0")
            if duration <= 0:
                raise ValueError(f"{path_str}:{line_number}: duration must be > 0")

            speaker = fields[7].strip()
            if not speaker:
                raise ValueError(
                    f"{path_str}:{line_number}: speaker field must be nonempty"
                )

            turns.append(SpeakerTurn(start, start + duration, speaker))

    turns.sort(key=lambda turn: (turn.start, turn.end, turn.speaker))
    return turns


def diarization_error_rate(
    reference: list[SpeakerTurn],
    hypothesis: list[SpeakerTurn],
    *,
    annotation_factory=None,
    segment_factory=None,
    metric=None,
) -> float:
    annotation_factory, segment_factory, metric = _resolve_diarization_dependencies(
        annotation_factory,
        segment_factory,
        metric,
    )
    reference_annotation = _turns_to_annotation(
        reference, annotation_factory, segment_factory
    )
    hypothesis_annotation = _turns_to_annotation(
        hypothesis, annotation_factory, segment_factory
    )
    return float(metric(reference_annotation, hypothesis_annotation))


class PyannoteDiarizer:
    def __init__(self, pipeline=None, token: str | None = None):
        self._pipeline = pipeline
        self._token = token

    def diarize(self, audio_path: str) -> list[SpeakerTurn]:
        pipeline = self._pipeline if self._pipeline is not None else self._load_pipeline()
        annotation = pipeline(audio_path, num_speakers=2)
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
        except Exception as exc:
            raise RuntimeError(
                f"Community-1 diarization failed for {COMMUNITY_MODEL}. "
                "Accept the model conditions on Hugging Face and try again."
            ) from exc


def _resolve_diarization_dependencies(
    annotation_factory,
    segment_factory,
    metric,
):
    if annotation_factory is not None and segment_factory is not None and metric is not None:
        return annotation_factory, segment_factory, metric

    try:
        core = importlib.import_module("pyannote.core")
        metrics = importlib.import_module("pyannote.metrics.diarization")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"pyannote is required; install it with {_OPTIONAL_DIARIZATION_HINT}"
        ) from exc

    default_annotation_factory = getattr(core, "Annotation")
    default_segment_factory = getattr(core, "Segment")
    default_metric = metrics.DiarizationErrorRate(collar=0.0, skip_overlap=False)

    return (
        annotation_factory or default_annotation_factory,
        segment_factory or default_segment_factory,
        metric or default_metric,
    )


def _turns_to_annotation(turns, annotation_factory, segment_factory):
    annotation = annotation_factory()
    for index, turn in enumerate(turns):
        segment = segment_factory(turn.start, turn.end)
        annotation[segment, f"{turn.speaker}:{index}"] = turn.speaker
    return annotation


def _annotation_to_turns(annotation) -> list[SpeakerTurn]:
    source = getattr(annotation, "exclusive_speaker_diarization", None)
    if source is None:
        source = getattr(annotation, "speaker_diarization", annotation)

    turns: list[SpeakerTurn] = []
    for item in _iter_annotation_items(source):
        turn, speaker = _coerce_turn_item(item)
        turns.append(SpeakerTurn(turn.start, turn.end, str(speaker)))
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


def _parse_rttm_float(field: str, path: str, line_number: int, reason: str) -> float:
    try:
        return float(field)
    except ValueError as exc:
        raise ValueError(f"{path}:{line_number}: {reason}") from exc
