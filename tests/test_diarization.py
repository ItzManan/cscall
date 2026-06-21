from __future__ import annotations

import importlib
import types
from decimal import Decimal

import pytest

from cscall.diarization import (
    COMMUNITY_MODEL,
    PyannoteDiarizer,
    SpeakerTurn,
    diarization_error_rate,
    load_rttm,
)


class FakePipeline:
    def __init__(self, annotation):
        self.annotation = annotation
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __call__(self, audio_path: str, **kwargs):
        self.calls.append((audio_path, kwargs))
        return self.annotation


class FakeTurnOutput:
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end


class FakeExclusiveAnnotation:
    def __init__(self, items):
        self.items = items

    def __iter__(self):
        yield from self.items


class FakeItertracksAnnotation:
    def __init__(self, items):
        self.items = items
        self.calls: list[bool] = []

    def itertracks(self, yield_label: bool = False):
        self.calls.append(yield_label)
        yield from self.items


class FakeTupleAnnotation:
    def __init__(self, items):
        self.items = items

    def __iter__(self):
        yield from self.items


class FakeLoadedPipeline:
    def __init__(self, annotation):
        self.annotation = annotation
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __call__(self, audio_path: str, **kwargs):
        self.calls.append((audio_path, kwargs))
        return self.annotation


class FakeSegment:
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end


class FakeAnnotation:
    def __init__(self):
        self.assignments: list[tuple[float, float, str, str]] = []

    def __setitem__(self, key, value):
        segment, track = key
        self.assignments.append((segment.start, segment.end, track, value))


class FakeMetric:
    def __init__(self, result):
        self.result = result
        self.calls: list[tuple[object, object]] = []

    def __call__(self, reference, hypothesis):
        self.calls.append((reference, hypothesis))
        return self.result


def _build_diarizer(pipeline=None, token=None):
    return PyannoteDiarizer(pipeline=pipeline, token=token)


def test_injected_pipeline_bypasses_import_and_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda *_args, **_kwargs: pytest.fail("pyannote import should not run"),
    )
    annotation = types.SimpleNamespace(
        exclusive_speaker_diarization=FakeExclusiveAnnotation(
            [
                (FakeTurnOutput(1.0, 2.0), "SPEAKER_01"),
                (FakeTurnOutput(0.0, 1.0), "SPEAKER_00"),
            ]
        ),
        speaker_diarization=None,
    )
    pipeline = FakePipeline(annotation)
    diarizer = _build_diarizer(pipeline=pipeline, token="secret-token")

    turns = diarizer.diarize("clip.wav")

    assert turns == [
        SpeakerTurn(0.0, 1.0, "SPEAKER_00"),
        SpeakerTurn(1.0, 2.0, "SPEAKER_01"),
    ]
    assert pipeline.calls == [("clip.wav", {"num_speakers": 2})]


def test_inference_exception_escapes_unchanged_for_injected_pipeline(monkeypatch):
    sentinel = RuntimeError("CUDA out of memory")

    def fail_import(*_args, **_kwargs):
        pytest.fail("pyannote import should not run")

    def fail_getenv(*_args, **_kwargs):
        pytest.fail("HF_TOKEN lookup should not run")

    class ExplodingPipeline:
        def __call__(self, audio_path: str, **kwargs):
            raise sentinel

    monkeypatch.setattr(importlib, "import_module", fail_import)
    monkeypatch.setattr("cscall.diarization.os.getenv", fail_getenv)

    diarizer = _build_diarizer(pipeline=ExplodingPipeline())

    with pytest.raises(RuntimeError) as excinfo:
        diarizer.diarize("clip.wav")

    assert excinfo.value is sentinel
    assert str(excinfo.value) == "CUDA out of memory"


def test_regular_annotation_itertracks_is_used_when_exclusive_missing(monkeypatch):
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda *_args, **_kwargs: pytest.fail("pyannote import should not run"),
    )
    regular = FakeItertracksAnnotation(
        [
            (FakeTurnOutput(1.0, 2.0), "track-b", "SPEAKER_02"),
            (FakeTurnOutput(0.0, 1.0), "track-a", "SPEAKER_01"),
        ]
    )
    annotation = types.SimpleNamespace(
        exclusive_speaker_diarization=None,
        speaker_diarization=regular,
    )
    pipeline = FakePipeline(annotation)
    diarizer = _build_diarizer(pipeline=pipeline)

    turns = diarizer.diarize("clip.wav")

    assert regular.calls == [True]
    assert turns == [
        SpeakerTurn(0.0, 1.0, "SPEAKER_01"),
        SpeakerTurn(1.0, 2.0, "SPEAKER_02"),
    ]


def test_tuple_iteration_form_is_supported_and_sorted(monkeypatch):
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda *_args, **_kwargs: pytest.fail("pyannote import should not run"),
    )
    annotation = FakeTupleAnnotation(
        [
            (FakeTurnOutput(1.0, 2.0), "SPEAKER_02"),
            (FakeTurnOutput(1.0, 2.0), "SPEAKER_01"),
        ]
    )
    pipeline = FakePipeline(
        types.SimpleNamespace(
            exclusive_speaker_diarization=None,
            speaker_diarization=annotation,
        )
    )
    diarizer = _build_diarizer(pipeline=pipeline)

    turns = diarizer.diarize("clip.wav")

    assert turns == [
        SpeakerTurn(1.0, 2.0, "SPEAKER_01"),
        SpeakerTurn(1.0, 2.0, "SPEAKER_02"),
    ]


def test_non_string_speaker_labels_are_coerced_before_sorting(monkeypatch):
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda *_args, **_kwargs: pytest.fail("pyannote import should not run"),
    )

    class Label:
        def __init__(self, text: str):
            self.text = text

        def __str__(self) -> str:
            return self.text

    annotation = FakeTupleAnnotation(
        [
            (FakeTurnOutput(1.0, 2.0), Label("SPEAKER_02")),
            (FakeTurnOutput(1.0, 2.0), Label("SPEAKER_01")),
        ]
    )
    pipeline = FakePipeline(
        types.SimpleNamespace(
            exclusive_speaker_diarization=None,
            speaker_diarization=annotation,
        )
    )
    diarizer = _build_diarizer(pipeline=pipeline)

    turns = diarizer.diarize("clip.wav")

    assert turns == [
        SpeakerTurn(1.0, 2.0, "SPEAKER_01"),
        SpeakerTurn(1.0, 2.0, "SPEAKER_02"),
    ]


def test_missing_token_fails_before_loading(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda *_args, **_kwargs: pytest.fail("pyannote import should not run"),
    )

    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        _build_diarizer().diarize("clip.wav")


def test_explicit_token_is_passed_to_model_loader(monkeypatch):
    loaded = {}

    class PipelineModule:
        class Pipeline:
            @staticmethod
            def from_pretrained(model, token=None):
                loaded["model"] = model
                loaded["token"] = token
                return FakeLoadedPipeline(
                    types.SimpleNamespace(
                        exclusive_speaker_diarization=None,
                        speaker_diarization=FakeTupleAnnotation(
                            [(FakeTurnOutput(0.0, 1.0), "SPEAKER_00")]
                        ),
                    )
                )

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(importlib, "import_module", lambda name: PipelineModule)

    turns = _build_diarizer(token="explicit-secret").diarize("clip.wav")

    assert loaded == {"model": COMMUNITY_MODEL, "token": "explicit-secret"}
    assert turns == [SpeakerTurn(0.0, 1.0, "SPEAKER_00")]


def test_env_token_is_used_when_explicit_token_is_missing(monkeypatch):
    loaded = {}

    class PipelineModule:
        class Pipeline:
            @staticmethod
            def from_pretrained(model, token=None):
                loaded["model"] = model
                loaded["token"] = token
                return FakeLoadedPipeline(
                    types.SimpleNamespace(
                        exclusive_speaker_diarization=None,
                        speaker_diarization=FakeTupleAnnotation(
                            [(FakeTurnOutput(0.0, 1.0), "SPEAKER_00")]
                        ),
                    )
                )

    monkeypatch.setenv("HF_TOKEN", "env-secret")
    monkeypatch.setattr(importlib, "import_module", lambda name: PipelineModule)

    turns = _build_diarizer().diarize("clip.wav")

    assert loaded == {"model": COMMUNITY_MODEL, "token": "env-secret"}
    assert turns == [SpeakerTurn(0.0, 1.0, "SPEAKER_00")]


def test_missing_dependency_error_mentions_install_hint(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "env-secret")

    def fail_import(name):
        raise ModuleNotFoundError("No module named 'pyannote.audio'")

    monkeypatch.setattr(importlib, "import_module", fail_import)

    with pytest.raises(RuntimeError) as excinfo:
        _build_diarizer().diarize("clip.wav")

    message = str(excinfo.value)
    assert 'pip install -e ".[diarization]"' in message
    assert excinfo.value.__cause__ is not None


def test_model_access_errors_include_community_agreement_guidance(monkeypatch):
    token = "secret-token"

    class PipelineModule:
        class Pipeline:
            @staticmethod
            def from_pretrained(model, token=None):
                raise PermissionError("model is gated")

    monkeypatch.setattr(importlib, "import_module", lambda name: PipelineModule)

    with pytest.raises(RuntimeError) as excinfo:
        _build_diarizer(token=token).diarize("clip.wav")

    message = str(excinfo.value)
    assert COMMUNITY_MODEL in message
    assert "accept" in message.lower()
    assert token not in message
    assert isinstance(excinfo.value.__cause__, PermissionError)


def test_load_rttm_parses_speaker_rows_and_sorts_output(tmp_path):
    path = tmp_path / "sample.rttm"
    path.write_text(
        "\n"
        "# ignored\n"
        "SPKR-INFO file-a 1 0.0 0.0 <NA> <NA> speaker-a <NA>\n"
        "SPEAKER file-b 1 2.0 1.0 <NA> <NA> SPEAKER_01 <NA>\n"
        "  # also ignored\n"
        "SPEAKER file-a 1 0.5 0.25 <NA> <NA> SPEAKER_00 <NA> <NA>\n",
        encoding="utf-8",
    )

    turns = load_rttm(path)

    assert turns == [
        SpeakerTurn(0.5, 0.75, "SPEAKER_00"),
        SpeakerTurn(2.0, 3.0, "SPEAKER_01"),
    ]


@pytest.mark.parametrize(
    "line, reason",
    [
        (
            "SPEAKER file-a 1 0.0 1.0 <NA> <NA> SPEAKER_00",
            "expected at least 9 fields",
        ),
        (
            "SPEAKER file-a 1 abc 1.0 <NA> <NA> SPEAKER_00 <NA>",
            "start and duration must be numeric",
        ),
        (
            "SPEAKER file-a 1 inf 1.0 <NA> <NA> SPEAKER_00 <NA>",
            "start and duration must be finite",
        ),
        (
            "SPEAKER file-a 1 -0.1 1.0 <NA> <NA> SPEAKER_00 <NA>",
            "start must be >= 0",
        ),
        (
            "SPEAKER file-a 1 0.0 0.0 <NA> <NA> SPEAKER_00 <NA>",
            "duration must be > 0",
        ),
    ],
)
def test_load_rttm_rejects_invalid_rows_with_path_and_line(
    tmp_path, line, reason
):
    path = tmp_path / "broken.rttm"
    path.write_text(f"\n{line}\n", encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        load_rttm(path)

    message = str(excinfo.value)
    assert f"{path}:2" in message
    assert reason in message


def test_diarization_error_rate_uses_injected_factories_and_metric(monkeypatch):
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda *_args, **_kwargs: pytest.fail("pyannote import should not run"),
    )
    metric = FakeMetric(Decimal("1.25"))

    reference = [
        SpeakerTurn(0.0, 1.0, "SPEAKER_00"),
        SpeakerTurn(0.5, 1.5, "SPEAKER_00"),
    ]
    hypothesis = [
        SpeakerTurn(0.0, 1.0, "SPEAKER_01"),
        SpeakerTurn(1.0, 2.0, "SPEAKER_02"),
    ]

    score = diarization_error_rate(
        reference,
        hypothesis,
        annotation_factory=FakeAnnotation,
        segment_factory=FakeSegment,
        metric=metric,
    )

    assert isinstance(score, float)
    assert score == pytest.approx(1.25)
    assert len(metric.calls) == 1
    reference_annotation, hypothesis_annotation = metric.calls[0]
    assert reference_annotation.assignments == [
        (0.0, 1.0, "SPEAKER_00:0", "SPEAKER_00"),
        (0.5, 1.5, "SPEAKER_00:1", "SPEAKER_00"),
    ]
    assert hypothesis_annotation.assignments == [
        (0.0, 1.0, "SPEAKER_01:0", "SPEAKER_01"),
        (1.0, 2.0, "SPEAKER_02:1", "SPEAKER_02"),
    ]


def test_diarization_error_rate_missing_dependency_mentions_install_hint(monkeypatch):
    def fail_import(name):
        raise ModuleNotFoundError(f"No module named {name!r}")

    monkeypatch.setattr(importlib, "import_module", fail_import)

    with pytest.raises(RuntimeError) as excinfo:
        diarization_error_rate([], [])

    message = str(excinfo.value)
    assert 'pip install -e ".[diarization]"' in message
    assert excinfo.value.__cause__ is not None
