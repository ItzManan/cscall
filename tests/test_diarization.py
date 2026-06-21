from __future__ import annotations

import importlib
import types

import pytest

from cscall.diarization import COMMUNITY_MODEL, PyannoteDiarizer, SpeakerTurn


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
