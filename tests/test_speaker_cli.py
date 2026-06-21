from __future__ import annotations

from pathlib import Path

import pytest

import cscall.cli as cli
from cscall.fusion import SpeakerTurn, SpeakerWord, TimedWord


class RecordingDiarizer:
    def __init__(self, calls, turns=None, **kwargs):
        calls.append(kwargs)
        self._turns = turns or []
        self.calls: list[str] = []

    def diarize(self, audio_path: str):
        self.calls.append(audio_path)
        return self._turns


class RecordingWhisperTranscriber:
    def __init__(self, calls, words=None, **kwargs):
        calls.append(kwargs)
        self._words = words or []
        self.calls: list[str] = []

    def transcribe_words(self, audio_path: str):
        self.calls.append(audio_path)
        return self._words


def test_diarize_prints_exact_turns_and_der_when_reference_rttm_is_supplied(
    monkeypatch, capsys, tmp_path
):
    validate_calls = []
    diarizer_calls = []
    rttm_calls = []
    der_calls = []
    diarizer_instances = []

    def fake_validate_pcm_wav(path):
        validate_calls.append(path)
        return cli.WavInfo(sample_rate=16000, channels=1, sample_width=2)

    def fake_diarization_error_rate(reference, hypothesis):
        der_calls.append((reference, hypothesis))
        return 0.1234

    reference_turns = [SpeakerTurn(0.0, 1.0, "SPEAKER_00")]

    class FakeDiarizer(RecordingDiarizer):
        def __init__(self, **kwargs):
            diarizer_instances.append(self)
            super().__init__(
                diarizer_calls,
                turns=[
                    SpeakerTurn(0.0, 1.0, "SPEAKER_00"),
                    SpeakerTurn(1.0, 2.0, "SPEAKER_01"),
                ],
                **kwargs,
            )

    monkeypatch.setattr(cli, "validate_pcm_wav", fake_validate_pcm_wav)
    monkeypatch.setattr(cli, "PyannoteDiarizer", FakeDiarizer)
    monkeypatch.setattr(cli, "load_rttm", lambda path: rttm_calls.append(path) or reference_turns)
    monkeypatch.setattr(cli, "diarization_error_rate", fake_diarization_error_rate)

    audio_path = tmp_path / "call.wav"
    audio_path.write_bytes(b"wav")
    rttm_path = tmp_path / "ref.rttm"
    rttm_path.write_text("rttm", encoding="utf-8")

    cli.main([
        "diarize",
        "--audio",
        str(audio_path),
        "--reference-rttm",
        str(rttm_path),
    ])

    out = capsys.readouterr().out.splitlines()

    assert validate_calls == [str(audio_path)]
    assert diarizer_calls == [{}]
    assert diarizer_instances[0].calls == [str(audio_path)]
    assert out == [
        "0.000\t1.000\tSPEAKER_00",
        "1.000\t2.000\tSPEAKER_01",
        "DER: 12.34%",
    ]
    assert rttm_calls == [str(rttm_path)]
    assert der_calls == [(
        reference_turns,
        [
            SpeakerTurn(0.0, 1.0, "SPEAKER_00"),
            SpeakerTurn(1.0, 2.0, "SPEAKER_01"),
        ],
    )]


def test_diarize_loads_reference_rttm_before_constructing_diarizer(
    monkeypatch, tmp_path
):
    validate_calls = []
    rttm_calls = []

    def fake_validate_pcm_wav(path):
        validate_calls.append(path)
        return cli.WavInfo(sample_rate=16000, channels=1, sample_width=2)

    def fake_load_rttm(path):
        rttm_calls.append(path)
        raise ValueError("bad rttm")

    def boom(*args, **kwargs):
        raise AssertionError("PyannoteDiarizer should not be instantiated")

    monkeypatch.setattr(cli, "validate_pcm_wav", fake_validate_pcm_wav)
    monkeypatch.setattr(cli, "load_rttm", fake_load_rttm)
    monkeypatch.setattr(cli, "PyannoteDiarizer", boom)

    audio_path = tmp_path / "call.wav"
    audio_path.write_bytes(b"wav")
    rttm_path = tmp_path / "ref.rttm"
    rttm_path.write_text("rttm", encoding="utf-8")

    with pytest.raises(ValueError, match="bad rttm"):
        cli.main(
            [
                "diarize",
                "--audio",
                str(audio_path),
                "--reference-rttm",
                str(rttm_path),
            ]
        )

    assert validate_calls == [str(audio_path)]
    assert rttm_calls == [str(rttm_path)]


def test_diarize_validates_pcm_wav_before_constructing_diarizer(monkeypatch, tmp_path):
    def boom(*args, **kwargs):
        raise AssertionError("PyannoteDiarizer should not be instantiated")

    def bad_validate(path):
        raise ValueError(f"{path} is not a supported PCM WAV")

    monkeypatch.setattr(cli, "validate_pcm_wav", bad_validate)
    monkeypatch.setattr(cli, "PyannoteDiarizer", boom)

    audio_path = tmp_path / "call.wav"
    audio_path.write_bytes(b"wav")

    with pytest.raises(ValueError, match="is not a supported PCM WAV"):
        cli.main(["diarize", "--audio", str(audio_path)])


def test_diarize_empty_turns_prints_nothing(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(
        cli,
        "validate_pcm_wav",
        lambda path: cli.WavInfo(sample_rate=16000, channels=1, sample_width=2),
    )

    class FakeDiarizer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.calls: list[str] = []

        def diarize(self, audio_path: str):
            self.calls.append(audio_path)
            return []

    monkeypatch.setattr(cli, "PyannoteDiarizer", FakeDiarizer)

    audio_path = tmp_path / "call.wav"
    audio_path.write_bytes(b"wav")

    cli.main(["diarize", "--audio", str(audio_path)])

    assert capsys.readouterr().out == ""


def test_transcribe_speakers_forwards_language_and_renders_transcript(
    monkeypatch, capsys, tmp_path
):
    validate_calls = []
    whisper_calls = []
    diarizer_calls = []
    fuse_calls = []
    render_calls = []
    whisper_instances = []
    diarizer_instances = []

    def fake_validate_pcm_wav(path):
        validate_calls.append(path)
        return cli.WavInfo(sample_rate=16000, channels=1, sample_width=2)

    class FakeWhisperTranscriber(RecordingWhisperTranscriber):
        def __init__(self, **kwargs):
            whisper_instances.append(self)
            super().__init__(
                whisper_calls,
                words=[
                    TimedWord(0.0, 0.5, "hello"),
                    TimedWord(0.5, 1.0, "world"),
                ],
                **kwargs,
            )

    class FakeDiarizer(RecordingDiarizer):
        def __init__(self, **kwargs):
            diarizer_instances.append(self)
            super().__init__(
                diarizer_calls,
                turns=[SpeakerTurn(0.0, 1.0, "SPEAKER_00")],
                **kwargs,
            )

    def fake_fuse_words(words, turns):
        fuse_calls.append((words, turns))
        return [
            SpeakerWord(0.0, 0.5, "hello", "SPEAKER_00"),
            SpeakerWord(0.5, 1.0, "world", "SPEAKER_00"),
        ]

    def fake_render_speaker_transcript(words):
        render_calls.append(words)
        return "speaker transcript"

    monkeypatch.setattr(cli, "validate_pcm_wav", fake_validate_pcm_wav)
    monkeypatch.setattr(cli, "WhisperTranscriber", FakeWhisperTranscriber)
    monkeypatch.setattr(cli, "PyannoteDiarizer", FakeDiarizer)
    monkeypatch.setattr(cli, "fuse_words", fake_fuse_words)
    monkeypatch.setattr(cli, "render_speaker_transcript", fake_render_speaker_transcript)

    audio_path = tmp_path / "call.wav"
    audio_path.write_bytes(b"wav")

    cli.main(
        [
            "transcribe-speakers",
            "--audio",
            str(audio_path),
            "--model",
            "medium",
            "--device",
            "cuda",
            "--compute-type",
            "float16",
            "--language",
            "hi",
        ]
    )

    assert capsys.readouterr().out == "speaker transcript\n"
    assert validate_calls == [str(audio_path)]
    assert whisper_calls == [
        {
            "model_size": "medium",
            "device": "cuda",
            "compute_type": "float16",
            "language": "hi",
        }
    ]
    assert diarizer_calls == [{}]
    assert diarizer_instances[0].calls == [str(audio_path)]
    assert whisper_instances[0].calls == [str(audio_path)]
    assert fuse_calls == [
        (
            [
                TimedWord(0.0, 0.5, "hello"),
                TimedWord(0.5, 1.0, "world"),
            ],
            [SpeakerTurn(0.0, 1.0, "SPEAKER_00")],
        )
    ]


def test_transcribe_speakers_runs_diarizer_before_constructing_whisper(
    monkeypatch, tmp_path
):
    validate_calls = []
    diarizer_instances = []

    def fake_validate_pcm_wav(path):
        validate_calls.append(path)
        return cli.WavInfo(sample_rate=16000, channels=1, sample_width=2)

    class SentinelDiarizer:
        def __init__(self, **kwargs):
            diarizer_instances.append(self)

        def diarize(self, audio_path: str):
            self.audio_path = audio_path
            raise RuntimeError("missing token")

    def boom(*args, **kwargs):
        raise AssertionError("WhisperTranscriber should not be instantiated")

    monkeypatch.setattr(cli, "validate_pcm_wav", fake_validate_pcm_wav)
    monkeypatch.setattr(cli, "PyannoteDiarizer", SentinelDiarizer)
    monkeypatch.setattr(cli, "WhisperTranscriber", boom)

    audio_path = tmp_path / "call.wav"
    audio_path.write_bytes(b"wav")

    with pytest.raises(RuntimeError, match="missing token"):
        cli.main(["transcribe-speakers", "--audio", str(audio_path)])

    assert validate_calls == [str(audio_path)]
    assert diarizer_instances[0].audio_path == str(audio_path)


def test_transcribe_speakers_validates_pcm_wav_before_constructing_models(
    monkeypatch, tmp_path
):
    def boom(*args, **kwargs):
        raise AssertionError("model construction should not happen")

    def bad_validate(path):
        raise ValueError(f"{path} is not a supported PCM WAV")

    monkeypatch.setattr(cli, "validate_pcm_wav", bad_validate)
    monkeypatch.setattr(cli, "WhisperTranscriber", boom)
    monkeypatch.setattr(cli, "PyannoteDiarizer", boom)

    audio_path = tmp_path / "call.wav"
    audio_path.write_bytes(b"wav")

    with pytest.raises(ValueError, match="is not a supported PCM WAV"):
        cli.main(["transcribe-speakers", "--audio", str(audio_path)])


def test_transcribe_speakers_empty_results_print_nothing(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(
        cli,
        "validate_pcm_wav",
        lambda path: cli.WavInfo(sample_rate=16000, channels=1, sample_width=2),
    )

    class FakeWhisperTranscriber:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.calls: list[str] = []

        def transcribe_words(self, audio_path: str):
            self.calls.append(audio_path)
            return []

    class FakeDiarizer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.calls: list[str] = []

        def diarize(self, audio_path: str):
            self.calls.append(audio_path)
            return []

    monkeypatch.setattr(cli, "WhisperTranscriber", FakeWhisperTranscriber)
    monkeypatch.setattr(cli, "PyannoteDiarizer", FakeDiarizer)
    monkeypatch.setattr(cli, "fuse_words", lambda words, turns: [])
    monkeypatch.setattr(cli, "render_speaker_transcript", lambda words: "")

    audio_path = tmp_path / "call.wav"
    audio_path.write_bytes(b"wav")

    cli.main(["transcribe-speakers", "--audio", str(audio_path)])

    assert capsys.readouterr().out == ""
