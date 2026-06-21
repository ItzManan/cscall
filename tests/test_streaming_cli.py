from pathlib import Path

import cscall.cli as cli
import pytest


def test_stream_fake_transcript_runs_without_instantiating_whisper(
    monkeypatch, capsys
):
    def boom(*args, **kwargs):
        raise AssertionError("WhisperTranscriber should not be instantiated")

    monkeypatch.setattr(cli, "WhisperTranscriber", boom)

    audio_path = Path("tests/fixtures/audio/a.wav")
    cli.main(
        [
            "stream",
            "--audio",
            str(audio_path),
            "--fake-transcript",
            "hello world",
            "--agreement",
            "3",
        ]
    )

    out = capsys.readouterr().out

    assert "final" in out
    assert "hello world" in out
    assert "Streaming metrics" in out


def test_stream_forwards_energy_threshold_to_chunk_reader(monkeypatch, capsys):
    seen = {}

    def fake_iter_wav_chunks(audio_path, chunk_ms, energy_threshold=200):
        seen["args"] = (audio_path, chunk_ms, energy_threshold)
        return [], (8000, 1, 1)

    monkeypatch.setattr(cli, "_iter_wav_chunks", fake_iter_wav_chunks)

    cli.main(
        [
            "stream",
            "--audio",
            str(Path("tests/fixtures/audio/a.wav")),
            "--fake-transcript",
            "hello world",
            "--energy-threshold",
            "321",
        ]
    )

    capsys.readouterr()
    assert seen["args"] == ("tests/fixtures/audio/a.wav", 500, 321)


def test_stream_rejects_invalid_wav_before_model_construction(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("WhisperTranscriber should not be instantiated")

    def bad_validate(path):
        raise ValueError(f"{path} is not a supported PCM WAV")

    monkeypatch.setattr(cli, "WhisperTranscriber", boom)
    monkeypatch.setattr(cli, "validate_pcm_wav", bad_validate)

    with pytest.raises(ValueError, match="is not a supported PCM WAV"):
        cli.main(
            [
                "stream",
                "--audio",
                str(Path("tests/fixtures/audio/a.wav")),
                "--agreement",
                "3",
            ]
        )
