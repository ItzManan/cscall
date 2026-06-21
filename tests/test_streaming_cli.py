from pathlib import Path

import cscall.cli as cli


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
