from pathlib import Path
import wave

import cscall.cli as cli
import pytest
from cscall.streaming.metrics import StreamingMetrics
from cscall.streaming.session import StreamingEvent


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


def test_stream_rejects_truncated_pcm_frame_before_model_construction(
    monkeypatch, tmp_path
):
    def boom(*args, **kwargs):
        raise AssertionError("WhisperTranscriber should not be instantiated")

    audio_path = tmp_path / "truncated.wav"
    with wave.open(str(audio_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"\x00\x00\x01\x00")
    audio_path.write_bytes(audio_path.read_bytes()[:-1])

    monkeypatch.setattr(cli, "WhisperTranscriber", boom)

    with pytest.raises(
        ValueError, match=rf"{audio_path} is not a supported PCM WAV"
    ):
        cli.main(["stream", "--audio", str(audio_path)])


def test_benchmark_aggregates_metrics_into_markdown_table(monkeypatch, capsys):
    calls = []

    def fake_run_stream_session(*call_args, **call_kwargs):
        audio_path = call_args[1]
        calls.append(audio_path)
        if audio_path.endswith("a.wav"):
            metrics = StreamingMetrics(
                audio_ms=1000,
                decode_ms=100,
                first_partial_latency_ms=100,
                final_latency_ms=10,
            )
        else:
            metrics = StreamingMetrics(
                audio_ms=1000,
                decode_ms=300,
                first_partial_latency_ms=300,
                final_latency_ms=30,
            )
        return [
            StreamingEvent(
                type="metrics",
                timestamp_ms=1,
                metrics=metrics,
            )
        ]

    monkeypatch.setattr(cli, "_run_stream_session", fake_run_stream_session)

    cli.main(
        [
            "benchmark",
            "--audio",
            str(Path("tests/fixtures/audio/a.wav")),
            str(Path("tests/fixtures/audio/b.wav")),
            "--fake-transcript",
            "hello world",
        ]
    )

    out = capsys.readouterr().out

    assert calls == [
        "tests/fixtures/audio/a.wav",
        "tests/fixtures/audio/b.wav",
    ]
    assert "| Metric | p50 | p99 |" in out
    assert "| RTF | 0.2 | 0.3 |" in out
    assert "| first_partial_ms | 200 | 300 |" in out
    assert "| final_ms | 20 | 30 |" in out
