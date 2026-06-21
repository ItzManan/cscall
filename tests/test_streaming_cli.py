from pathlib import Path
import wave

import cscall.cli as cli
import pytest
from cscall.streaming.metrics import StreamingMetrics
from cscall.streaming.session import StreamingEvent


class RecordingTranscriber:
    def __init__(self, calls, **kwargs):
        calls.append(kwargs)

    def transcribe(self, audio_path: str) -> str:
        return f"transcribed:{audio_path}"


def test_baseline_forwards_language_to_whisper_transcriber(monkeypatch, capsys):
    calls = []

    def fake_whisper_transcriber(*args, **kwargs):
        return RecordingTranscriber(calls, **kwargs)

    monkeypatch.setattr(cli, "WhisperTranscriber", fake_whisper_transcriber)
    monkeypatch.setattr(cli, "render_markdown", lambda report: "baseline")

    cli.main(
        [
            "baseline",
            "--manifest",
            "tests/fixtures/mini_manifest.jsonl",
            "--language",
            "hi",
        ]
    )

    capsys.readouterr()
    assert calls == [
        {"model_size": "small", "device": "cpu", "compute_type": "int8", "language": "hi"}
    ]


def test_compare_forwards_language_to_both_whisper_transcribers(monkeypatch, capsys):
    calls = []

    def fake_whisper_transcriber(*args, **kwargs):
        return RecordingTranscriber(calls, **kwargs)

    monkeypatch.setattr(cli, "WhisperTranscriber", fake_whisper_transcriber)
    monkeypatch.setattr(cli, "render_comparison_markdown", lambda result: "compare")

    cli.main(
        [
            "compare",
            "--manifest",
            "tests/fixtures/mini_manifest.jsonl",
            "--finetuned-ct2",
            "out/ct2",
            "--language",
            "hi",
        ]
    )

    capsys.readouterr()
    assert calls == [
        {"model_size": "small", "device": "cpu", "compute_type": "int8", "language": "hi"},
        {"model_size": "out/ct2", "device": "cpu", "compute_type": "int8", "language": "hi"},
    ]


def test_stream_forwards_language_to_whisper_transcriber(monkeypatch, capsys):
    calls = []

    def fake_whisper_transcriber(*args, **kwargs):
        return RecordingTranscriber(calls, **kwargs)

    def fake_validate_pcm_wav(path):
        return cli.WavInfo(sample_rate=8000, channels=1, sample_width=2)

    def fake_iter_wav_chunks(audio_path, chunk_ms, energy_threshold=200):
        return [], (8000, 1, 2)

    monkeypatch.setattr(cli, "WhisperTranscriber", fake_whisper_transcriber)
    monkeypatch.setattr(cli, "validate_pcm_wav", fake_validate_pcm_wav)
    monkeypatch.setattr(cli, "_iter_wav_chunks", fake_iter_wav_chunks)

    cli.main(
        [
            "stream",
            "--audio",
            str(Path("tests/fixtures/audio/a.wav")),
            "--language",
            "hi",
        ]
    )

    capsys.readouterr()
    assert calls == [
        {"model_size": "small", "device": "cpu", "compute_type": "int8", "language": "hi"}
    ]


def test_benchmark_forwards_language_to_shared_whisper_transcriber(
    monkeypatch, capsys
):
    calls = []
    seen_transcribers = []

    def fake_whisper_transcriber(*args, **kwargs):
        transcriber = RecordingTranscriber(calls, **kwargs)
        seen_transcribers.append(transcriber)
        return transcriber

    def fake_validate_pcm_wav(path):
        return cli.WavInfo(sample_rate=8000, channels=1, sample_width=2)

    def fake_run_stream_session(*call_args, **call_kwargs):
        seen_transcribers.append(call_kwargs["transcriber"])
        return []

    monkeypatch.setattr(cli, "WhisperTranscriber", fake_whisper_transcriber)
    monkeypatch.setattr(cli, "validate_pcm_wav", fake_validate_pcm_wav)
    monkeypatch.setattr(cli, "_run_stream_session", fake_run_stream_session)

    cli.main(
        [
            "benchmark",
            "--audio",
            str(Path("tests/fixtures/audio/a.wav")),
            str(Path("tests/fixtures/audio/b.wav")),
            "--language",
            "hi",
        ]
    )

    capsys.readouterr()
    assert calls == [
        {"model_size": "small", "device": "cpu", "compute_type": "int8", "language": "hi"}
    ]
    assert seen_transcribers[0] is seen_transcribers[1] is seen_transcribers[2]


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


def test_benchmark_rejects_invalid_audio_before_model_construction(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("WhisperTranscriber should not be instantiated")

    def bad_validate(path):
        raise ValueError(f"{path} is not a supported PCM WAV")

    monkeypatch.setattr(cli, "WhisperTranscriber", boom)
    monkeypatch.setattr(cli, "validate_pcm_wav", bad_validate)

    with pytest.raises(ValueError, match="is not a supported PCM WAV"):
        cli.main(
            [
                "benchmark",
                "--audio",
                str(Path("tests/fixtures/audio/a.wav")),
            ]
        )


def test_benchmark_uses_manifest_audio_paths_in_order(monkeypatch, tmp_path, capsys):
    manifest = tmp_path / "mini_manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                '{"id": "u1", "audio_path": "tests/fixtures/audio/b.wav", "text": "one"}',
                '{"id": "u2", "audio_path": "tests/fixtures/audio/a.wav", "text": "two"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    seen = []

    def fake_validate_pcm_wav(path):
        return cli.WavInfo(sample_rate=8000, channels=1, sample_width=2)

    def fake_run_stream_session(*call_args, **call_kwargs):
        seen.append(call_args[1])
        return [
            StreamingEvent(
                type="metrics",
                timestamp_ms=1,
                metrics=StreamingMetrics(audio_ms=1000, decode_ms=100),
            )
        ]

    monkeypatch.setattr(cli, "validate_pcm_wav", fake_validate_pcm_wav)
    monkeypatch.setattr(cli, "_run_stream_session", fake_run_stream_session)

    cli.main(
        [
            "benchmark",
            "--manifest",
            str(manifest),
            "--fake-transcript",
            "hello world",
        ]
    )

    capsys.readouterr()

    assert seen == [
        "tests/fixtures/audio/b.wav",
        "tests/fixtures/audio/a.wav",
    ]


def test_benchmark_empty_manifest_does_not_instantiate_model(monkeypatch, tmp_path, capsys):
    manifest = tmp_path / "empty_manifest.jsonl"
    manifest.write_text("", encoding="utf-8")

    def boom(*args, **kwargs):
        raise AssertionError("WhisperTranscriber should not be instantiated")

    monkeypatch.setattr(cli, "WhisperTranscriber", boom)

    cli.main(
        [
            "benchmark",
            "--manifest",
            str(manifest),
        ]
    )

    out = capsys.readouterr().out

    assert "| Metric | p50 | p99 |" in out
    assert "| RTF | n/a | n/a |" in out
    assert "| first_partial_ms | n/a | n/a |" in out
    assert "| final_ms | n/a | n/a |" in out
