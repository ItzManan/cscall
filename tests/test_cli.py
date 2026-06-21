import pytest

import cscall.cli as cli
from cscall.cli import build_parser


def test_baseline_subcommand_parses():
    parser = build_parser()
    args = parser.parse_args(
        [
            "baseline",
            "--manifest",
            "m.jsonl",
            "--model",
            "small",
            "--group-by",
            "accent",
            "--language",
            "hi",
        ]
    )
    assert args.command == "baseline"
    assert args.manifest == "m.jsonl"
    assert args.model == "small"
    assert args.group_by == "accent"
    assert args.language == "hi"


def test_model_defaults_to_small():
    parser = build_parser()
    args = parser.parse_args(["baseline", "--manifest", "m.jsonl"])
    assert args.model == "small"
    assert args.group_by is None
    assert args.language is None


def test_device_defaults_to_cpu():
    parser = build_parser()
    base = parser.parse_args(["baseline", "--manifest", "m.jsonl"])
    cmp = parser.parse_args(
        ["compare", "--manifest", "m.jsonl", "--finetuned-ct2", "out/ct2"]
    )
    assert base.device == "cpu"
    assert cmp.device == "cpu"
    assert base.language is None
    assert cmp.language is None


def test_device_can_be_set_to_cuda():
    parser = build_parser()
    args = parser.parse_args(
        ["compare", "--manifest", "m.jsonl", "--finetuned-ct2", "out/ct2",
         "--device", "cuda"]
    )
    assert args.device == "cuda"


def test_compare_subcommand_parses():
    parser = build_parser()
    args = parser.parse_args(
        [
            "compare",
            "--manifest", "m.jsonl",
            "--baseline-model", "small",
            "--finetuned-ct2", "out/ct2",
            "--group-by", "accent",
            "--language",
            "hi",
        ]
    )
    assert args.command == "compare"
    assert args.manifest == "m.jsonl"
    assert args.baseline_model == "small"
    assert args.finetuned_ct2 == "out/ct2"
    assert args.group_by == "accent"
    assert args.language == "hi"


def test_stream_subcommand_parses_with_defaults():
    parser = build_parser()
    args = parser.parse_args(["stream", "--audio", "tests/fixtures/audio/a.wav"])

    assert args.command == "stream"
    assert args.audio == "tests/fixtures/audio/a.wav"
    assert args.model == "small"
    assert args.chunk_ms == 500
    assert args.agreement == 2
    assert args.compute_type == "int8"
    assert args.device == "cpu"
    assert args.energy_threshold == 200
    assert args.fake_transcript is None
    assert args.language is None


def test_stream_subcommand_parses_custom_options():
    parser = build_parser()
    args = parser.parse_args(
        [
            "stream",
            "--audio",
            "tests/fixtures/audio/a.wav",
            "--model",
            "tiny",
            "--chunk-ms",
            "250",
            "--agreement",
            "3",
            "--compute-type",
            "float16",
            "--device",
            "cuda",
            "--language",
            "hi",
            "--fake-transcript",
            "hello",
        ]
    )

    assert args.command == "stream"
    assert args.model == "tiny"
    assert args.chunk_ms == 250
    assert args.agreement == 3
    assert args.compute_type == "float16"
    assert args.device == "cuda"
    assert args.energy_threshold == 200
    assert args.language == "hi"
    assert args.fake_transcript == "hello"


def test_stream_subcommand_parses_custom_energy_threshold():
    parser = build_parser()
    args = parser.parse_args(
        [
            "stream",
            "--audio",
            "tests/fixtures/audio/a.wav",
            "--energy-threshold",
            "37",
        ]
    )

    assert args.command == "stream"
    assert args.energy_threshold == 37


def test_benchmark_subcommand_parses_multiple_audio_paths_and_defaults():
    parser = build_parser()
    args = parser.parse_args(
        [
            "benchmark",
            "--audio",
            "tests/fixtures/audio/a.wav",
            "tests/fixtures/audio/b.wav",
            "--fake-transcript",
            "hello",
            "--language",
            "hi",
        ]
    )

    assert args.command == "benchmark"
    assert args.audio == [
        "tests/fixtures/audio/a.wav",
        "tests/fixtures/audio/b.wav",
    ]
    assert args.model == "small"
    assert args.chunk_ms == 500
    assert args.agreement == 2
    assert args.compute_type == "int8"
    assert args.device == "cpu"
    assert args.energy_threshold == 200
    assert args.language == "hi"
    assert args.fake_transcript == "hello"


def test_benchmark_subcommand_parses_manifest_path():
    parser = build_parser()
    args = parser.parse_args(
        [
            "benchmark",
            "--manifest",
            "tests/fixtures/mini_manifest.jsonl",
            "--fake-transcript",
            "hello",
        ]
    )

    assert args.command == "benchmark"
    assert args.manifest == "tests/fixtures/mini_manifest.jsonl"
    assert args.audio is None
    assert args.language is None


def test_ui_subcommand_parses_defaults():
    parser = build_parser()
    args = parser.parse_args(["ui"])

    assert args.command == "ui"
    assert args.host == "127.0.0.1"
    assert args.port == 8000
    assert args.model == "small"
    assert args.device == "cpu"
    assert args.compute_type == "int8"
    assert args.language is None


def test_ui_subcommand_parses_custom_model_and_runtime_options():
    parser = build_parser()
    args = parser.parse_args(
        [
            "ui",
            "--host",
            "0.0.0.0",
            "--port",
            "8123",
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

    assert args.command == "ui"
    assert args.host == "0.0.0.0"
    assert args.port == 8123
    assert args.model == "medium"
    assert args.device == "cuda"
    assert args.compute_type == "float16"
    assert args.language == "hi"


@pytest.mark.parametrize("port", ["0", "65536"])
def test_ui_subcommand_rejects_invalid_port(port: str):
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["ui", "--port", port])


def test_main_ui_forwards_run_server_arguments(monkeypatch):
    captured = {}

    def fake_run_server(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli, "run_server", fake_run_server)

    cli.main(
        [
            "ui",
            "--host",
            "0.0.0.0",
            "--port",
            "8123",
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

    assert captured == {
        "host": "0.0.0.0",
        "port": 8123,
        "model": "medium",
        "device": "cuda",
        "compute_type": "float16",
        "language": "hi",
    }


def test_run_server_uses_lazy_model_factories_and_closes_on_keyboard_interrupt(
    monkeypatch, capsys
):
    init_log: list[tuple[str, ...]] = []

    class FakeWhisperTranscriber:
        def __init__(
            self,
            model_size: str = "small",
            device: str = "cpu",
            compute_type: str = "int8",
            language: str | None = None,
        ):
            init_log.append(("whisper", model_size, device, compute_type, language or ""))

    class FakePyannoteDiarizer:
        def __init__(self):
            init_log.append(("diarizer",))

    monkeypatch.setattr(cli, "WhisperTranscriber", FakeWhisperTranscriber)
    monkeypatch.setattr(cli, "PyannoteDiarizer", FakePyannoteDiarizer)

    server_state = {}

    class FakeServer:
        def __init__(self, service):
            self.service = service
            self.server_address = ("127.0.0.1", 8123)
            self.closed = False

        def serve_forever(self):
            server_state["started"] = True
            assert init_log == []
            assert self.service._transcriber is None
            assert self.service._diarizer is None
            self.service._get_transcriber()
            self.service._get_diarizer()
            assert init_log == [
                ("whisper", "large", "cuda", "float16", "hi"),
                ("diarizer",),
            ]
            raise KeyboardInterrupt

        def server_close(self):
            self.closed = True
            server_state["closed"] = True

    def fake_server_factory(host, port, service):
        server_state["host"] = host
        server_state["port"] = port
        server_state["service"] = service
        server_state["server"] = FakeServer(service)
        return server_state["server"]

    cli.run_server(
        "127.0.0.1",
        8123,
        "large",
        "cuda",
        "float16",
        "hi",
        server_factory=fake_server_factory,
    )

    output = capsys.readouterr().out.strip().splitlines()
    assert output == ["http://127.0.0.1:8123"]
    assert server_state["host"] == "127.0.0.1"
    assert server_state["port"] == 8123
    assert server_state["closed"] is True
    assert isinstance(server_state["service"], cli.SpeakerTranscriptionService)


@pytest.mark.parametrize(
    ("argv", "flag"),
    [
        (["stream", "--audio", "tests/fixtures/audio/a.wav", "--chunk-ms", "0"], "--chunk-ms"),
        (["stream", "--audio", "tests/fixtures/audio/a.wav", "--agreement", "0"], "--agreement"),
        (["stream", "--audio", "tests/fixtures/audio/a.wav", "--energy-threshold", "-1"], "--energy-threshold"),
        (
            [
                "benchmark",
                "--audio",
                "tests/fixtures/audio/a.wav",
                "--chunk-ms",
                "0",
            ],
            "--chunk-ms",
        ),
        (
            [
                "benchmark",
                "--audio",
                "tests/fixtures/audio/a.wav",
                "--agreement",
                "-1",
            ],
            "--agreement",
        ),
        (
            [
                "benchmark",
                "--audio",
                "tests/fixtures/audio/a.wav",
                "--energy-threshold",
                "-1",
            ],
            "--energy-threshold",
        ),
    ],
)
def test_stream_and_benchmark_reject_nonpositive_numeric_args(argv, flag):
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(argv)
