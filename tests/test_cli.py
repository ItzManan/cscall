from cscall.cli import build_parser


def test_baseline_subcommand_parses():
    parser = build_parser()
    args = parser.parse_args(
        ["baseline", "--manifest", "m.jsonl", "--model", "small", "--group-by", "accent"]
    )
    assert args.command == "baseline"
    assert args.manifest == "m.jsonl"
    assert args.model == "small"
    assert args.group_by == "accent"


def test_model_defaults_to_small():
    parser = build_parser()
    args = parser.parse_args(["baseline", "--manifest", "m.jsonl"])
    assert args.model == "small"
    assert args.group_by is None


def test_device_defaults_to_cpu():
    parser = build_parser()
    base = parser.parse_args(["baseline", "--manifest", "m.jsonl"])
    cmp = parser.parse_args(
        ["compare", "--manifest", "m.jsonl", "--finetuned-ct2", "out/ct2"]
    )
    assert base.device == "cpu"
    assert cmp.device == "cpu"


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
        ]
    )
    assert args.command == "compare"
    assert args.manifest == "m.jsonl"
    assert args.baseline_model == "small"
    assert args.finetuned_ct2 == "out/ct2"
    assert args.group_by == "accent"


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
    assert args.fake_transcript == "hello"
