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
