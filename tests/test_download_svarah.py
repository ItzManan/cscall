from data.download_svarah import build_parser


def test_dry_run_flag_parses():
    args = build_parser().parse_args(["--out", "data/raw/svarah", "--dry-run"])
    assert args.out == "data/raw/svarah"
    assert args.dry_run is True


def test_out_is_required():
    import pytest
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
