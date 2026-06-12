from data.build_hiacc_manifests import (
    parse_transcription_line,
    build_split_rows,
    build_parser,
)


def test_parse_splits_on_first_comma_only():
    # transcript contains commas; only the first separates wav from text
    wav, text = parse_transcription_line("AD09002.wav, hello, world, ji")
    assert wav == "AD09002.wav"
    assert text == "hello, world, ji"


def test_parse_blank_line_returns_none():
    assert parse_transcription_line("   ") is None


def test_build_split_rows(tmp_path):
    # minimal HiACC-shaped layout
    root = tmp_path / "adult"
    (root / "transcription").mkdir(parents=True)
    (root / "audio" / "test_split").mkdir(parents=True)
    (root / "transcription" / "combined_output_changed_test_output.txt").write_text(
        "AD09001.wav, order कहां है\nAD13005.wav, please wait\n", encoding="utf-8"
    )
    rows = build_split_rows(str(root), "test")
    assert len(rows) == 2
    r = rows[0]
    assert r.id == "AD09001"
    assert r.speaker == "AD09"
    assert r.lang == "hi-en"
    assert r.audio_path.endswith("audio/test_split/AD09001.wav")
    assert r.text == "order कहां है"
    # 2 of 3 tokens Devanagari -> ~0.667
    assert r.cs_density == round(2 / 3, 4)
    assert r.cs_bucket == "mid"  # 0.6667 < 0.67 boundary -> mid


def test_build_parser_requires_root_and_prefix():
    import pytest
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
