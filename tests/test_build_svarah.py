from data.build_svarah_manifest import build_utterance, build_parser


def test_build_utterance_maps_fields():
    ex = {"text": " Hello world ", "native_place_state": "Kerala"}
    u = build_utterance(7, ex, "data/raw/svarah")
    assert u.id == "svarah_00007"
    assert u.audio_path == "data/raw/svarah/svarah_00007.wav"
    assert u.text == "Hello world"
    assert u.lang == "en"
    assert u.accent == "Kerala"
    assert u.cs_density == 0.0
    assert u.cs_bucket == "none"


def test_build_utterance_handles_missing_fields():
    u = build_utterance(0, {}, "d")
    assert u.text == ""
    assert u.accent is None


def test_build_parser_requires_outputs():
    import pytest
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
