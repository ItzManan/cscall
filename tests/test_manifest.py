import pytest

from cscall.manifest import Utterance, load_manifest


def test_loads_all_rows():
    utts = load_manifest("tests/fixtures/mini_manifest.jsonl")
    assert len(utts) == 3
    assert all(isinstance(u, Utterance) for u in utts)


def test_fields_parsed():
    utts = load_manifest("tests/fixtures/mini_manifest.jsonl")
    u = utts[0]
    assert u.id == "u1"
    assert u.text == "order kahan hai"
    assert u.speaker == "customer"
    assert u.cs_density == 0.5


def test_missing_required_field_raises():
    import tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write('{"id": "x", "text": "no audio path"}\n')
        path = f.name
    try:
        with pytest.raises(ValueError, match="audio_path"):
            load_manifest(path)
    finally:
        os.unlink(path)


def test_optional_fields_default():
    import tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write('{"id": "y", "audio_path": "p.wav", "text": "hello"}\n')
        path = f.name
    try:
        u = load_manifest(path)[0]
        assert u.speaker is None
        assert u.cs_density is None
    finally:
        os.unlink(path)
