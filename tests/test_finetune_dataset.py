from cscall.finetune.dataset import to_training_records, split_manifest
from cscall.manifest import load_manifest


def test_to_training_records_shape():
    utts = load_manifest("tests/fixtures/mini_manifest.jsonl")
    records = to_training_records(utts)
    assert len(records) == 3
    assert records[0] == {"audio_path": "tests/fixtures/audio/a.wav", "text": "order kahan hai"}


def test_split_is_deterministic_and_disjoint():
    utts = load_manifest("tests/fixtures/mini_manifest.jsonl")
    train1, test1 = split_manifest(utts, test_frac=0.34, seed=0)
    train2, test2 = split_manifest(utts, test_frac=0.34, seed=0)
    # deterministic
    assert [u.id for u in train1] == [u.id for u in train2]
    assert [u.id for u in test1] == [u.id for u in test2]
    # disjoint and complete
    ids_train = {u.id for u in train1}
    ids_test = {u.id for u in test1}
    assert ids_train.isdisjoint(ids_test)
    assert ids_train | ids_test == {"u1", "u2", "u3"}


def test_split_test_frac_size():
    utts = load_manifest("tests/fixtures/mini_manifest.jsonl")
    _, test = split_manifest(utts, test_frac=0.34, seed=0)
    assert len(test) == 1  # round(3 * 0.34) == 1


def test_write_manifest_roundtrips(tmp_path):
    from cscall.finetune.dataset import write_manifest
    from cscall.manifest import load_manifest
    utts = load_manifest("tests/fixtures/mini_manifest.jsonl")
    out = tmp_path / "rt.jsonl"
    write_manifest(utts, str(out))
    reloaded = load_manifest(str(out))
    assert [u.id for u in reloaded] == [u.id for u in utts]
    assert reloaded[0].text == utts[0].text
    assert reloaded[0].cs_density == utts[0].cs_density
    assert reloaded[0].speaker == utts[0].speaker
