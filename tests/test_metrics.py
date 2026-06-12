from cscall.metrics import score


def test_perfect_match_zero_wer():
    r = score(["hello world"], ["hello world"])
    assert r["wer"] == 0.0
    assert r["cer"] == 0.0
    assert r["n"] == 1


def test_one_substitution():
    # 2 words, 1 wrong -> WER 0.5
    r = score(["hello world"], ["hello there"])
    assert r["wer"] == 0.5


def test_normalization_ignores_script_and_case():
    # Full Devanagari reference vs romanized + capitalized hypothesis: after
    # normalization (ITRANS romanization + lowercasing) they are identical, so a
    # script/case-only difference must not be counted as ASR errors.
    r = score(["नमस्ते खाना है"], ["Namaste Khana Hai"])
    assert r["wer"] == 0.0


def test_multiple_utterances_aggregate():
    r = score(["a b", "c d"], ["a b", "c x"])
    assert r["n"] == 2
    assert r["wer"] == 0.25  # 1 error / 4 words


def test_mismatched_lengths_raises():
    import pytest
    with pytest.raises(ValueError):
        score(["a"], ["a", "b"])


def test_empty_corpus_returns_zero():
    r = score([], [])
    assert r == {"wer": 0.0, "cer": 0.0, "n": 0}
