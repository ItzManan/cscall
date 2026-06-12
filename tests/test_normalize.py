from cscall.normalize import normalize_text, romanize_devanagari


def test_lowercases_and_strips_punctuation():
    assert normalize_text("Hello, World!!") == "hello world"


def test_collapses_whitespace():
    assert normalize_text("  too   many\tspaces  ") == "too many spaces"


def test_romanizes_devanagari_word():
    # मेरा -> "meraa" in ITRANS-style romanization (lowercased)
    out = romanize_devanagari("मेरा")
    assert "mer" in out
    assert out == out.lower()


def test_normalize_text_romanizes_mixed_script():
    # "order kahan hai" written half in Devanagari should romanize then normalize
    out = normalize_text("order कहां hai")
    assert "order" in out
    assert "hai" in out
    # the Devanagari word becomes ascii letters only
    assert all(ord(c) < 128 for c in out)


def test_empty_string():
    assert normalize_text("") == ""
