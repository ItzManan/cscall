from cscall.finetune.cs_density import code_switch_density, cs_bucket


def test_cs_bucket_boundaries():
    assert cs_bucket(0.0) == "none"
    assert cs_bucket(0.2) == "low"
    assert cs_bucket(0.5) == "mid"
    assert cs_bucket(0.9) == "high"
    assert cs_bucket(1.0) == "high"


def test_all_english_is_zero():
    assert code_switch_density("please wait a moment") == 0.0


def test_all_devanagari_is_one():
    # every token is Devanagari script
    assert code_switch_density("नमस्ते खाना है") == 1.0


def test_half_and_half():
    # 2 of 4 tokens Devanagari (खाना, है)
    assert code_switch_density("order खाना है ready") == 0.5


def test_empty_is_zero():
    assert code_switch_density("") == 0.0


def test_romanized_hindi_counts_as_english_script():
    # script-based metric: romanized Hindi is Latin script, so density 0.0
    # (documents the known limitation — this measures SCRIPT mixing, not language)
    assert code_switch_density("thoda wait karo") == 0.0
