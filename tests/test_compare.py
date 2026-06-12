from cscall.compare import compare_models, render_comparison_markdown
from cscall.manifest import load_manifest


def baseline_fake(audio_path: str) -> str:
    return {
        "tests/fixtures/audio/a.wav": "order kahan",          # drops "hai"
        "tests/fixtures/audio/b.wav": "your refund is processed",
        "tests/fixtures/audio/c.wav": "thoda wait karo",       # drops "please"
    }[audio_path]


def finetuned_fake(audio_path: str) -> str:
    # perfect — simulates the fine-tuned model fixing the errors
    return {
        "tests/fixtures/audio/a.wav": "order kahan hai",
        "tests/fixtures/audio/b.wav": "your refund is processed",
        "tests/fixtures/audio/c.wav": "thoda wait karo please",
    }[audio_path]


def test_compare_reports_both_and_delta():
    utts = load_manifest("tests/fixtures/mini_manifest.jsonl")
    result = compare_models(utts, baseline_fake, finetuned_fake)
    assert result["baseline"]["overall"]["wer"] > 0.0
    assert result["finetuned"]["overall"]["wer"] == 0.0
    # delta = baseline - finetuned (positive == improvement)
    assert result["delta_wer"] == result["baseline"]["overall"]["wer"]


def test_render_comparison_has_both_columns():
    utts = load_manifest("tests/fixtures/mini_manifest.jsonl")
    result = compare_models(utts, baseline_fake, finetuned_fake, group_by="accent")
    md = render_comparison_markdown(result)
    assert "Baseline WER" in md
    assert "Fine-tuned WER" in md
    assert "north" in md
