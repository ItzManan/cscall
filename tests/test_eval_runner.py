from cscall.eval_runner import run_eval, render_markdown
from cscall.manifest import load_manifest


def fake_transcriber(audio_path: str) -> str:
    # Deterministic fake keyed by file name; pretends to mis-hear u3.
    return {
        "tests/fixtures/audio/a.wav": "order kahan hai",
        "tests/fixtures/audio/b.wav": "your refund is processed",
        "tests/fixtures/audio/c.wav": "thoda wait karo",  # dropped "please"
    }[audio_path]


def test_run_eval_overall_metrics():
    utts = load_manifest("tests/fixtures/mini_manifest.jsonl")
    report = run_eval(utts, fake_transcriber)
    assert report["overall"]["n"] == 3
    # 1 deleted word out of 11 total reference words
    assert report["overall"]["wer"] > 0.0


def test_run_eval_groups_by_accent():
    utts = load_manifest("tests/fixtures/mini_manifest.jsonl")
    report = run_eval(utts, fake_transcriber, group_by="accent")
    assert set(report["groups"].keys()) == {"north", "south"}
    assert report["groups"]["north"]["wer"] == 0.0  # north utts transcribed perfectly


def test_render_markdown_contains_table():
    utts = load_manifest("tests/fixtures/mini_manifest.jsonl")
    report = run_eval(utts, fake_transcriber, group_by="accent")
    md = render_markdown(report)
    assert "| WER |" in md
    assert "north" in md
