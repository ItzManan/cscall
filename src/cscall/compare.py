"""Run two transcribers over the same manifest and report the WER/CER delta.

Reuses the Phase 0 eval_runner so the comparison is apples-to-apples with the
baseline numbers. Positive delta == fine-tuned is better.
"""
from typing import Optional

from cscall.eval_runner import Transcriber, run_eval
from cscall.manifest import Utterance


def compare_models(
    utterances: list[Utterance],
    baseline: Transcriber,
    finetuned: Transcriber,
    group_by: Optional[str] = None,
) -> dict:
    """Return {baseline, finetuned, delta_wer, delta_cer} reports."""
    base = run_eval(utterances, baseline, group_by=group_by)
    fine = run_eval(utterances, finetuned, group_by=group_by)
    return {
        "baseline": base,
        "finetuned": fine,
        "delta_wer": base["overall"]["wer"] - fine["overall"]["wer"],
        "delta_cer": base["overall"]["cer"] - fine["overall"]["cer"],
    }


def render_comparison_markdown(result: dict) -> str:
    """Render a before/after table with per-group rows."""
    lines = [
        "| Group | Baseline WER | Fine-tuned WER | Δ WER |",
        "|---|---|---|---|",
    ]
    bo = result["baseline"]["overall"]
    fo = result["finetuned"]["overall"]
    lines.append(
        f"| **overall** | {bo['wer']:.3f} | {fo['wer']:.3f} | {bo['wer'] - fo['wer']:+.3f} |"
    )
    bgroups = result["baseline"].get("groups", {})
    fgroups = result["finetuned"].get("groups", {})
    for key in sorted(bgroups):
        bw = bgroups[key]["wer"]
        fw = fgroups.get(key, {}).get("wer", float("nan"))
        lines.append(f"| {key} | {bw:.3f} | {fw:.3f} | {bw - fw:+.3f} |")
    return "\n".join(lines)
