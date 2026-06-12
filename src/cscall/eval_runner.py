"""Run a transcriber over a manifest and compute baseline metrics.

The transcriber is injected as a callable (audio_path -> text) so this module is
testable without loading a model and so the same runner serves baseline and
fine-tuned models in later phases.
"""
from collections import defaultdict
from typing import Callable, Optional

from cscall.manifest import Utterance
from cscall.metrics import score

Transcriber = Callable[[str], str]

# Utterance attributes that make sense as breakdown dimensions.
GROUPABLE_FIELDS = ("speaker", "lang", "accent", "cs_density", "cs_bucket")


def run_eval(
    utterances: list[Utterance],
    transcribe: Transcriber,
    group_by: Optional[str] = None,
) -> dict:
    """Transcribe every utterance, return overall (and optionally grouped) metrics."""
    if group_by is not None and group_by not in GROUPABLE_FIELDS:
        raise ValueError(
            f"group_by must be one of {GROUPABLE_FIELDS}, got {group_by!r}"
        )
    refs = [u.text for u in utterances]
    hyps = [transcribe(u.audio_path) for u in utterances]
    report = {"overall": score(refs, hyps)}

    if group_by:
        buckets: dict[str, list[int]] = defaultdict(list)
        for i, u in enumerate(utterances):
            key = getattr(u, group_by)
            buckets[str(key)].append(i)
        report["groups"] = {
            key: score([refs[i] for i in idxs], [hyps[i] for i in idxs])
            for key, idxs in buckets.items()
        }
    return report


def render_markdown(report: dict) -> str:
    """Render a report dict as a markdown results table."""
    lines = ["| Group | WER | CER | N |", "|---|---|---|---|"]
    o = report["overall"]
    lines.append(f"| **overall** | {o['wer']:.3f} | {o['cer']:.3f} | {o['n']} |")
    for key, m in sorted(report.get("groups", {}).items()):
        lines.append(f"| {key} | {m['wer']:.3f} | {m['cer']:.3f} | {m['n']} |")
    return "\n".join(lines)
