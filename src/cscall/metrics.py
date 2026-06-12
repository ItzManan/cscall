"""WER/CER scoring, applied AFTER code-switch normalization."""
import jiwer

from cscall.normalize import normalize_text


def score(references: list[str], hypotheses: list[str]) -> dict:
    """Compute corpus WER/CER over normalized text.

    Returns {"wer": float, "cer": float, "n": int}.
    """
    if len(references) != len(hypotheses):
        raise ValueError(
            f"reference/hypothesis count mismatch: {len(references)} vs {len(hypotheses)}"
        )
    refs = [normalize_text(r) for r in references]
    hyps = [normalize_text(h) for h in hypotheses]
    return {
        "wer": jiwer.wer(refs, hyps),
        "cer": jiwer.cer(refs, hyps),
        "n": len(refs),
    }
