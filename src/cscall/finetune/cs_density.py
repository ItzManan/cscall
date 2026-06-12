"""Code-switch density: fraction of whitespace tokens written in Devanagari script.

This is a SCRIPT-based proxy (not language ID): romanized Hindi counts as 0.
Used to bucket eval utterances by how heavily code-switched they are.
"""
import re

_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")


def code_switch_density(text: str) -> float:
    """Return the fraction of tokens that contain at least one Devanagari char."""
    tokens = text.split()
    if not tokens:
        return 0.0
    deva = sum(1 for t in tokens if _DEVANAGARI_RE.search(t))
    return deva / len(tokens)
