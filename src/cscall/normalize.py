"""Text normalization for code-switch WER scoring.

The eval set mixes romanized Hindi, Devanagari, and English. To score fairly we
canonicalize everything to a single lowercase, punctuation-free, romanized form
BEFORE computing WER/CER, so "कहां" and "kahan" are not counted as errors.
"""
import re
import unicodedata

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")


def romanize_devanagari(text: str) -> str:
    """Transliterate any Devanagari runs to ASCII (ITRANS), leaving other text."""
    if not _DEVANAGARI_RE.search(text):
        return text.lower()
    romanized = transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
    return romanized.lower()


def normalize_text(text: str) -> str:
    """Canonicalize text for WER/CER: romanize, lowercase, strip punctuation, collapse ws."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = romanize_devanagari(text)
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()
