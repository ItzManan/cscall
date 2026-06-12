"""Evaluation manifest: one JSONL row per utterance."""
import json
from dataclasses import dataclass
from typing import Optional

_REQUIRED = ("id", "audio_path", "text")


@dataclass
class Utterance:
    id: str
    audio_path: str
    text: str
    speaker: Optional[str] = None
    lang: Optional[str] = None
    accent: Optional[str] = None
    cs_density: Optional[float] = None


def load_manifest(path: str) -> list[Utterance]:
    """Load a JSONL manifest into Utterance objects, validating required fields."""
    utts: list[Utterance] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for key in _REQUIRED:
                if key not in row:
                    raise ValueError(f"{path}:{lineno} missing required field '{key}'")
            utts.append(
                Utterance(
                    id=row["id"],
                    audio_path=row["audio_path"],
                    text=row["text"],
                    speaker=row.get("speaker"),
                    lang=row.get("lang"),
                    accent=row.get("accent"),
                    cs_density=row.get("cs_density"),
                )
            )
    return utts
