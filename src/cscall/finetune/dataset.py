"""Convert eval manifests into training records and deterministic train/test splits.

A training record is the minimal {audio_path, text} the Colab training loop needs;
keeping it tiny avoids coupling the GPU loop to the full Utterance schema.
"""
import random

from cscall.manifest import Utterance


def to_training_records(utterances: list[Utterance]) -> list[dict]:
    """Project Utterances to minimal training records."""
    return [{"audio_path": u.audio_path, "text": u.text} for u in utterances]


def split_manifest(
    utterances: list[Utterance],
    test_frac: float = 0.2,
    seed: int = 0,
) -> tuple[list[Utterance], list[Utterance]]:
    """Deterministically split into (train, test) by a shuffled copy.

    Seeded so the split is reproducible; test set is round(n * test_frac) items.
    """
    if not 0.0 <= test_frac <= 1.0:
        raise ValueError(f"test_frac must be in [0,1], got {test_frac}")
    items = list(utterances)
    random.Random(seed).shuffle(items)
    n_test = round(len(items) * test_frac)
    test = items[:n_test]
    train = items[n_test:]
    return train, test
