"""Convert the HiACC corpus into cscall JSONL manifests.

HiACC ships split-aligned transcripts as lines `WAVNAME.wav, <transcript>` under
each speaker-group's `transcription/`/`transcript/` folder, with the audio under
`audio/<split>_split/`. The speaker id is the filename prefix (e.g. AD09001 -> AD09).

Usage:
    python data/build_hiacc_manifests.py --root data/raw/hiacc/Corpus/adult \
        --out-prefix data/manifests/codeswitch

Writes <out-prefix>_train.jsonl, _val.jsonl, _test.jsonl.
"""
import argparse
import os

from cscall.finetune.dataset import write_manifest
from cscall.finetune.cs_density import code_switch_density
from cscall.manifest import Utterance

# Maps our split name -> (transcript filename substring, audio subdir).
# HiACC names adult transcripts "combined_output_changed_<split>_output.txt" and
# children transcripts "<split>_output.txt"; we match by the "<split>_output.txt"
# suffix so both layouts work.
SPLITS = ("train", "val", "test")


def parse_transcription_line(line: str) -> tuple[str, str] | None:
    """Parse one `WAVNAME.wav, transcript` line. Returns (wav, text) or None if blank.

    Splits on the FIRST comma only, since transcripts themselves contain commas.
    """
    line = line.strip()
    if not line:
        return None
    wav, _, text = line.partition(",")
    return wav.strip(), text.strip()


def _find_transcript_file(root: str, split: str) -> str:
    """Locate the transcript .txt for a split under <root>/transcription or /transcript."""
    for sub in ("transcription", "transcript"):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            if name.endswith(f"{split}_output.txt"):
                return os.path.join(d, name)
    raise FileNotFoundError(f"no {split} transcript under {root}/transcription|transcript")


def build_split_rows(root: str, split: str, lang: str = "hi-en") -> list[Utterance]:
    """Build Utterances for one split, pairing each transcript line with its wav."""
    transcript_file = _find_transcript_file(root, split)
    audio_dir = os.path.join(root, "audio", f"{split}_split")
    rows: list[Utterance] = []
    with open(transcript_file, encoding="utf-8") as fh:
        for line in fh:
            parsed = parse_transcription_line(line)
            if parsed is None:
                continue
            wav, text = parsed
            stem = wav[:-4] if wav.endswith(".wav") else wav
            rows.append(
                Utterance(
                    id=stem,
                    audio_path=os.path.join(audio_dir, wav),
                    text=text,
                    speaker=stem[:4],  # HiACC PID prefix, e.g. AD09001 -> AD09
                    lang=lang,
                    accent=None,
                    cs_density=round(code_switch_density(text), 4),
                )
            )
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build cscall manifests from HiACC")
    p.add_argument("--root", required=True, help="HiACC speaker-group dir (e.g. .../Corpus/adult)")
    p.add_argument("--out-prefix", dest="out_prefix", required=True,
                   help="output manifest prefix (e.g. data/manifests/codeswitch)")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    for split in SPLITS:
        rows = build_split_rows(args.root, split)
        out = f"{args.out_prefix}_{split}.jsonl"
        write_manifest(rows, out)
        print(f"wrote {len(rows):5d} utterances -> {out}")


if __name__ == "__main__":
    main()
