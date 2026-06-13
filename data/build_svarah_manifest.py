"""Download the AI4Bharat Svarah eval set and build a cscall manifest.

Svarah is Indian-accented *English* speech. We stream the HF dataset, write each
clip to a 16 kHz wav under the output audio dir, and emit an Utterance manifest with
`accent` = speaker's native-place state (great for `--group-by accent` WER breakdowns)
and `lang="en"`. Requires `huggingface-cli login` (the dataset is gated).

NOTE on dependencies: use `datasets<4` (e.g. `uv pip install 'datasets<4'`).
datasets>=4 routes ALL audio access through `torchcodec` (a heavy torch+ffmpeg dep)
even with `Audio(decode=False)`; the <4 line returns raw bytes without it.

Usage:
    uv pip install 'datasets<4'        # one-time, into the project venv
    python data/build_svarah_manifest.py --out-manifest data/manifests/svarah.jsonl \
        --audio-dir data/raw/svarah --limit 400
"""
import argparse
import os

from cscall.manifest import Utterance
from cscall.finetune.dataset import write_manifest


def build_utterance(idx: int, ex: dict, audio_path: str) -> Utterance:
    """Map one Svarah record (+ its already-written audio path) to an Utterance.

    Pure mapping (no IO) so it is unit-testable; the audio file is written by the
    caller. Svarah is English, so cs_density/cs_bucket are fixed to 0 / "none".
    """
    return Utterance(
        id=f"svarah_{idx:05d}",
        audio_path=audio_path,
        text=(ex.get("text") or "").strip(),
        speaker=None,
        lang="en",
        accent=ex.get("native_place_state"),
        cs_density=0.0,
        cs_bucket="none",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build a cscall manifest from Svarah")
    p.add_argument("--out-manifest", dest="out_manifest", required=True)
    p.add_argument("--audio-dir", dest="audio_dir", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=0, help="cap number of clips (0 = all)")
    return p


def main(argv=None) -> None:
    # Heavy imports kept inside main so unit tests of build_utterance don't need them.
    from datasets import Audio, load_dataset

    args = build_parser().parse_args(argv)
    os.makedirs(args.audio_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_manifest) or ".", exist_ok=True)

    ds = load_dataset("ai4bharat/Svarah", split=args.split, streaming=True)
    # decode=False: hand us the raw file bytes (no torchcodec/ffmpeg decode needed);
    # faster-whisper decodes + resamples to 16 kHz at inference time.
    ds = ds.cast_column("audio", Audio(decode=False))

    rows: list[Utterance] = []
    for i, ex in enumerate(ds):
        if args.limit and i >= args.limit:
            break
        a = ex["audio"]
        ext = os.path.splitext(a.get("path") or "")[1] or ".wav"
        audio_path = os.path.join(args.audio_dir, f"svarah_{i:05d}{ext}")
        with open(audio_path, "wb") as fh:
            fh.write(a["bytes"])
        rows.append(build_utterance(i, ex, audio_path))
        if (i + 1) % 100 == 0:
            print(f"  ...{i + 1} clips")
    write_manifest(rows, args.out_manifest)
    print(f"wrote {len(rows)} utterances -> {args.out_manifest}")


if __name__ == "__main__":
    main()
