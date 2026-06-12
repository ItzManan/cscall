"""CLI: `python -m cscall.cli baseline --manifest <path> [--model small] [--group-by accent]`."""
import argparse
import json

from cscall.asr_baseline import WhisperTranscriber
from cscall.compare import compare_models, render_comparison_markdown
from cscall.eval_runner import render_markdown, run_eval
from cscall.manifest import load_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cscall")
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("baseline", help="run baseline ASR eval over a manifest")
    b.add_argument("--manifest", required=True)
    b.add_argument("--model", default="small")
    b.add_argument("--group-by", dest="group_by", default=None)
    b.add_argument("--compute-type", dest="compute_type", default="int8")

    c = sub.add_parser("compare", help="baseline vs fine-tuned WER on a manifest")
    c.add_argument("--manifest", required=True)
    c.add_argument("--baseline-model", dest="baseline_model", default="small")
    c.add_argument("--finetuned-ct2", dest="finetuned_ct2", required=True)
    c.add_argument("--group-by", dest="group_by", default=None)
    c.add_argument("--compute-type", dest="compute_type", default="int8")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "baseline":
        utts = load_manifest(args.manifest)
        transcriber = WhisperTranscriber(
            model_size=args.model, compute_type=args.compute_type
        )
        report = run_eval(utts, transcriber.transcribe, group_by=args.group_by)
        print(render_markdown(report))
        print("\nJSON:\n" + json.dumps(report, indent=2))
    elif args.command == "compare":
        utts = load_manifest(args.manifest)
        baseline = WhisperTranscriber(
            model_size=args.baseline_model, compute_type=args.compute_type
        )
        finetuned = WhisperTranscriber(
            model_size=args.finetuned_ct2, compute_type=args.compute_type
        )
        result = compare_models(
            utts, baseline.transcribe, finetuned.transcribe, group_by=args.group_by
        )
        print(render_comparison_markdown(result))


if __name__ == "__main__":
    main()
