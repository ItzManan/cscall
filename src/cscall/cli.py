"""CLI for baseline, compare, streaming demo, and speaker fusion workflows."""
import argparse
import json
import os
import tempfile
import wave

from cscall.asr_baseline import WhisperTranscriber
from cscall.compare import compare_models, render_comparison_markdown
from cscall.diarization import PyannoteDiarizer, diarization_error_rate, load_rttm
from cscall.fusion import fuse_words, render_speaker_transcript
from cscall.eval_runner import render_markdown, run_eval
from cscall.manifest import load_manifest
from cscall.streaming.audio import WavInfo, is_speech_pcm, validate_pcm_wav
from cscall.streaming.endpointing import EndpointConfig, EndpointDetector
from cscall.streaming.metrics import StreamingMetrics, summarize_metrics
from cscall.streaming.session import AudioChunk, StreamingEvent, StreamingSession


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def _add_benchmark_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("benchmark", help="benchmark the phase 2 streaming demo")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--audio", nargs="+")
    group.add_argument("--manifest")
    _add_stream_args(parser)


def _add_stream_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="small")
    parser.add_argument("--chunk-ms", dest="chunk_ms", type=_positive_int, default=500)
    parser.add_argument("--agreement", type=_positive_int, default=2)
    parser.add_argument("--compute-type", dest="compute_type", default="int8")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--language", default=None)
    parser.add_argument(
        "--energy-threshold",
        dest="energy_threshold",
        type=_non_negative_int,
        default=200,
    )
    parser.add_argument("--fake-transcript", dest="fake_transcript", default=None)


def _add_speaker_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="small")
    parser.add_argument("--compute-type", dest="compute_type", default="int8")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--language", default=None)


def _add_diarize_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("diarize", help="run two-speaker diarization on a WAV")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--reference-rttm", dest="reference_rttm", default=None)


def _add_transcribe_speakers_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "transcribe-speakers",
        help="transcribe a WAV with speaker attribution",
    )
    parser.add_argument("--audio", required=True)
    _add_speaker_model_args(parser)


def _add_stream_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("stream", help="run the phase 2 streaming demo")
    parser.add_argument("--audio", required=True)
    _add_stream_args(parser)


def _iter_wav_chunks(
    audio_path: str, chunk_ms: int, energy_threshold: int = 200
) -> tuple[list[AudioChunk], tuple[int, int, int]]:
    chunks: list[AudioChunk] = []
    with wave.open(audio_path, "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sampwidth = wav.getsampwidth()
        frames_per_chunk = max(1, int(sample_rate * chunk_ms / 1000))
        timestamp_ms = 0

        while True:
            data = wav.readframes(frames_per_chunk)
            if not data:
                break

            frames_read = len(data) // (channels * sampwidth)
            duration_ms = max(1, round(frames_read * 1000 / sample_rate))
            timestamp_ms += duration_ms
            try:
                is_speech = is_speech_pcm(data, sampwidth, energy_threshold)
            except ValueError as exc:
                if str(exc) == "PCM data must contain a whole number of frames":
                    raise ValueError(
                        f"{audio_path} is not a supported PCM WAV"
                    ) from None
                raise
            chunks.append(
                AudioChunk(
                    timestamp_ms=timestamp_ms,
                    duration_ms=duration_ms,
                    data=data,
                    is_speech=is_speech,
                )
            )

    return chunks, (sample_rate, channels, sampwidth)


def _append_silence_chunks(
    chunks: list[AudioChunk], chunk_ms: int, detector: EndpointDetector
) -> None:
    silence_chunks = max(
        1,
        (detector.config.trailing_silence_ms + detector.config.frame_ms - 1)
        // detector.config.frame_ms,
    )
    timestamp_ms = chunks[-1].timestamp_ms if chunks else 0
    for _ in range(silence_chunks):
        timestamp_ms += chunk_ms
        chunks.append(
            AudioChunk(
                timestamp_ms=timestamp_ms,
                duration_ms=chunk_ms,
                data=b"",
                is_speech=False,
            )
        )


def _validate_audio_inputs(audio_paths: list[str]) -> dict[str, WavInfo]:
    return {audio_path: validate_pcm_wav(audio_path) for audio_path in audio_paths}


def _benchmark_audio_paths(args: argparse.Namespace) -> list[str]:
    if args.manifest is not None:
        return [utterance.audio_path for utterance in load_manifest(args.manifest)]
    return list(args.audio)


def _print_diarization(turns) -> None:
    for turn in turns:
        print(f"{turn.start:.3f}\t{turn.end:.3f}\t{turn.speaker}")


def _run_diarize(args: argparse.Namespace) -> None:
    validate_pcm_wav(args.audio)
    if args.reference_rttm is not None:
        reference = load_rttm(args.reference_rttm)
    else:
        reference = None
    diarizer = PyannoteDiarizer()
    turns = diarizer.diarize(args.audio)
    _print_diarization(turns)
    if reference is not None:
        score = diarization_error_rate(reference, turns)
        print(f"DER: {score * 100:.2f}%")


def _run_transcribe_speakers(args: argparse.Namespace) -> None:
    validate_pcm_wav(args.audio)
    diarizer = PyannoteDiarizer()
    turns = diarizer.diarize(args.audio)
    transcriber = WhisperTranscriber(
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
    )
    words = transcriber.transcribe_words(args.audio)
    rendered = render_speaker_transcript(fuse_words(words, turns))
    if rendered:
        print(rendered)


def _build_transcribe(args, wav_info: WavInfo, transcriber: WhisperTranscriber | None = None):
    if args.fake_transcript is not None:

        def transcribe(_audio: bytes) -> str:
            return args.fake_transcript

        return transcribe

    model = transcriber or WhisperTranscriber(
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
    )
    return _build_wav_transcribe(model, wav_info)


def _build_wav_transcribe(transcriber: WhisperTranscriber, wav_info: WavInfo):
    def _transcribe(audio: bytes) -> str:
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_path = tmp.name
            with wave.open(temp_path, "wb") as wav_out:
                wav_out.setnchannels(wav_info.channels)
                wav_out.setsampwidth(wav_info.sample_width)
                wav_out.setframerate(wav_info.sample_rate)
                wav_out.writeframes(audio)
            return transcriber.transcribe(temp_path)
        finally:
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except FileNotFoundError:
                    pass

    return _transcribe


def _run_stream_session(
    args: argparse.Namespace,
    audio_path: str,
    transcriber: WhisperTranscriber | None = None,
    wav_info: WavInfo | None = None,
):
    wav_info = wav_info or validate_pcm_wav(audio_path)
    chunks, _ = _iter_wav_chunks(audio_path, args.chunk_ms, args.energy_threshold)
    endpoint_detector = EndpointDetector(EndpointConfig(frame_ms=args.chunk_ms))
    transcribe = _build_transcribe(args, wav_info, transcriber=transcriber)

    session = StreamingSession(
        transcribe=transcribe,
        step_ms=args.chunk_ms,
        agreement=args.agreement,
        endpoint_detector=endpoint_detector,
    )

    _append_silence_chunks(chunks, args.chunk_ms, endpoint_detector)
    events = []
    for chunk in chunks:
        events.extend(session.update(chunk))
    return events


def _print_stream_events(events) -> None:
    for event in events:
        if event.type == "metrics" and event.metrics is not None:
            print(event.metrics.render())
        elif event.text:
            print(f"{event.type}\t{event.timestamp_ms}\t{event.text}")
        else:
            print(f"{event.type}\t{event.timestamp_ms}")


def _run_stream(args: argparse.Namespace) -> None:
    events = _run_stream_session(args, args.audio)
    if not any(event.type == "metrics" for event in events):
        events.append(
            StreamingEvent(
                type="metrics",
                timestamp_ms=0,
                metrics=StreamingMetrics(),
            )
        )
    _print_stream_events(events)


def _format_benchmark_value(value):
    if value is None:
        return "n/a"
    return f"{value:g}"


def _render_benchmark_table(summary: dict[str, dict[str, int | float | None]]) -> str:
    rows = [
        "| Metric | p50 | p99 |",
        "|---|---:|---:|",
        "| RTF | "
        f"{_format_benchmark_value(summary['rtf']['p50'])} | "
        f"{_format_benchmark_value(summary['rtf']['p99'])} |",
        "| first_partial_ms | "
        f"{_format_benchmark_value(summary['first_partial_ms']['p50'])} | "
        f"{_format_benchmark_value(summary['first_partial_ms']['p99'])} |",
        "| final_ms | "
        f"{_format_benchmark_value(summary['final_ms']['p50'])} | "
        f"{_format_benchmark_value(summary['final_ms']['p99'])} |",
    ]
    return "\n".join(rows)


def _run_benchmark(args: argparse.Namespace) -> None:
    audio_paths = _benchmark_audio_paths(args)
    wav_infos = _validate_audio_inputs(audio_paths)
    transcriber = None
    if audio_paths and args.fake_transcript is None:
        transcriber = WhisperTranscriber(
            model_size=args.model,
            device=args.device,
            compute_type=args.compute_type,
            language=args.language,
        )

    metrics = []
    for audio_path in audio_paths:
        for event in _run_stream_session(
            args,
            audio_path,
            transcriber=transcriber,
            wav_info=wav_infos[audio_path],
        ):
            if event.type == "metrics" and event.metrics is not None:
                metrics.append(event.metrics)

    print(_render_benchmark_table(summarize_metrics(metrics)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cscall")
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("baseline", help="run baseline ASR eval over a manifest")
    b.add_argument("--manifest", required=True)
    b.add_argument("--model", default="small")
    b.add_argument("--group-by", dest="group_by", default=None)
    b.add_argument("--compute-type", dest="compute_type", default="int8")
    b.add_argument("--device", default="cpu", help="cpu or cuda")
    b.add_argument("--language", default=None)

    c = sub.add_parser("compare", help="baseline vs fine-tuned WER on a manifest")
    c.add_argument("--manifest", required=True)
    c.add_argument("--baseline-model", dest="baseline_model", default="small")
    c.add_argument("--finetuned-ct2", dest="finetuned_ct2", required=True)
    c.add_argument("--group-by", dest="group_by", default=None)
    c.add_argument("--compute-type", dest="compute_type", default="int8")
    c.add_argument("--device", default="cpu", help="cpu or cuda")
    c.add_argument("--language", default=None)

    _add_diarize_parser(sub)
    _add_transcribe_speakers_parser(sub)
    _add_stream_parser(sub)
    _add_benchmark_parser(sub)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "baseline":
        utts = load_manifest(args.manifest)
        transcriber = WhisperTranscriber(
            model_size=args.model,
            device=args.device,
            compute_type=args.compute_type,
            language=args.language,
        )
        report = run_eval(utts, transcriber.transcribe, group_by=args.group_by)
        print(render_markdown(report))
        print("\nJSON:\n" + json.dumps(report, indent=2))
    elif args.command == "compare":
        utts = load_manifest(args.manifest)
        baseline = WhisperTranscriber(
            model_size=args.baseline_model,
            device=args.device,
            compute_type=args.compute_type,
            language=args.language,
        )
        finetuned = WhisperTranscriber(
            model_size=args.finetuned_ct2,
            device=args.device,
            compute_type=args.compute_type,
            language=args.language,
        )
        result = compare_models(
            utts, baseline.transcribe, finetuned.transcribe, group_by=args.group_by
        )
        print(render_comparison_markdown(result))
    elif args.command == "stream":
        _run_stream(args)
    elif args.command == "benchmark":
        _run_benchmark(args)
    elif args.command == "diarize":
        _run_diarize(args)
    elif args.command == "transcribe-speakers":
        _run_transcribe_speakers(args)


if __name__ == "__main__":
    main()
