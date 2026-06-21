"""CLI for baseline, compare, and streaming demo workflows."""
import argparse
import json
import os
import tempfile
import wave

from cscall.asr_baseline import WhisperTranscriber
from cscall.compare import compare_models, render_comparison_markdown
from cscall.eval_runner import render_markdown, run_eval
from cscall.manifest import load_manifest
from cscall.streaming.audio import WavInfo, is_speech_pcm, validate_pcm_wav
from cscall.streaming.endpointing import EndpointConfig, EndpointDetector
from cscall.streaming.session import AudioChunk, StreamingSession


def _add_stream_parser(subparsers: argparse._SubParsersAction) -> None:
    stream = subparsers.add_parser("stream", help="run the phase 2 streaming demo")
    stream.add_argument("--audio", required=True)
    stream.add_argument("--model", default="small")
    stream.add_argument("--chunk-ms", dest="chunk_ms", type=int, default=500)
    stream.add_argument("--agreement", type=int, default=2)
    stream.add_argument("--compute-type", dest="compute_type", default="int8")
    stream.add_argument("--device", default="cpu")
    stream.add_argument("--energy-threshold", dest="energy_threshold", type=int, default=200)
    stream.add_argument("--fake-transcript", dest="fake_transcript", default=None)


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


def _print_stream_events(events) -> None:
    for event in events:
        if event.type == "metrics" and event.metrics is not None:
            print(event.metrics.render())
        elif event.text:
            print(f"{event.type}\t{event.timestamp_ms}\t{event.text}")
        else:
            print(f"{event.type}\t{event.timestamp_ms}")


def _run_stream(args: argparse.Namespace) -> None:
    wav_info = validate_pcm_wav(args.audio)
    chunks, _ = _iter_wav_chunks(args.audio, args.chunk_ms, args.energy_threshold)
    endpoint_detector = EndpointDetector(EndpointConfig(frame_ms=args.chunk_ms))

    if args.fake_transcript is not None:

        def transcribe(_audio: bytes) -> str:
            return args.fake_transcript

    else:
        transcriber = WhisperTranscriber(
            model_size=args.model, device=args.device, compute_type=args.compute_type
        )
        transcribe = _build_wav_transcribe(transcriber, wav_info)

    session = StreamingSession(
        transcribe=transcribe,
        step_ms=args.chunk_ms,
        agreement=args.agreement,
        endpoint_detector=endpoint_detector,
    )

    _append_silence_chunks(chunks, args.chunk_ms, endpoint_detector)
    for chunk in chunks:
        _print_stream_events(session.update(chunk))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cscall")
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("baseline", help="run baseline ASR eval over a manifest")
    b.add_argument("--manifest", required=True)
    b.add_argument("--model", default="small")
    b.add_argument("--group-by", dest="group_by", default=None)
    b.add_argument("--compute-type", dest="compute_type", default="int8")
    b.add_argument("--device", default="cpu", help="cpu or cuda")

    c = sub.add_parser("compare", help="baseline vs fine-tuned WER on a manifest")
    c.add_argument("--manifest", required=True)
    c.add_argument("--baseline-model", dest="baseline_model", default="small")
    c.add_argument("--finetuned-ct2", dest="finetuned_ct2", required=True)
    c.add_argument("--group-by", dest="group_by", default=None)
    c.add_argument("--compute-type", dest="compute_type", default="int8")
    c.add_argument("--device", default="cpu", help="cpu or cuda")

    _add_stream_parser(sub)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "baseline":
        utts = load_manifest(args.manifest)
        transcriber = WhisperTranscriber(
            model_size=args.model, device=args.device, compute_type=args.compute_type
        )
        report = run_eval(utts, transcriber.transcribe, group_by=args.group_by)
        print(render_markdown(report))
        print("\nJSON:\n" + json.dumps(report, indent=2))
    elif args.command == "compare":
        utts = load_manifest(args.manifest)
        baseline = WhisperTranscriber(
            model_size=args.baseline_model, device=args.device,
            compute_type=args.compute_type
        )
        finetuned = WhisperTranscriber(
            model_size=args.finetuned_ct2, device=args.device,
            compute_type=args.compute_type
        )
        result = compare_models(
            utts, baseline.transcribe, finetuned.transcribe, group_by=args.group_by
        )
        print(render_comparison_markdown(result))
    elif args.command == "stream":
        _run_stream(args)


if __name__ == "__main__":
    main()
