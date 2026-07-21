"""Local Whisper LoRA fine-tuning entrypoint."""
import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Audio, Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from cscall.finetune.config import FineTuneConfig
from cscall.finetune.dataset import to_training_records
from cscall.finetune.export_ct2 import build_convert_command
from cscall.manifest import load_manifest


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        batch = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features],
            return_tensors="pt",
        )
        labels_batch = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features],
            return_tensors="pt",
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if labels[:, 0].eq(self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _ct2_command(merged_dir: Path, ct2_dir: Path, quantization: str) -> list[str]:
    cmd = build_convert_command(str(merged_dir), str(ct2_dir), quantization, force=True)
    if shutil.which(cmd[0]):
        return cmd
    local_converter = Path(sys.executable).with_name(cmd[0])
    if local_converter.exists():
        cmd[0] = str(local_converter)
    return cmd


def build_dataset(manifest: str, processor: WhisperProcessor, limit: int | None) -> Dataset:
    records = to_training_records(load_manifest(manifest))
    if limit is not None:
        records = records[:limit]
    ds = Dataset.from_list(records).cast_column("audio_path", Audio(sampling_rate=16000))

    def prepare(batch: dict) -> dict:
        audio = batch["audio_path"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=16000
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    return ds.map(prepare, remove_columns=ds.column_names)


def train(args: argparse.Namespace) -> None:
    cfg = FineTuneConfig()
    device = _device()
    processor_kwargs = {"task": cfg.task}
    if args.language not in {"auto", "none", ""}:
        processor_kwargs["language"] = args.language
    processor = WhisperProcessor.from_pretrained(cfg.base_model, **processor_kwargs)
    ds = build_dataset(args.manifest, processor, args.limit)

    model = WhisperForConditionalGeneration.from_pretrained(cfg.base_model)
    model.config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.target_modules,
            bias="none",
        ),
    )
    model.print_trainable_parameters()

    output_dir = Path(args.output_dir)
    train_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "train"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_epochs,
        max_steps=args.max_steps,
        warmup_steps=cfg.warmup_steps,
        fp16=device == "cuda",
        gradient_checkpointing=True,
        predict_with_generate=False,
        logging_steps=1 if args.max_steps > 0 else 10,
        save_strategy="no",
        remove_unused_columns=False,
        label_names=["labels"],
        report_to="none",
        dataloader_pin_memory=device == "cuda",
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=ds,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor),
        processing_class=processor,
    )
    print(f"Training on {device} with {len(ds)} examples")
    trainer.train()

    merged_dir = output_dir / "merged"
    merged = model.merge_and_unload()
    merged.generation_config.suppress_tokens = []
    merged.save_pretrained(merged_dir)
    processor.save_pretrained(merged_dir)
    print(f"Merged model saved to {merged_dir}")

    if not args.skip_ct2:
        cmd = _ct2_command(merged_dir, output_dir / "ct2", args.quantization)
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/codeswitch_train.jsonl")
    parser.add_argument("--output-dir", default="output_model/local")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--language", default="hi")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--quantization", default="int8")
    parser.add_argument("--skip-ct2", action="store_true")
    train(parser.parse_args(argv))


if __name__ == "__main__":
    main()
