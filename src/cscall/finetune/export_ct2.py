"""Build the ct2-transformers-converter command to export a merged Whisper to CT2.

Fine-tuning produces LoRA adapters; before export they must be MERGED into the base
model (peft `merge_and_unload`) in the Colab notebook. This module only constructs
the converter command so it is unit-testable without running the heavy conversion.
"""
_VALID_QUANT = {"int8", "int8_float16", "float16", "float32"}


def build_convert_command(
    merged_model_dir: str,
    output_dir: str,
    quantization: str = "int8",
    force: bool = False,
) -> list[str]:
    """Return argv for ct2-transformers-converter. Faster-whisper loads the result."""
    if quantization not in _VALID_QUANT:
        raise ValueError(
            f"quantization must be one of {sorted(_VALID_QUANT)}, got {quantization!r}"
        )
    cmd = [
        "ct2-transformers-converter",
        "--model", merged_model_dir,
        "--output_dir", output_dir,
        "--quantization", quantization,
    ]
    if force:
        cmd.append("--force")
    return cmd
