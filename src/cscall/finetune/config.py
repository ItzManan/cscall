"""Hyperparameters for LoRA fine-tuning Whisper. Pure data — no training here."""
from dataclasses import asdict, dataclass, field


def default_lora_target_modules() -> list[str]:
    """Attention projection layers LoRA adapts in Whisper."""
    return ["q_proj", "v_proj"]


@dataclass
class FineTuneConfig:
    base_model: str = "openai/whisper-small"
    language: str = "hi"          # Whisper task language tag for decoding
    task: str = "transcribe"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=default_lora_target_modules)
    batch_size: int = 8           # per-device; T4-friendly with fp16 + grad checkpointing
    grad_accum: int = 2
    learning_rate: float = 1e-3   # LoRA tolerates higher LR than full fine-tune
    num_epochs: int = 3
    warmup_steps: int = 50

    def to_dict(self) -> dict:
        return asdict(self)
