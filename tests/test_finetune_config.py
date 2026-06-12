from cscall.finetune.config import FineTuneConfig, default_lora_target_modules


def test_defaults_are_t4_friendly():
    cfg = FineTuneConfig()
    assert cfg.base_model == "openai/whisper-small"
    assert cfg.lora_r == 16
    assert cfg.lora_alpha == 32
    assert cfg.batch_size <= 16  # fits a T4
    assert cfg.language == "hi"


def test_target_modules_are_attention_projections():
    mods = default_lora_target_modules()
    assert "q_proj" in mods
    assert "v_proj" in mods


def test_to_dict_roundtrip():
    cfg = FineTuneConfig(lora_r=8)
    d = cfg.to_dict()
    assert d["lora_r"] == 8
    assert "base_model" in d
