from cscall.finetune.export_ct2 import build_convert_command


def test_command_targets_merged_model_dir():
    cmd = build_convert_command(
        merged_model_dir="out/merged", output_dir="out/ct2", quantization="int8"
    )
    assert cmd[0] == "ct2-transformers-converter"
    assert "--model" in cmd
    assert "out/merged" in cmd
    assert "--output_dir" in cmd
    assert "out/ct2" in cmd
    assert "--quantization" in cmd
    assert "int8" in cmd


def test_force_flag_included():
    cmd = build_convert_command("m", "o", "int8", force=True)
    assert "--force" in cmd


def test_invalid_quantization_raises():
    import pytest
    with pytest.raises(ValueError, match="quantization"):
        build_convert_command("m", "o", "float64")
