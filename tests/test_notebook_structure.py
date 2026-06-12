import json


def test_notebook_is_valid_and_has_sections():
    with open("notebooks/phase1_finetune_colab.ipynb", encoding="utf-8") as f:
        nb = json.load(f)
    assert nb["nbformat"] >= 4
    sources = "\n".join(
        "".join(cell.get("source", [])) for cell in nb["cells"]
    )
    # references the project modules it must wire together
    assert "cscall.finetune.config" in sources
    assert "cscall.finetune.dataset" in sources
    assert "peft" in sources
    assert "merge_and_unload" in sources
    assert "ct2-transformers-converter" in sources or "build_convert_command" in sources
