import math
from pathlib import Path

import pytest

from cfgx import load

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "training"


def test_composed_training_example():
    cfg = load(EXAMPLES / "run.py")
    assert cfg["batch_size"] == 64
    assert cfg["steps"] == 48_000
    assert cfg["lr"] == pytest.approx(3e-4 * math.sqrt(0.5))
    assert cfg["checkpointing"]["keep_steps"] == [43_200]
    assert cfg["optimizer"]["params"] == [
        {"name": "backbone", "lr": cfg["lr"] * 0.1},
        {"name": "decoder", "lr": cfg["lr"]},
    ]
    assert cfg["model"]["decoder"]["in_channels"] == 384
    assert cfg["data"]["image_augmentation"] == ["flipping", "jittering"]
    assert cfg["data"]["datasets"] == {
        "hypersim": {
            "path": "data/hypersim",
            "label_type": "synthetic",
            "weight": pytest.approx(16 / 15),
            "image_augmentation": ["flipping", "jittering"],
        },
        "structured3d": {
            "path": "data/structured3d",
            "label_type": "synthetic",
            "weight": pytest.approx(32 / 15),
        },
        "blended_mvs": {
            "path": "data/blended_mvs",
            "label_type": "sfm",
            "weight": 0.8,
        },
    }


def test_training_overrides_recompute_dependents():
    cfg = load(
        EXAMPLES / "run.py",
        overrides=[
            "batch_size=32",
            "steps=24000",
            "lr=expr:value * 0.5",
            "model.backbone.width=768",
        ],
    )
    assert cfg["checkpointing"]["keep_steps"] == [21_600]
    assert cfg["scheduler"]["total_steps"] == 24_000
    assert cfg["scheduler"]["cooldown_steps"] == 2_400
    assert cfg["lr"] == pytest.approx(7.5e-5)
    assert cfg["optimizer"]["params"][0]["lr"] == cfg["lr"] * 0.1
    assert cfg["optimizer"]["params"][1]["lr"] == cfg["lr"]
    assert cfg["model"]["decoder"]["in_channels"] == 768


def test_training_sources_without_experiment_patches():
    cfg = load(EXAMPLES / "base.py", EXAMPLES / "model.py", EXAMPLES / "data.py")
    assert cfg["batch_size"] == 128
    assert cfg["steps"] == 96_000
    assert cfg["checkpointing"]["keep_steps"] == [86_400]
    assert cfg["data"]["datasets"]["hypersim"]["weight"] == 1
    assert cfg["data"]["datasets"]["structured3d"]["weight"] == 2
    assert cfg["data"]["datasets"]["blended_mvs"]["weight"] == 1
    assert "blurring" in cfg["data"]["image_augmentation"]
