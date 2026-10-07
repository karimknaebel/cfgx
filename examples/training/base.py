from cfgx import final

config = {
    "batch_size": 128,
    "steps": 96_000,
    "lr": 3e-4,
    "cooldown_steps": final.steps // 10,
    "optimizer": {
        "type": "AdamW",
        "params": [
            {"name": "backbone", "lr": final.lr * 0.1},
            {"name": "decoder", "lr": final.lr},
        ],
    },
    "scheduler": {
        "type": "warmup_stable_decay",
        "warmup_steps": 1_000,
        "total_steps": final.steps,
        "cooldown_steps": final.cooldown_steps,
    },
    "checkpointing": {"keep_steps": [final.steps - final.cooldown_steps]},
    "data": {"image_augmentation": ["flipping", "jittering", "blurring"]},
}
