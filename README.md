# cfgx

[![PyPI version](https://img.shields.io/pypi/v/cfgx.svg)](https://pypi.org/project/cfgx/)

Python-first config loader with parent chaining, lazy computed values, and CLI-style overrides.

Docs: https://kabouzeid.github.io/cfgx/

## Install

```bash
pip install cfgx
```

## Quick start

Example config file:

```python
# configs/model.py
config = {
    "model": {"name": "resnet18"},
    "optimizer": {"lr": 3e-4},
}
```

```python
from cfgx import load

cfg = load("configs/model.py", overrides=["optimizer.lr=1e-3"])
```

Works well with [`specbuild`](https://github.com/kabouzeid/specbuild) when you want to build your model and other classes from config dictionaries.

## Advanced example

Base config:

```python
# configs/base.py
from cfgx import Lazy

config = {
    "model": {"depth": 8, "width": 512},
    "optimizer": {
        "lr": 3e-4,
        "weight_decay": 0.01,
        "schedule": {"type": "linear", "warmup_steps": 1_000},
    },
    "trainer": {
        "max_steps": 50_000,
        "hooks": ["progress", "checkpoint"],
        "stages": [{"name": "warmup", "max_steps": 5_000}],
        "log_every": Lazy("c.trainer.max_steps // 100"),
    },
}
```

Derived config:

```python
# configs/finetune.py
from cfgx import Delete, Lazy, Replace

parents = ["base.py"]

config = {
    "model": {"depth": 12},
    "optimizer": {
        "weight_decay": Delete(),
        "schedule": Replace({"type": "cosine", "t_max": 40_000}),
    },
    "trainer": {"max_steps": 10_000},
    "scheduler": {
        "warmup_steps": 500,
        "decay_steps": Lazy(lambda c: c.trainer.max_steps - c.scheduler.warmup_steps),
    },
}
```

Load, override, and snapshot:

```python
from cfgx import dump, format, load

cfg = load(
    "configs/finetune.py",
    overrides=[
        "optimizer.lr=1e-4",
        "trainer.hooks+=wandb",
        "trainer.hooks-='checkpoint'",
        "trainer.stages[0].max_steps=2_000",
        "scheduler.warmup_steps=lazy:c.trainer.max_steps * 0.1",
    ],
)

print(format(cfg, format="pprint"))
with open("runs/finetune_config.py", "w") as f:
    dump(cfg, f, format="ruff")
```
