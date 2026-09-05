# cfgx

[![PyPI version](https://img.shields.io/pypi/v/cfgx.svg)](https://pypi.org/project/cfgx/)

Python-first config loader with config composition, lazy computed values, and CLI-style overrides.

Docs: https://karimknaebel.github.io/cfgx/

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

Works well with [`specbuild`](https://github.com/karimknaebel/specbuild) when you want to build your model and other classes from config dictionaries.

## Composition model

Declare a dictionary, or a list of dictionaries and file paths:

```python
config = ["foo.py", {"x": 3}, "bar.py"]
```

cfgx expands file references, concatenates config lists, then merges the
dictionaries from left to right. It applies overrides afterward, then resolves
`Lazy` values. Lists inside dictionaries remain ordinary config data.

`Delete`, `Replace`, and `Update` act on the accumulated result at their position
in the sequence. Referenced configs are not merged independently, and repeated
references are applied each time they occur. Every file must define `config`.
The same composition works directly as `load("foo.py", {"x": 3}, "bar.py")`. See
[Config composition](docs/composition.md) for examples and the implications for
reusing configs.

Loading can modify mutable source values. For independent variants, create fresh
values for each load; see [Reusing configs](docs/composition.md#ownership-and-mutation).

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

config = [
    "base.py",
    {
        "model": {"depth": 12},
        "optimizer": {
            "weight_decay": Delete(),
            "schedule": Replace({"type": "cosine", "t_max": 40_000}),
        },
        "trainer": {"max_steps": 10_000},
        "scheduler": {
            "warmup_steps": 500,
            "decay_steps": Lazy(
                lambda c: c.trainer.max_steps - c.scheduler.warmup_steps
            ),
        },
    },
]
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

print(format(cfg))
with open("runs/finetune_config.py", "w") as f:
    dump(cfg, f, format="ruff")
```
