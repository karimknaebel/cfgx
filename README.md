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
    "data": {"dataset": "imagenet", "batch_size": 128},
    "model": {"depth": 8, "width": 512, "dropout": 0.1},
    "optimizer": {"lr": 3e-4, "weight_decay": 0.01},
    "trainer": {"max_steps": 50_000, "mixed_precision": "bf16"},
}
```

```python
from cfgx import apply_overrides, load

cfg = load("configs/model.py")
cfg = apply_overrides(cfg, ["optimizer.lr=1e-3"])  # update nested keys
```

Works well with [`specbuild`](https://github.com/kabouzeid/specbuild) when you want to build your model and other classes from config dictionaries.
