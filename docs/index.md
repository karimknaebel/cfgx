---
icon: lucide/settings
---

# cfgx

Keep configuration logic in regular Python modules. Start with a single dictionary, then scale into lazy computed values, inheritance chains, and CLI-friendly overrides without learning a new DSL.

> Everything you write stays Python: functions, conditionals, list comprehensions, imports. `cfgx` focuses on loading, layering, and mutating dictionaries so you can drop the result into any workflow.

## Highlights

- Load any Python config module with `load`.
- Compose configs via `parents = [...]` chains or by supplying multiple paths at once.
- Compute values lazily with `Lazy`.
- Adjust values on the fly using `apply_overrides` and a compact CLI syntax.
- Control merge behavior with `Delete()` and `Replace(value)`.
- Snapshot final dictionaries back to Python with `dump`, or pretty-print them with `format`.

## Quick tour

1. **Define a base config** as a plain dictionary:

    ```python
    # configs/base.py
    config = {
        "model": {
            "encoder": {"channels": 64},
            "head": {"in_channels": 64, "out_channels": 10},
        },
        "optimizer": {"type": "adam", "lr": 3e-4},
        "trainer": {"max_steps": 50_000},
    }
    ```

2. **Extend it with parents and lazy values** when the project grows:

    ```python
    # configs/finetune.py
    from cfgx import Lazy

    parents = ["base.py"]

    config = {
        "model": {
            "head": {"out_channels": 10},
        },
        "scheduler": {
            "type": "linear_warmup_cosine_decay",
            "warmup_steps": 1_000,
            "decay_steps": Lazy(
                lambda cfg: cfg["trainer"]["max_steps"]
                - cfg["scheduler"]["warmup_steps"]
            ),
        },
        "trainer": {"max_steps": 10_000},
    }
    ```

3. **Load everything** and apply runtime tweaks:

    ```python
    from cfgx import load

    cfg = load("configs/finetune.py", overrides=["model.head.out_channels=5"])
    ```

4. **Feed the dictionary wherever you need it**—serialize to JSON, log to disk, or hand it to any factory you already use:

    ```python
    import json

    with open("runs/finetune_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    ```

## Anatomy of a config module

Each config file is just Python. The loader only pays attention to two attributes:

- `config`: dictionary.
- `parents`: string or list of strings pointing to other config files (paths resolved relative to the current file).

Use `Lazy` when you want values derived from other parts of the merged config.

```python
from cfgx import Lazy

config = {
    "trainer": {"max_steps": 50_000},
    "scheduler": {
        "warmup_steps": 1_000,
        "decay_steps": Lazy(
            lambda cfg: cfg["trainer"]["max_steps"] - cfg["scheduler"]["warmup_steps"]
        ),
    },
}
```

`Lazy` also accepts a string expression. Use `c` as the config proxy:

```python
config = {
    "trainer": {"max_steps": 50_000},
    "warmup_steps": Lazy("c.trainer.max_steps * 0.1"),
}
```

### Multiple parents or paths

Parents are loaded depth-first (left to right), so later parents override earlier ones before the child applies its updates.

```python
# configs/experiment.py
parents = ["base.py", "schedules/cosine.py"]

config = {"trainer": {"max_epochs": 40}}
```

`load` also accepts a sequence of paths when you want to compose parts dynamically:

```python
cfg = load(
    [
        "configs/base.py",
        "configs/backbones/resnet.py",
        "configs/modes/eval.py",
    ]
)
```

## Runtime overrides

`apply_overrides(config_dict, sequence_of_strings)` mutates the dictionary in place. Each string uses a compact syntax designed for CLI usage.

- `path=value` → assign (dict keys or list indices)
- `path+=value` → append to a list
- `path-=value` → remove a matching element from a list
- `path!=` → delete a key or remove a list index

Values are parsed with `ast.literal_eval`, so strings, numbers, booleans, lists, dictionaries, and `None` all work. If parsing fails, the raw string is used, so most string values do not need to be quoted.

```python
apply_overrides(
    cfg,
    [
        "optimizer.lr=5e-4",
        "trainer.max_steps=10_000",
        "trainer.hooks+='wandb'",
        "trainer.hooks-='checkpoint'",
        "data.pipeline[0]!=",
    ],
)
```

## Merge semantics

When configs are layered, `cfgx` walks the override dictionary and combines it with the base using:

- Dicts merge recursively.
- `Delete()` removes the key entirely.
- `Replace(value)` uses `value` as-is without deeper merging.
- Otherwise the override value replaces the base.

```python
from cfgx import Delete, Replace, merge

base = {
    "optimizer": {
        "lr": 3e-4,
        "weight_decay": 0.01,
        "schedule": {"type": "linear", "warmup": 1_000},
    },
    "trainer": {"hooks": ["progress", "checkpoint"]},
}

override = {
    "optimizer": {
        "weight_decay": Delete(),
        "schedule": Replace({"type": "cosine", "t_max": 20_000}),
    },
    "trainer": {"steps": 10_000, "hooks": ["progress"]},
}

merged = merge(base, override)
```

`merge` is exported in case you want to reuse the algorithm, but `load` and `apply_overrides` already rely on it internally.

## Formatting and snapshots

Freeze the exact configuration you ran:

```python
from pathlib import Path
from cfgx import dump, format

print(format(cfg))  # Black-formatted string
dump(cfg, Path("runs/2024-01-10/config_snapshot.py"))
```

- `format` returns a nicely formatted string—useful for logging.
- `dump` writes the same representation to disk with a short header and `# fmt: off`. Because the file is valid Python, you can load it again with `load`.

## Tips for structuring configs

- Organize by concern: `configs/data/imagenet.py`, `configs/optim/adamw.py`, `configs/modes/eval.py`.
- Expose helper functions alongside `config` for reusable snippets.
- Pair with object builders only if you want to: the output is a plain dict, so you can plug it into your own factories or formats.
- Prefer `Lazy` for derived values over repeated overrides when you always compute the same thing.


::: cfgx
    options:
        show_root_heading: true
        heading: "API reference"
        toc_label: "API reference"
        summary:
            attributes: true
            classes: true
            functions: true
