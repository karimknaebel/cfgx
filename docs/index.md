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

## Core workflow

Start with a config file, load it, and apply CLI-style tweaks.

!!! example "Define a config"
    ```python
    # configs/base.py
    config = {
        "model": {"name": "resnet18"},
        "trainer": {"max_steps": 50_000},
    }
    ```

!!! example "Load with overrides"
    ```python
    from cfgx import load

    cfg = load("configs/base.py", overrides=["trainer.max_steps=12_000"])
    ```

The result is a plain dictionary you can serialize, log, or feed into factories.

## Config modules

Each config file is just Python. The loader only pays attention to two attributes:

- `config`: dictionary.
- `parents`: string or list of strings pointing to other config files (paths resolved relative to the current file).

!!! example "Parent chaining"
    ```python
    # configs/finetune.py
    parents = ["base.py", "schedules/cosine.py"]

    config = {"trainer": {"max_steps": 10_000}}
    ```

You can also compose multiple files by passing a sequence of paths to `load`.

!!! example "Multiple paths"
    ```python
    from cfgx import load

    cfg = load(
        [
            "configs/base.py",
            "configs/backbones/resnet.py",
            "configs/modes/eval.py",
        ]
    )
    ```

## Runtime overrides

Overrides can be passed to `load` or applied later with
`apply_overrides(config_dict, sequence_of_strings)`, which mutates the
dictionary in place. Each string uses a compact syntax designed for CLI usage.

- `path=value` → assign (dict keys or list indices)
- `path+=value` → append to a list
- `path-=value` → remove a matching element from a list
- `path!=` → delete a key or remove a list index

Values are parsed with `ast.literal_eval`, so strings, numbers, booleans, lists,
dictionaries, and `None` all work. If parsing fails, the raw string is used, so
most string values do not need to be quoted. You can also use `lazy:` to define
a `Lazy` expression from the CLI (see [Lazy values](#lazy-values)).

Assignments create intermediate dicts and extend lists with `None` as needed.
List indices follow Python semantics: negative indices are allowed when the list
already exists and are in range (otherwise `IndexError`). Deletes and list
removals are forgiving no-ops when the path is missing or out of range.

!!! example "Override syntax"
    ```python
    from cfgx import apply_overrides

    apply_overrides(
        cfg,
        [
            "optimizer.lr=5e-4",
            "trainer.max_steps=10_000",
            "trainer.hooks+='wandb'",
            "trainer.hooks-='checkpoint'",
            "data.pipeline[0]!=",
            "trainer.warmup_steps=lazy:c.trainer.max_steps * 0.1",
        ],
    )
    ```

## Merge semantics

When configs are layered, `cfgx` walks the override dictionary and combines it
with the base using:

- Dicts merge recursively.
- `Delete()` removes the key entirely.
- `Replace(value)` uses `value` as-is without deeper merging.
- Otherwise the override value replaces the base.

!!! example "Delete and Replace"
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

`merge` is exported in case you want to reuse the algorithm, but `load` already
relies on it internally.

## Lazy values

Use `Lazy` for values that should be computed from the merged config. A `Lazy`
receives `c`, a read-only proxy for the config where dicts are `Mapping`s and
lists are `Sequence`s. You can use attribute access (`c.trainer.max_steps`),
string keys (`c["trainer"]["max_steps"]`), and list indices
(`c.trainer.stages[0].max_steps`). Lazy values are resolved in-place after
loading (or when you call `resolve_lazy`) and only when they appear inside
nested dict/list structures.

!!! warning
    The proxy references the original config values. Avoid side effects inside
    Lazy functions and don't rely on any specific resolution order.

!!! example "Lazy with a function"
    ```python
    from cfgx import Lazy

    config = {
        "trainer": {"max_steps": 50_000},
        "scheduler": {
            "warmup_steps": 1_000,
            "decay_steps": Lazy(
                lambda c: c.trainer.max_steps - c.scheduler.warmup_steps
            ),
        },
    }
    ```

!!! example "Lazy from an expression"
    ```python
    from cfgx import Lazy

    config = {
        "trainer": {"max_steps": 50_000},
        "warmup_steps": Lazy("c.trainer.max_steps * 0.1"),
    }
    ```

## Formatting and snapshots

Freeze the exact configuration you ran:

!!! example "Format or dump configs"
    ```python
    from pathlib import Path
    from cfgx import dump, format

    print(format(cfg))  # Ruff-formatted when available, pprint fallback otherwise
    dump(cfg, Path("runs/2026-01-12/config_snapshot.py"))
    ```

- `format` returns a nicely formatted string—useful for logging. Use `formatter="pprint"` or `formatter="ruff"` to force a choice.
- `dump` writes the same representation to disk with a short header and `# fmt: off`. Because the file is valid Python, you can load it again with `load`.
  Install `ruff` or `cfgx[format]` to use the Ruff formatter.

## Tips for structuring configs

- Organize by concern: `configs/base.py`, `configs/data/imagenet.py`, `configs/model/resnet.py`.
- Expose helper functions alongside `config` for reusable snippets.
- Prefer `Lazy` for repeated derived values, e.g. a single base learning rate that feeds multiple param groups (`backbone_lr = base_lr * 0.1`).
