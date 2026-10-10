---
icon: lucide/settings
---

# cfgx

Keep configuration logic in Python files, compose them in order, and load an
ordinary dictionary with computed values resolved against the complete config.

## Define and compose

```python
# base.py
from cfgx import final

config = {
    "steps": 96_000,
    "cooldown_steps": final.steps // 10,
    "lr": 3e-4,
    "backbone_lr": final.lr * 0.1,
    "optimizer": {"type": "AdamW", "weight_decay": 0.01},
}
```

```python
# finetune.py
from cfgx import delete, value

config = (
    "base.py",
    {
        "steps": 48_000,
        "lr": value * 0.5,
        "optimizer": {"weight_decay": delete},
    },
)
```

A dictionary alone is shorthand for a one-element tuple. Tuples compose sources;
inside the config, tuples and lists are ordinary sequence data. Dictionaries
merge recursively, and file references are relative to the declaring file.
Use `include("model.py")` to insert a file's definitions at a nested location,
or `compose("defaults.py", {...})` to compose sources there. See the
[mental model](model.md) for what accepts sources, what accepts values, and
which forms are implicit.

## Load and override

```python
from cfgx import load

cfg = load("finetune.py", overrides=["steps=24_000"])
assert cfg["cooldown_steps"] == 2_400
assert cfg["backbone_lr"] == cfg["lr"] * 0.1
```

All sources and overrides participate in one composition. `final` reads the
complete composition; `value` reads earlier definitions at its own location.
The result contains ordinary Python values and has no ongoing reactive behavior.

```sh
cfgx render finetune.py -o 'steps=24000' 'lr=expr:value * 0.1'
cfgx dump finetune.py --format pretty > snapshot.py
```

## Calculate with Python

Use arithmetic for simple expressions, `.map` for a transformation of one value,
and `computed` for calculations that read several values or choose dependencies
conditionally:

```python
from cfgx import computed, final, value

config = {
    "checkpointing": {
        "keep_steps": value.default(()).map(
            lambda x: [*x, final.steps - final.cooldown_steps]
        ),
    },
    "effective_batch": computed(
        lambda get: int(get(final.batch_size)) * get(final.accumulation)
    ),
}
```

Callbacks receive fully resolved values. Returned dictionaries, lists, and tuples
may contain new expressions. See [Expressions](expressions.md) for precise reads,
relative references, missing values, and cycles.

## Ownership

cfgx structurally copies exact built-in dictionaries, lists, and tuples. Reusing a
source container or referring to it through `final` does not share mutable output
containers. Other Python objects, including container subclasses, are opaque and
retain identity. Avoid side effects when their behavior or type is uncertain.

## Snapshots

```python
from cfgx import dump, format

print(format(cfg))
with open("snapshot.py", "w") as fd:
    dump(cfg, fd)
```

`format`, `dump`, and `dumps` accept `format="pretty"` (the default), `"raw"`, or
`"ruff"`, and `sort_keys=True`. Ruff formatting requires `cfgx[format]`.
Snapshots use Python representations; custom objects may require imports or may
not have a reloadable representation. A snapshot stores values, not expressions.
