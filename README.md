# cfgx

[![PyPI version](https://img.shields.io/pypi/v/cfgx.svg)](https://pypi.org/project/cfgx/)

Python configs with ordered composition, computed values, and CLI overrides.
Loading produces an ordinary dictionary. Configuration logic stays in Python.

[Documentation](https://karimknaebel.github.io/cfgx/)

```sh
pip install cfgx
```

Define a base:

```python
# base.py
from cfgx import final

config = {
    "steps": 96_000,
    "lr": 3e-4,
    "backbone_lr": final.lr * 0.1,
    "cooldown_steps": final.steps // 10,
}
```

Compose a variant with a tuple of sources:

```python
# finetune.py
from cfgx import value

config = "base.py", {"steps": 48_000, "lr": value * 0.5}
```

Load it with optional overrides:

```python
from cfgx import load

cfg = load("finetune.py", overrides=["lr=expr:value * 0.1"])
assert cfg["steps"] == 48_000
assert cfg["backbone_lr"] == cfg["lr"] * 0.1
```

Overrides contribute ordinary layers: `optimizer.lr=1e-4` adds
`{"optimizer": {"lr": replace(1e-4)}}`, and `optimizer.decay!=` adds
`{"optimizer": {"decay": delete}}`. Use `expr:{...}` for a complete layer.

Dictionaries merge recursively, including computed dictionaries. Lists and tuples
replace earlier sequences. Use `replace(x)` to replace a dictionary and `delete`
to remove a dictionary entry.

Use `include("model.py")` to contribute a file at a nested location, and
`compose("defaults.py", {...})` for ordered contributions there. Root source
strings and tuples are shorthand for these operations; strings and tuples in
config values remain data. The [mental model](docs/model.md) explains these
contexts and how nested layers read inherited values.

- `final.lr` reads the complete config, including later files and overrides.
- `value` reads the inherited value at the current location.
- `previous.lr` reads earlier definitions at another path.
- `final(1).lr` reads a sibling; `previous(1).lr` reads its inherited definition.
- `expr.map(fn)` transforms a resolved value. `expr.map("x * 2")` is shorthand.
- `computed(lambda get: ...)` supports dynamic, conditional reads with `get(expr)`.
- `.default(...)`, `.keys()`, `.len()`, and `.exists()` compose with other expressions.

```python
from cfgx import computed, delete, final

config = {
    "resume": False,
    "checkpoint": computed(
        lambda get: get(final.checkpoint_path) if get(final.resume) else delete
    ),
}
```

Exact built-in dictionaries, lists, and tuples are rebuilt independently for each
output location. Other objects are opaque and retain identity. Avoid side effects
when working with opaque values or unknown types.

See [composition](docs/composition.md), [expressions](docs/expressions.md),
[overrides](docs/cli.md), and the [resolution contract](docs/design.md).
[Training example](examples/training/README.md) demonstrates a compact setup
with model and dataset composition, learning-rate scaling, computed checkpoints,
and dataset transformations.

Snapshots and formatting remain available through `dump`, `dumps`, and `format`.
A snapshot preserves resolved values, not their formulas.
