# CLI and overrides

## Load and render

```sh
cfgx render base.py finetune.py -o 'steps=48000' 'lr=expr:value * 0.1'
cfgx dump base.py --format pretty --sort-keys > snapshot.py
```

`print` is an alias for `render`; `freeze` is an alias for `dump`. Both commands
resolve the complete config. `--format` accepts `pretty`, `raw`, or `ruff`.
`dump` emits a Python `config = ...` assignment, while `render` prints the value.

The same overrides work through Python:

```python
from cfgx import load

cfg = load("base.py", overrides=["steps=48000", "lr=expr:value * 0.1"])
```

## Assignment and deletion

| Override | Meaning |
| --- | --- |
| `optimizer.lr=1e-4` | Replace the target value |
| `optimizer={'lr': 1e-4}` | Replace the whole optimizer dictionary |
| `optimizer.decay!=` | Delete a dictionary entry |
| `layers[0].width=128` | Patch an existing sequence element |
| `options['literal.key']=True` | Select a literal key containing a dot |
| `lr=expr:value * 0.1` | Transform the target's inherited value |
| `backbone_lr=expr:final.lr * 0.1` | Contribute a final-value expression |

Each override is a separate ordered layer. Assignment replaces only its target,
leaving surrounding dictionary fields or sequence elements intact. For dictionary
paths, `foo.bar=baz` behaves like `{"foo": {"bar": replace("baz")}}`.

Values use Python literal parsing, with an unquoted-string fallback. `expr:`
evaluates a Python expression with `final`, `previous`, `value`, `computed`,
`replace`, `delete`, `math`, and ordinary builtins. `x=expr:delete` also deletes.
Quote overrides at the shell when they contain spaces or shell operators.

```sh
cfgx render base.py -o 'tags=expr:value.default([]).map("x + [\"debug\"]")'
```

There are no separate append/remove operators. Express sequence transformations
with `value.map(...)`, using `x` in string maps. Assigning a literal dictionary
replaces it; an expression can explicitly combine it with `value` if needed.

## Path rules

Missing dictionary ancestors are created. Existing lists and tuples can be
indexed, including with negative indices, and are rebuilt around the changed
element. Indices must already exist; assignment does not extend sequences or
infer new lists. Missing ancestors are dictionaries, including for integer keys.

Deleting a missing dictionary path is a no-op. Deleting a list/tuple element
errors; replace or transform the sequence to change its membership. Writes
through scalars, custom containers, or other opaque objects are rejected.
Replace the opaque object as a whole instead.

Overrides, including expressions, resolve as part of `load`. There is no separate
in-place override or lazy-resolution API. To load another variant, reuse the
sources. Applying overrides to an already resolved snapshot cannot recover its
original computed relationships.
