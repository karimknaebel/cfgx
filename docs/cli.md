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
| `options['literal.key']=True` | Select a literal key containing a dot |
| `lr=expr:value * 0.1` | Transform the target's inherited value |
| `backbone_lr=expr:final.lr * 0.1` | Contribute a final-value expression |

Each override is a separate ordered layer. Assignment replaces only its target,
while its dictionary ancestors merge normally. `foo.bar=baz` is shorthand for
`{"foo": {"bar": replace("baz")}}`; `foo.bar!=` is shorthand for
`{"foo": {"bar": delete}}`. These contributions use the same resolver as config
files, with no separate path-editing stage.

`final` includes all overrides; `value` reads the target before the current
override. An inherited computation can read an overridden key without first
finishing its own container:

```python
from cfgx import computed, final, load

assert load(
    computed(lambda get: {"answer": get(final.x) * 2}),
    overrides=["x=3"],
) == {"answer": 6, "x": 3}
```

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

## Whole-layer expressions

An override starting with `expr:` contributes a complete layer. It must produce
a plain dictionary or a cfgx expression that produces a plain dictionary, just
like a single contribution in a config file:

```sh
cfgx render base.py -o \
  'expr:{"optimizer": {"lr": value * 0.1}, "lookup": {0: "first"}}'
```

Whole-layer expressions follow ordinary composition rules, including dictionary
merging. Use `replace` explicitly when a dictionary should be replaced. They
support arbitrary dictionary keys and can be mixed with assignment and deletion
shorthand. Each argument contributes one layer; definitions within the same layer
share the same `previous` view.

## Path rules

Shorthand paths select string dictionary keys using dots or quoted subscripts.
Use a whole-layer expression for non-string keys. Numeric indices such as
`layers[0].width=128` are rejected. Expressions can still read sequence indices
and slices; replace or transform the sequence to change it:

```sh
cfgx render base.py -o \
  'layers=expr:value.map(lambda x: [{**x[0], "width": 128}, *x[1:]])'
```

`map` reads the inherited sequence completely before transforming it, so its
dependencies include all inherited elements.

Because shorthand produces ordinary dictionary contributions, missing ancestors
are created. An inherited scalar, sequence, or opaque object at an ancestor is
replaced by the contributed dictionary. Opaque objects are not mutated.

Deletion removes only its own entry. Declared parent dictionaries remain present,
even when empty or newly introduced. All deletion spellings follow this rule:

```python
assert load(overrides=["missing.deep!="]) == {"missing": {}}
assert load(overrides=["missing.deep=expr:delete"]) == {"missing": {}}
```

An expression returning `delete` has the same effect. Deletion at the root or
inside a list/tuple remains invalid.

Overrides, including expressions, resolve as part of `load`. There is no separate
in-place override or lazy-resolution API. To load another variant, reuse the
sources. Applying overrides to an already resolved snapshot cannot recover its
original computed relationships.
