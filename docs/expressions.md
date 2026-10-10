# Expressions

## Select a value

```python
from cfgx import final, previous, value

config = {
    "lr": value * 0.1,
    "backbone_lr": final.lr * 0.1,
    "inherited_width": previous.model.width,
}
```

`final` includes all layers and CLI overrides. `previous` includes definitions
before the expression's originating layer. `value` is `previous(0)`: the inherited
value at the expression's output location. Earlier definitions keep their own
origins, and their `final` references still see the complete composition.
Inside a nested `compose`, `previous` also includes preceding local contributions
at that location. Surrounding sibling fields retain their enclosing layer's
previous view; see [the mental model](model.md#local-layers-and-inherited-values).

Attributes select dictionary keys. Subscripts support arbitrary hashable keys
and sequence indices; `final[("a", "b")]` selects a tuple key, not two path parts.
Use subscripts for keys that are not Python identifiers or conflict with Python
special attributes. Sequences support negative indices and slices.

Both roots accept a parent count:

| Expression | Starting location |
| --- | --- |
| `final` or `final(None)` | Config root |
| `final(0)` | Current output location |
| `final(1)` | Immediately containing container |
| `final(2)` | Two path levels above the output location |
| `previous(n)` | Same location, using earlier definitions |

Dictionary keys and sequence indices each count as one level. Expression
operators and maps do not add levels. Going above the root errors.

```python
block = {"width": 64, "hidden": final(1).width * 2}
config = {"encoder": block, "decoder": block}, {"decoder": {"width": 128}}
```

Here the hidden widths are 128 and 256. Reusing a declaration binds it separately
at each output location.

## Transform and combine

Arithmetic and comparisons build expressions. Use `.map` for ordinary Python
functions:

```python
config = {
    "warmup_steps": (final.steps * 0.025).map(int),
    "tags": value.default([]).map(lambda x: [*x, "finetune"]),
    "scaled": final.lr.map("x * 2"),
}
```

The string form is a Python expression with `x` as its resolved input. It also
provides the expression API, `math`, and normal builtins. This is trusted Python,
just like the code in a config file.

For conditional or multi-value computations:

```python
from cfgx import computed, delete, final

config = {
    "checkpoint": computed(
        lambda get: get(final.checkpoint_path) if get(final.resume) else delete
    ),
    "effective_batch": computed(
        lambda get: int(get(final.batch_size)) * get(final.accumulation)
    ),
}
```

`get` takes any expression and returns its fully resolved value, with supported
containers structurally copied. Only reads actually executed create dependencies.
`get(final.lr.map("x * 2"))` is valid. A callback may return nested expressions,
`replace`, `delete`, `include`, or `compose`. Returned values bind at their output
locations within the same originating layer; explicit compositions introduce
ordered local layers. Includes returned by a callback resolve paths relative to
the expression's declaring file. Returned strings and tuples remain data.

Unresolved expressions have no Python truth value and cannot be iterated.
Use `get` inside `computed` for `if`, `and`, `or`, comprehensions, and functions
that require ordinary values. Expressions are declarations, not container proxies.

## Missing values

```python
final.optimizer.default({"lr": 1e-3}).lr.default(1e-4)
final.optional.default(final.fallback)
computed(lambda get: delete).default(None)
```

Defaults apply to direct absence, including deletion. `None`, `False`, and zero
are present values. Fallbacks are evaluated only when needed and do not merge
into an existing dictionary. Defaults do not suppress callback exceptions,
missing dependencies inside another definition, or cycles. An unresolved missing
reference without a default raises `MissingValueError`.

## Read structure precisely

- `.keys()` returns a list of mapping keys, after computed deletions.
- `.len()` returns length without materializing supported sequence elements or
  dictionary child contents. Dictionary membership must still be determined.
- `.exists()` checks presence without materializing a present subtree.

```python
from cfgx import computed, value

config = {
    "datasets": computed(
        lambda get: {
            name: {"image_augmentation": []}
            for name in get(value.keys())
        }
    ),
}
```

This returns a partial patch. It preserves other dataset fields without reading
their values. In contrast, `value.map(fn)` materializes the whole inherited
subtree before calling `fn`.

If a callback may delete an entry, it may need to execute before that entry's
presence is known. A callback producing an entire dictionary must execute to
reveal its keys. Structural reads cannot skip Python statements inside callbacks.
`expr.map(len)` materializes the input; `expr.len()` requests only its length.

Names remain available as keys: `final.map` selects `"map"`, `final.map(fn)`
transforms the root, and `final.map.map(fn)` transforms the `"map"` value. The
same distinction applies to `default`, `keys`, `len`, and `exists`.

## Caching and cycles

Within one load, each computation runs at most once per expression binding,
output location, and originating layer. Structure and full-value reads share
producer results. A fresh load computes fresh results. Opaque-object mutations
and external side effects remain ordinary Python behavior; evaluation order is
not a public guarantee.

Cycles raise `ConfigError` with a path, layer, and operation trace. Reading your
own full final value, including through an ancestor, creates a cycle. Earlier
values can also cycle when they contain final references back to the calculation.

```python
config = {"x": final.x * 2}  # cycle
```

A keys-only read can succeed where a full read would cycle, but computed deletion
can itself make membership cyclic. The [reweighting example](https://github.com/karimknaebel/cfgx/tree/main/examples/training)
reads names, weights, and label types explicitly, then patches just the weights.
This leaves paths, augmentations, and other dataset settings intact.

## Opaque objects

Explicit selectors can read through an opaque object's `__getitem__` interface.
The opaque object is resolved as one dependency, then ordinary Python access
applies. Attribute syntax remains key sugar, not object attribute lookup.
`.keys()` and `.len()` use its normal methods. cfgx does not discover declarations
hidden inside opaque objects. Build custom objects from resolved inputs when
needed, for example `computed(lambda get: MyOptions(get(final.options)))`.
