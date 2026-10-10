# Mental model

cfgx composes definitions, then resolves them into an ordinary dictionary. A
definition can be a literal value, an expression, a merge instruction, a
composition, or an include. These are conceptual categories, not types you need
to instantiate directly.

| Kind | Examples | Meaning |
| --- | --- | --- |
| Values | `1`, `"foo"`, dictionaries, lists, tuples, opaque objects | Config data |
| Expressions | `final.lr`, `value * 2`, `computed(...)` | Deferred values or contributions |
| Merge instructions | `replace(x)`, `delete` | Control inheritance at a location |
| Composition | `compose(a, b, ...)` | Ordered config contributions at a location |
| Inclusion | `include("foo.py")` | Contribute a file's unresolved definitions |

## Source positions and value positions

Implicit behavior depends on the position, not its nesting depth.

**Source positions** are a file's exported `config`, arguments to `load(...)`,
and arguments to `compose(...)`.

| Source | Interpretation |
| --- | --- |
| Plain dictionary | One contribution |
| `compose(...)` | Explicit ordered composition |
| `include(path)` | Explicit file inclusion |
| String or path object | Implicit `include(path)` |
| Tuple of sources | Implicit `compose(*sources)` |
| Expression | A contribution that must produce a plain dictionary |
| `replace(...)` | A dictionary contribution that discards inherited contents |

Every contribution in a composition must produce a plain dictionary. Lists and
scalars are invalid sources; `delete` is invalid at the root of a composition.
An empty composition adds no layers in a source position. As a value, it
produces an empty dictionary; loading no sources also produces `{}`.

These declarations are equivalent:

```python
from cfgx import compose, include

config = "base.py", {"lr": 1e-4}
config = compose("base.py", {"lr": 1e-4})
config = compose(include("base.py"), {"lr": 1e-4})
```

A single dictionary needs no wrapper: `config = {...}` is equivalent to
`config = compose({...})`.

**Value positions** are dictionary values and list/tuple elements. Arguments to
`replace` and expression operands/results also use value semantics: for example,
`replace("model.py")` contains a string, not an implicit include. Strings,
path objects, and tuples here are data. Explicit `include` and `compose`
declarations can contribute dictionaries at these locations, including inside
sequences:

```python
from cfgx import compose, include

config = (
    "base.py",
    {
        "filename": "model.py",
        "pair": ("small.py", "large.py"),
        "model": include("model.py"),
        "decoder": compose("decoder_defaults.py", {"width": 768}),
        "ensemble": [include("small.py"), include("large.py")],
    },
)
```

`"decoder_defaults.py"` is a source because it is an argument to `compose`.
`"model.py"` under `filename` and the strings under `pair` are ordinary values.
Only exact built-in dictionaries, lists, and tuples are traversed. Declarations
inside custom containers or other opaque objects are left untouched.

## Inclusion inserts definitions

Think of `include` as inserting a file's definitions at its location. It does
not load a resolved snapshot. If `model.py` defines several contributions, they
compose in order at the insertion location. Later layers and overrides can
still change the values its expressions read.

```python
# model.py
from cfgx import final

config = {"width": 384, "output_width": final(1).width}
```

```python
# experiment.py
from cfgx import include

config = (
    {"model": include("model.py")},
    {"model": {"width": 768}},
)
```

Both widths become `768`. Relative references bind at the insertion location;
`final` without a parent argument always refers to the complete config root.
Including the same file at another location creates independent bindings.

Paths are relative to the declaring config file. Sources supplied directly to
`load`, including overrides, use the working directory. An included file's own
relative paths use that file's directory. Include cycles report the file chain.

## Local layers and inherited values

`final` sees the complete config. Each contribution has a `previous` view of the
definitions before it; `value` selects that view at the expression's own path.

Inside a nested composition, the first contribution sees the config before the
enclosing layer. Later contributions see that same config with earlier local
contributions applied at the composition's location. Surrounding dictionary
fields are still in the enclosing layer: they apply once and do not become
visible through `previous` merely because they appear earlier in the dictionary.

```python
from cfgx import compose, load, previous, value

cfg = load(
    {"model": {"width": 3}, "batch_size": 8},
    {
        "model": compose(
            {"width": value * 2},
            {"width": value + 1, "batch_before": previous.batch_size},
        ),
        "batch_size": value * 2,
    },
)
assert cfg == {"model": {"width": 7, "batch_before": 8}, "batch_size": 16}
```

Sibling compositions have independent local histories. Reordering their
dictionary entries does not change those views. Earlier expressions retain
their original `previous` view and still read the actual final config through
`final`.

A local previous view does not invent values for surrounding sequence elements.
If a composition is inside a newly introduced sequence, preceding local layers
make its own element readable; other elements absent from the earlier config
remain missing. Reading the whole previous sequence can therefore raise
`MissingValueError`, while selecting the local element succeeds.

## Expressions can return declarations

Expressions can produce values, nested expressions, `replace`, `delete`,
`include`, and `compose`. The resulting declarations follow the same rules as
declarations written directly at that location:

```python
from cfgx import computed, final, include

config = {
    "model_file": "small.py",
    "model": computed(lambda get: include(get(final.model_file))),
}
```

Returned includes retain the expression's declaring-file directory. A returned
composition introduces ordered local contributions; returning an ordinary
dictionary does not introduce additional layers.

Implicit source conversions do not apply to expression results. A computed
`"model.py"` remains a string and a computed `("a.py", "b.py")` remains a tuple.
Both are valid as values and invalid as the root result of a contribution. Use
explicit `include` or `compose` to return further sources.

## Inclusion and composition do not choose the merge behavior

Dictionaries merge by default, including dictionaries contributed by an include
or a composition. `replace` explicitly discards inherited contents:

```python
from cfgx import compose, include, replace

config = {
    "model": replace(include("model.py")),
    "decoder": compose("decoder_defaults.py", replace({"width": 768})),
}
```

`replace` can also appear as a root source. Later contributions can patch its
result. Expressions inside a replacement can still read inherited values.

CLI assignment always replaces its target, so `model=include:foo.py` contributes
`{"model": replace(include("foo.py"))}`. To merge, use the whole-layer form
`expr:{"model": include("foo.py")}`. See [CLI and overrides](cli.md).
