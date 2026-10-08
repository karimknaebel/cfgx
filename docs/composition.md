# Config composition

## Sources and layers

A source is a path, a plain dictionary, an expression producing a dictionary, or
a tuple of sources. Every file defines `config` with the same syntax.

```python
config = {"lr": 3e-4}
```

To add inheritance, introduce a tuple:

```python
config = "base.py", {"lr": 1e-4}
```

For larger compositions:

```python
config = (
    "base.py",
    "data.py",
    {"steps": 48_000},
    "schedule.py",
)
```

`load` accepts the same sources as positional arguments or nested tuples. Empty
composition is `()` or `{}`. Lists are not accepted as source collections.
Strings, lists, and tuples inside a dictionary are data, not source directives.
Computed contributions produce data, not further includes.

Each contribution is a layer. Files and nested tuples simply expand in place;
referenced files are not resolved or merged independently. Relative includes use
the including file's directory. Repeated includes apply repeatedly. Include
cycles report their path chain. The root must be a plain dictionary.

## Merge behavior

- Plain dictionaries merge recursively.
- Lists, tuples, scalars, and opaque objects replace the previous value.
- `replace(x)` replaces inherited contents at its location.
- `delete` removes a dictionary entry. It is not valid at the root or as a
  list/tuple element; sequences never implicitly shrink when an element resolves.

```python
from cfgx import delete, load, replace

cfg = load(
    {"optimizer": {"lr": 1e-3, "decay": 0.01}, "schedule": {"old": 1}},
    {"optimizer": {"decay": delete}, "schedule": replace({"type": "cosine"})},
)
assert cfg == {"optimizer": {"lr": 1e-3}, "schedule": {"type": "cosine"}}
```

Later layers can patch a replacement. An intervening scalar, sequence, opaque
object, or deletion breaks dictionary inheritance from earlier layers.
`replace(delete)` still deletes: replacement does not quote merge instructions.

Deletion affects only its own entry. A declared dictionary remains present even
when all its children are deleted: `load({"a": {"b": delete}})` produces
`{"a": {}}`. Literal and computed deletions follow the same rule.

Computed dictionaries participate in the same merging:

```python
from cfgx import computed, load

assert load(
    {"model": computed(lambda get: {"width": 64, "depth": 8})},
    {"model": {"width": 128}},
) == {"model": {"width": 128, "depth": 8}}
```

Dictionary key order follows first contribution within the current mapping;
keys that resolve to deletion are omitted. Reintroducing a deleted key retains
its original ordering position. Replacing the mapping starts a new order.
This keeps ordering from forcing evaluation of overwritten definitions.

## Earlier definitions and final values

`value` reads definitions before the entire layer containing the expression.
Dictionary entry order does not create additional layers. `previous` offers the
same view starting at the root or a chosen ancestor.

Earlier definitions retain their references to the final config:

```python
from cfgx import final, load, value

assert load(
    {"steps": 100, "cooldown": final.steps // 10},
    {"steps": 200, "cooldown": value * 2},
) == {"steps": 200, "cooldown": 40}
```

The earlier cooldown definition first reads the final `steps=200`. It is not a
snapshot taken when the first layer was declared.

## Ownership and mutation

Only exact built-in `dict`, `list`, and `tuple` belong to the supported config
structure. cfgx rebuilds these containers, resolves their children, and treats
source aliases as independent occurrences. Dictionaries and lists are distinct
from their sources and from other output locations, including `final` references.
Tuples retain type and contents but need not have distinct identity.

```python
from cfgx import final, load

shared = {"items": []}
cfg = load({"a": shared, "b": shared, "c": final.a})
cfg["a"]["items"].append(1)
assert shared == cfg["b"] == cfg["c"] == {"items": []}
```

This is structural copying, not `deepcopy`. Sets, custom mappings, container
subclasses, functions, resources, and other objects remain opaque and retain
identity. A copied list can still contain shared opaque objects. Raw cycles in
supported containers are rejected.

Callback inputs and `get` results are structural copies. A producer's returned
structure is captured before its children resolve. Mutating an opaque object can
still change sources or other locations. Prefer functions without side effects,
especially when you do not know a value's exact type. A custom list subclass is
opaque even if it looks like an ordinary list.

## Reusing sources

`load` does not mutate supported source containers. Reuse a source tuple to load
several variants; each load resolves its expressions afresh. Referenced files
execute again on every inclusion. Objects imported from other modules may remain
shared because ordinary Python import semantics still apply.

The output is a snapshot. Passing it back to `load` is valid, but formulas that
were already resolved cannot respond to new overrides. Retain the original
sources when you need recomputation.
