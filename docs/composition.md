---
icon: lucide/files
---

# Config composition

**Expand file references, concatenate config lists, then merge dictionaries in
order.** A config can be a dictionary or a list of dictionaries and file paths:

```python
config = ["foo.py", {"x": 3}, "bar.py"]
```

The contents of `foo.py` are applied first, then `{"x": 3}`, then the contents
of `bar.py`. Each referenced file can itself declare a config list, which is
expanded at that position. A plain `config = {...}` is the simplest case.

## Expansion and evaluation

`load(*sources, overrides=None, resolve_lazy=True)` accepts file paths,
dictionaries, and lists as positional arguments. Configs expand recursively
according to these rules:

- A dictionary contributes its values and merge operations.
- A file path expands to that file's `config`. Every file must explicitly define
  it; use `config = {}` or `config = []` for an empty config.
- A list expands each entry in order and concatenates the results. Nested config
  lists follow the same rule.

Expansion stops at dictionaries. Lists and strings **inside a dictionary** are
ordinary config data: they are not flattened or treated as file references.
For example, `config = {"files": ["foo.py", "bar.py"]}` stores a list of names.

After expansion, cfgx starts with `{}` and merges each dictionary into the
accumulated result from left to right, using the [merge rules](index.md#merge-semantics).
It then applies CLI-style overrides in order and resolves `Lazy` values against
the final config, unless `resolve_lazy=False`.

File references are relative to the file declaring them. Paths passed directly
to `load` are relative to the working directory. Both strings and `Path` objects
work. Every occurrence is expanded, including repeated references to the same
file.

!!! example "An inline change before a referenced config"
    ```python
    # base.py
    config = {"x": 10}

    # double.py
    from cfgx import Update

    config = {"x": Update("v * 2")}

    # experiment.py
    config = [{"x": 3}, "double.py"]

    # run.py
    from cfgx import Update

    config = ["base.py", "experiment.py", {"x": Update("v + 1")}]
    ```

    The dictionaries are applied in this order:
    `{"x": 10} → {"x": 3} → {"x": Update("v * 2")} → {"x": Update("v + 1")}`.
    `load("run.py")` produces `{"x": 7}`.

The same composition can be passed directly to `load`:

```python
from cfgx import Update, load

cfg = load("base.py", {"x": 3}, "double.py", {"x": Update("v + 1")})
```

`load(a, b, c)` and `load([a, b, c])` have the same behavior. With no sources,
`load()` starts from an empty dictionary. Options such as `overrides` and
`resolve_lazy` must be passed by keyword.

A file boundary groups configs for reuse; it does not isolate their effects.
A config using `Update` may require a value supplied earlier in the sequence
and need not be loadable independently.

This ordering describes dictionary merging. Files execute as Python to discover
their `config`; merge order is not a promise about Python execution order.
Values from referenced files are not injected into another file's Python
variables. Use `Update` to transform a previous value and `Lazy` to read the
final config.

## Why referenced configs are not merged separately

Operations apply to the accumulated config at their position in the sequence:

- `Delete()` removes a key supplied by any earlier dictionary.
- `Replace(value)` replaces the value at that point without recursively merging
  it with earlier values. Later dictionaries can still change it.
- `Update(...)` transforms the previous value at that path. If that value is
  lazy, the transformation is deferred with it; see [Update values](index.md#update-values).
- `Lazy(...)` reads the final config after all dictionaries and overrides, rather
  than a snapshot at the point where it was declared.

Merging a group of configs independently consumes its operations. Its resulting
dictionary can therefore behave differently when used in another merge:

```python
from cfgx import Delete, merge

a = {"x": 1}
b = {"y": 2}
c = {"x": Delete()}

merge(merge(a, b), c)  # {"y": 2}
merge(a, merge(b, c))  # {"x": 1, "y": 2}
```

In the second expression, merging `b` and `c` first loses the deletion: `b` had
no `x` to remove. The resulting dictionary leaves `a`'s `x` intact.

Merge is **not associative**: changing the grouping can change the result.
This also occurs without sentinels when a value changes between a dictionary
and a scalar. For example, applying `{"x": {"a": 1}}`, `{"x": 0}`, and
`{"x": {"b": 2}}` in order gives `{"x": {"b": 2}}`. Merging the last two
first would allow `"a"` to survive in the final dictionary.

`load([a_path, b_path])` expands both files into the same sequence. It is not
generally equivalent to `merge(load(a_path), load(b_path))`, even with lazy
resolution disabled. A saved snapshot likewise preserves the resulting values,
not the original sequence of operations.

## Ownership and mutation

Treat mutable source objects as consumed by `load`: do not assume they remain
unchanged or that reusing them produces independent results. This does not mean
every object will be modified. cfgx does not rewrite source files.

Config values are ordinary Python objects. cfgx creates dictionaries while
merging, but does not deep-copy their contents or guarantee isolation from the
supplied configs. Lists, objects passed through `Replace`, and other mutable
values can remain shared.

The handling of ordinary values is determined by the operation:

| Value | During merging | During lazy resolution |
| --- | --- | --- |
| Dictionary | Recursively merge into result dictionaries. Dictionary identity is not preserved by merging. | Visit values and replace resolved entries in place. |
| List | Replace the previous value with the supplied list, without copying it. | Visit items and replace resolved entries in place. |
| Other Python object | Use the supplied object without copying it. | Leave it untouched; do not traverse its attributes or contents. |

The dictionary and list rules include subclasses. Other mappings and sequences,
including tuples, are not automatically traversed by lazy resolution.
`Delete`, `Replace`, `Update`, and `Lazy` are the explicit operations described
above. `Replace` bypasses recursive merging, but its value still participates
in lazy resolution according to its type.

`merge` copies the base dictionary at each recursive merge; entries it does not
change can still reference existing objects. `apply_overrides` and
`resolve_lazy` modify the supplied result in place. `load` performs those same
operations after merging, so a new result dictionary does not imply that every
object reachable from it is independent of the inputs.

`Update` receives the previous value directly. The `Lazy` proxy provides read-only
access to dictionaries and lists; arbitrary objects reached through it retain
their own mutation behavior. Prefer callbacks that return values without
modifying their inputs.

Shared values can also be modified by cfgx itself: overrides mutate their
targets, and lazy resolution replaces `Lazy` entries in dictionaries and lists.
This can affect the source even when every callback has no side effects:

```python
from cfgx import load

source = {"tags": ["base"]}
cfg = load(source, overrides=["tags+=extra"])

assert source["tags"] == ["base", "extra"]
assert cfg["tags"] is source["tags"]
```

Likewise, resolving a `Lazy` inside a shared list replaces that entry in the
source list. Reusing the source can therefore reuse an already-computed value.

Callers decide which objects may be shared. When independent loads are needed,
construct fresh mutable values or explicitly copy the objects whose semantics
allow it. Loading a file recreates its dictionary and list literals, but values
imported from other modules may still be shared across loads.

For repeated runs with different overrides, fresh values can be supplied using
an ordinary Python function:

```python
from cfgx import Lazy, load

def make_config():
    return {"steps": 10, "values": [Lazy("c.steps * 2")]}

first = load(make_config(), overrides=["steps=20"])
second = load(make_config(), overrides=["steps=30"])

assert first["values"] == [40]
assert second["values"] == [60]
```

Passing the same already-created dictionary to both calls would reuse its
mutable contents. A new outer dictionary alone is insufficient if it still
contains shared lists or other mutable objects.

## Repeated references are applied again

Suppose `left.py` and `right.py` both start their config lists with `"base.py"`,
and `run.py` declares `config = ["left.py", "right.py"]`. The dictionaries from
these files are merged in this order:

```text
base.py → left.py's own dictionaries → base.py → right.py's own dictionaries
```

If `base.py` sets `x = 1`, `left.py` then sets `x = 2`, and `right.py` leaves `x`
alone after including the base, the result is `x = 1`. The second occurrence of
`base.py` resets it. Repeated configs containing updates also apply those
updates again.

When several configs should share one application of a base, list the base once
in the composing file and remove its references from the others:

```python
config = ["base.py", "left.py", "right.py"]
```
