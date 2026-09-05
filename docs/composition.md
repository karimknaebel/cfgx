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

cfgx does not deep-copy config values. The result can share lists and other
mutable objects with its sources, so loading or modifying a config can also
change those sources. Treat mutable inputs as consumed: reuse them only when
you intend to share their state.

### Reusing configs

For independent variants, create fresh values for each load. An ordinary Python
factory works well for a shared base:

```python
from cfgx import Lazy, load

def make_config():
    return {"steps": 10, "values": [Lazy("c.steps * 2")]}

first = load(make_config(), overrides=["steps=20"])
second = load(make_config(), overrides=["steps=30"])

assert first["values"] == [40]
assert second["values"] == [60]
```

Fresh inline dictionaries work the same way. Loading the same file again
re-executes it and recreates its dictionary and list literals. Objects imported
from other modules can still be shared; a config file can call an imported
factory to create fresh values instead.

A new outer dictionary alone is insufficient if it contains reused lists or
other mutable objects. `base.copy()` and `merge(base, changes)` can both leave
nested values shared. For several variants, reuse the base's file path or call
its factory each time.

You can also copy values explicitly when you know how they should be copied.
Arbitrary Python objects may depend on their identity or not support copying,
so cfgx leaves that choice to you.

### What is copied or modified

- `merge` creates a new plain dictionary for each branch it recursively merges.
  Untouched base values remain shared. Lists, `Replace` values, and callback
  results are used as-is.
- `apply_overrides` modifies the supplied config in place. For example, appending
  to a shared list changes that list for every config that uses it.
- `resolve_lazy` replaces lazies in dictionaries and lists in place, including
  containers returned by callbacks. Shared containers are modified too.
- `Update` receives the previous value directly. Prefer returning new values
  over mutating the input. Likewise, avoid side effects in `Lazy` callbacks.

These rules include dict/list subclasses; recursively merged dictionaries
become plain dicts. Other objects, including tuples and other mapping types, are
kept as-is and are not traversed by lazy resolution. They can still be modified
by callbacks or explicit override paths.

Because merge copies dictionary branches, two keys that referred to the same
dictionary may end up with separate dictionaries. Shared lists remain shared.

### Reusing resolved values

Resolving a lazy replaces it with its result. If it lives in a shared container,
subsequent loads see the computed value:

```python
from cfgx import Lazy, load

source = {"steps": 10, "values": [Lazy("c.steps * 2")]}
first = load(source, overrides=["steps=20"])
second = load(source, overrides=["steps=30"])

assert first["values"] is second["values"] is source["values"]
assert second["values"] == [40]
```

The list no longer contains a lazy to recompute. Use fresh source values when
changing dependencies for another run. `resolve_lazy=False` postpones resolution
but does not copy or isolate the config's contents.

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
