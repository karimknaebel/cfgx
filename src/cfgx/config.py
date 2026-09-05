import ast
import math
import os
import re
import runpy
import subprocess
from collections.abc import Callable, Mapping, Sequence
from functools import reduce
from pathlib import Path
from pprint import pformat
from typing import TextIO


class Delete:
    """Sentinel that removes a key from a merged config."""
    pass


class Replace:
    """Replace a value without recursively merging it with the previous value."""

    def __init__(self, value, /):
        self.value = value


class Update:
    """
    Transform the previous value during merge or an override.

    Accepts a callable or an expression using `v` for the previous value.
    If the value is missing, the callable is invoked with no argument.
    If the value is Lazy, the update runs when it resolves.
    """

    def __init__(self, func: Callable | str, /):
        self._expr = func if isinstance(func, str) else None
        if isinstance(func, str):
            code = compile(func, "<update>", "eval")

            def _from_expr(v):
                return eval(code, {}, {"v": v, "math": math})

            self.func = _from_expr
        else:
            self.func = func

    def __repr__(self) -> str:
        if self._expr is not None:
            return f"Update({self._expr!r})"
        return f"Update({self.func!r})"


class Lazy:
    """
    Compute a value from the final config when lazies are resolved.

    Accepts a callable or an expression using `c` to access the config.
    Dictionaries and lists are exposed through read-only proxies.
    """

    def __init__(self, func: Callable | str, /):
        self._expr = func if isinstance(func, str) else None
        if isinstance(func, str):
            code = compile(func, "<lazy>", "eval")

            def _from_expr(c):
                return eval(code, {}, {"c": c, "math": math})

            self.func = _from_expr
        else:
            self.func = func

    def __repr__(self) -> str:
        if self._expr is not None:
            return f"Lazy({self._expr!r})"
        return f"Lazy({self.func!r})"


def load(
    *sources: str | os.PathLike | dict | Sequence,
    overrides: Sequence[str] | None = None,
    resolve_lazy: bool = True,
):
    """
    Expand config files and sequences, then merge dictionaries in order.

    Accepts file paths, dictionaries, or sequences of these. Each file's
    `config` can likewise be a dictionary or a sequence of dictionaries and
    file paths. Sequences expand recursively into one sequence of dictionaries,
    which are merged left to right. Lists inside dictionaries remain data.

    File references are relative to the declaring file; input paths are relative
    to the working directory. Repeated references are expanded each time.
    Each file must define `config`. Referenced files are not merged independently.

    Options are keyword-only. Overrides apply after merging, then Lazy values
    resolve against the result unless `resolve_lazy=False`.

    Mutable source values may be shared with the result and modified during
    loading. Values are not deep-copied; use fresh values for independent loads.
    Source files are not rewritten.
    """

    cfg = reduce(merge, _collect_config_specs(sources, Path.cwd()), {})
    if overrides:
        apply_overrides(cfg, overrides)
    if resolve_lazy:
        _resolve_lazy(cfg)
    return cfg


def _collect_config_specs(
    config: str | os.PathLike | dict | Sequence, base_dir: Path
) -> list[dict]:
    """
    Expand file references and sequences into unmerged dictionaries.
    """
    if isinstance(config, dict):
        return [config]
    if isinstance(config, (str, os.PathLike)):
        path = (base_dir / config).resolve()
        config_module_globs = runpy.run_path(str(path), run_name="__config__")
        if "config" not in config_module_globs:
            raise ValueError(f"Config file {path} must define 'config'")
        return _collect_config_specs(config_module_globs["config"], path.parent)

    return [cfg for item in config for cfg in _collect_config_specs(item, base_dir)]


def dump(
    config: dict,
    fd: TextIO,
    *,
    format: str = "pretty",
    sort_keys: bool = False,
):
    """
    Persist a config dictionary to a Python snapshot (`config = ...`).

    Formatting uses repr(config); default formatting is "pretty". The caller is
    responsible for ensuring it is valid Python that can be reloaded with any
    required imports available. Otherwise formatting can raise or the snapshot
    may fail to load. Use format="raw" for raw repr output.

    sort_keys orders dict keys throughout nested dict/list structures,
    including dict subclasses.
    """

    fd.write(dumps(config, format=format, sort_keys=sort_keys))


def dumps(
    config: dict,
    *,
    format: str = "pretty",
    sort_keys: bool = False,
) -> str:
    """
    Return a Python snapshot string (`config = ...`) for a config dictionary.

    Formatting uses repr(config); default formatting is "pretty". The caller is
    responsible for ensuring it is valid Python that can be reloaded with any
    required imports available. Otherwise formatting can raise or the snapshot
    may fail to load. Use format="raw" for raw repr output.

    sort_keys orders dict keys throughout nested dict/list structures,
    including dict subclasses.
    """
    return _format_snapshot(config, format=format, sort_keys=sort_keys)


def _format_snapshot(
    config: dict,
    *,
    format: str = "pretty",
    sort_keys: bool = False,
) -> str:
    if sort_keys and format in {"raw", "ruff"}:
        config = _sort_keys(config)
    if format == "raw":
        config_str = "config = " + repr(config)
    elif format == "pretty":
        config_str = "config = " + pformat(
            config, width=88, sort_dicts=sort_keys
        )
    elif format == "ruff":
        config_str = _ruff_format("config = " + repr(config))
    else:
        raise ValueError(f"Unknown format: {format}")
    return config_str + "\n"


def format(
    config: dict,
    *,
    format: str = "pretty",
    sort_keys: bool = False,
) -> str:
    """
    Return a string representation based on repr(config).

    Formatting is best-effort; invalid repr output can raise or fail to reload.
    Formatting defaults to "pretty"; use format="raw" for raw repr output.

    sort_keys orders dict keys throughout nested dict/list structures,
    including dict subclasses.
    """
    if format == "raw":
        if sort_keys:
            config = _sort_keys(config)
        return repr(config)
    if format == "pretty":
        return pformat(config, width=88, sort_dicts=sort_keys)
    if format == "ruff":
        if sort_keys:
            config = _sort_keys(config)
        return _ruff_format(repr(config))
    raise ValueError(f"Unknown format: {format}")


def _ruff_format(source: str) -> str:
    try:
        from ruff.__main__ import find_ruff_bin
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Ruff is not installed; install cfgx[format] to use format='ruff'."
        ) from exc

    result = subprocess.run(
        [find_ruff_bin(), "format", "--isolated", "--stdin-filename=config.py", "-"],
        input=source,
        text=True,
        capture_output=True,
        check=True,
        cwd=Path.cwd(),
    )
    return result.stdout.rstrip("\n")


def _sort_keys(value):
    if isinstance(value, dict):
        return {key: _sort_keys(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_sort_keys(item) for item in value]
    return value


def merge(base: dict, override: dict):
    """
    Recursively merge two dictionaries, honoring Delete/Replace/Update sentinels.

    If both sides contain dicts, merge continues down the tree. Delete removes a key
    from the base config, Replace overwrites without further deep merging, Update
    transforms the previous value, and other values simply override.

    Returns a new plain dictionary. Recursively merged branches are copied,
    but other values may be shared with the inputs. Values are not deep-copied,
    and Update callbacks receive the previous value directly.

    Merge is not associative: merging a group of overrides separately
    can change their effect. Apply dictionaries left to right to preserve their
    operations on the accumulated config.
    """
    base = dict(base)
    for k, v in override.items():
        if isinstance(v, dict):
            if k in base and isinstance(base[k], dict):
                base[k] = merge(base[k], v)
            else:
                base[k] = merge({}, v)
        elif isinstance(v, Delete):
            base.pop(k, None)
        elif isinstance(v, Replace):
            base[k] = v.value
        elif isinstance(v, Update):
            if k in base:
                base[k] = _apply_update(base[k], v)
            else:
                base[k] = _apply_missing_update(v)
        else:
            base[k] = v
    return base


def apply_overrides(cfg: dict, overrides: Sequence[str]):
    """
    Apply CLI-style override strings to a config dictionary.

    Supports assignment (`=`), append (`+=`), delete (`!=`), and removal from list
    (`-=`) using dotted/indexed key paths like ``model.layers[0].units``.
    Mutates the config in place and returns it. Changes also affect any shared
    containers targeted by the overrides.
    """

    for override in overrides:
        key, op, value = _split_override(override)
        keys = parse_key_path(key)
        if op == "+=":
            append_to_nested(cfg, keys, infer_type(value))
        elif op == "!=":
            if value:
                raise ValueError(
                    f"Delete overrides must not include a value: {override}"
                )
            delete_nested(cfg, keys)
        elif op == "-=":
            remove_value_from_list(cfg, keys, infer_type(value))
        else:
            parsed_value = infer_type(value)
            if isinstance(parsed_value, Update):
                update_nested(cfg, keys, parsed_value)
            else:
                set_nested(cfg, keys, parsed_value)
    return cfg


def resolve_lazy(cfg: dict):
    """
    Resolve Lazy values reachable through dictionaries and lists in place.

    Each Lazy is evaluated against the config and replaced with its result.
    This also modifies shared dictionaries and lists. Other object types,
    including tuples, are not traversed. Lazy dependency cycles raise an error.

    Container proxies returned by callbacks remain live references to config
    paths. They are not converted to ordinary dictionaries or lists.
    """
    return _resolve_lazy(cfg)


def _resolve_lazy(cfg: dict):
    resolver = _LazyResolver(cfg)
    resolver.resolve_all()
    return cfg


def _get_path(root, path):
    value = root
    for key in path:
        value = value[key]
    return value


def _set_path(root, path, value):
    target = root
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def _format_path(path: tuple):
    out = []
    for key in path:
        if isinstance(key, int):
            out.append(f"[{key}]")
        else:
            if out:
                out.append(".")
            out.append(str(key))
    return "".join(out)


class _LazyResolver:
    def __init__(self, root):
        self.root = root
        self._resolving = []

    def resolve_all(self):
        self._resolve_value((), self.root, resolve_children=True)

    def resolve_at(self, path):
        return self._resolve_value(
            path, _get_path(self.root, path), resolve_children=False
        )

    def _resolve_value(self, path, value, *, resolve_children: bool):
        if isinstance(value, Lazy):
            if path in self._resolving:
                raise ValueError(f"Lazy cycle detected at {_format_path(path)}")
            self._resolving.append(path)
            try:
                value = value.func(_wrap_proxy(self, (), self.root))
            finally:
                self._resolving.pop()
            # Make returned containers available to their children's dependencies.
            _set_path(self.root, path, value)
        if resolve_children and isinstance(value, dict):
            for key in list(value.keys()):
                self._resolve_value(
                    path + (key,),
                    value[key],
                    resolve_children=True,
                )
        elif resolve_children and isinstance(value, list):
            for index in range(len(value)):
                self._resolve_value(
                    path + (index,),
                    value[index],
                    resolve_children=True,
                )
        return value


def _wrap_proxy(resolver: _LazyResolver, path: tuple, value):
    if isinstance(value, dict):
        return _LazyDictProxy(resolver, path)
    if isinstance(value, list):
        return _LazyListProxy(resolver, path)
    return value


class _LazyDictProxy(Mapping):
    def __init__(self, resolver: _LazyResolver, path: tuple):
        self._resolver = resolver
        self._path = path

    def __getitem__(self, key):
        path = self._path + (key,)
        value = self._resolver.resolve_at(path)
        return _wrap_proxy(self._resolver, path, value)

    def __iter__(self):
        container = _get_path(self._resolver.root, self._path)
        return iter(container)

    def __len__(self):
        container = _get_path(self._resolver.root, self._path)
        return len(container)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class _LazyListProxy(Sequence):
    def __init__(self, resolver: _LazyResolver, path: tuple):
        self._resolver = resolver
        self._path = path

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        path = self._path + (index,)
        value = self._resolver.resolve_at(path)
        return _wrap_proxy(self._resolver, path, value)

    def __len__(self):
        container = _get_path(self._resolver.root, self._path)
        return len(container)


def parse_key_path(path: str):
    """Parse 'a.b[0].c' → ['a', 'b', 0, 'c']"""
    tokens = []
    parts = re.split(r"(\[-?\d+\]|\.)", path)
    for part in parts:
        if not part or part == ".":
            continue
        if part.startswith("[") and part.endswith("]"):
            tokens.append(int(part[1:-1]))
        else:
            tokens.append(part)
    return tokens


def _split_override(override: str):
    try:
        idx = override.index("=")
    except ValueError as exc:
        raise ValueError(f"Invalid override: {override}") from exc
    op = "="
    key_end = idx
    if idx > 0 and override[idx - 1] in "+-!":
        op = override[idx - 1 : idx + 1]
        key_end = idx - 1
    return override[:key_end], op, override[idx + 1 :]


def set_nested(d: dict, keys, value):
    parent, last_key = _walk_to_parent(d, keys, create=True)
    _assign_item(parent, last_key, value)


def update_nested(d: dict, keys, updater: Update):
    parent, last_key = _walk_to_parent(d, keys, create=True)
    try:
        current_value = parent[last_key]
    except (KeyError, IndexError):
        next_value = _apply_missing_update(updater)
    else:
        next_value = _apply_update(current_value, updater)
    _assign_item(parent, last_key, next_value)


def append_to_nested(d: dict, keys, value):
    parent, last_key = _walk_to_parent(d, keys, create=True)
    try:
        target = parent[last_key]
    except IndexError:
        if isinstance(parent, list) and isinstance(last_key, int) and last_key < 0:
            raise
        target = []
        _assign_item(parent, last_key, target)
    except KeyError:
        target = []
        _assign_item(parent, last_key, target)
    if not isinstance(target, list):
        raise ValueError("Target is not a list")
    target.append(value)


def delete_nested(d: dict, keys):
    parent, last_key = _walk_to_parent_if_exists(d, keys)
    if parent is None:
        return
    try:
        del parent[last_key]
    except (KeyError, IndexError):
        return


def remove_value_from_list(d: dict, keys, value):
    parent, last_key = _walk_to_parent_if_exists(d, keys)
    if parent is None:
        return
    try:
        target = parent[last_key]
    except (KeyError, IndexError):
        return
    if not isinstance(target, list):
        raise ValueError("Target is not a list")
    if value in target:
        target.remove(value)


def _assign_item(container, key, value):
    if isinstance(container, list) and isinstance(key, int) and key >= 0:
        while len(container) <= key:
            container.append(None)
    container[key] = value


def _apply_update(value, updater: Update):
    if isinstance(value, Lazy):
        def _lifted(c):
            updated = updater.func(value.func(c))
            if isinstance(updated, Lazy):
                return updated.func(c)
            return updated

        return Lazy(_lifted)
    return updater.func(value)


def _apply_missing_update(updater: Update):
    return updater.func()


def _walk_to_parent(d: dict, keys, *, create: bool):
    current = d
    for i, key in enumerate(keys[:-1]):
        try:
            child = current[key]
        except (KeyError, IndexError):
            if not create:
                raise
            child = {} if isinstance(keys[i + 1], str) else []
            _assign_item(current, key, child)
        current = child
    return current, keys[-1]


def _walk_to_parent_if_exists(d: dict, keys):
    current = d
    for key in keys[:-1]:
        try:
            current = current[key]
        except (KeyError, IndexError):
            return None, None
    return current, keys[-1]


def infer_type(val: str):
    if val.startswith("lazy:"):
        return Lazy(val[len("lazy:") :])
    if val.startswith("update:"):
        return Update(val[len("update:") :])
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val
