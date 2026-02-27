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
    """Sentinel that forces a value to replace a mapping during merge."""

    def __init__(self, value, /):
        self.value = value


class Update:
    """Sentinel that transforms an existing value during merge."""

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
    """Callable wrapper that defers computation until config resolution."""

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
    path: os.PathLike | Sequence[os.PathLike],
    overrides: Sequence[str] | None = None,
    resolve_lazy: bool = True,
):
    """
    Load config modules from a path or sequence of paths, apply overrides, and merge the results.

    Parent configs (via `parents`) are resolved first, then later paths override
    earlier ones. `config` must be a dictionary.
    """

    paths = [path] if isinstance(path, (str, os.PathLike)) else list(path)
    configs = [cfg for p in paths for cfg in _collect_config_specs(Path(p))]
    cfg = reduce(merge, configs, {})
    if overrides:
        apply_overrides(cfg, overrides)
    if resolve_lazy:
        _resolve_lazy(cfg)
    return cfg


def _collect_config_specs(path: os.PathLike) -> list[dict]:
    """
    Return the flattened inheritance chain for the config at `path`. Ordered from the farthest parent first.
    """
    path = Path(path).resolve()
    config_module_globs = runpy.run_path(str(path), run_name="__config__")

    config = config_module_globs.get("config", {})

    parents = config_module_globs.get("parents", None)
    if isinstance(parents, str):
        parents = [parents]

    return [
        parent_cfg_specs
        for parent in parents or []
        for parent_cfg_specs in _collect_config_specs(path.parent / Path(parent))
    ] + [config]


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
    transforms the previous value, and other values simply override. Returns a new
    dictionary without mutating the inputs.
    """
    base = base.copy()
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
    (`-=`) using dotted/indexed key paths like ``model.layers[0].units``. Mutates
    the dictionary in place and returns it.
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
    Resolve Lazy values in a config dictionary.

    Lazies are evaluated against the fully merged config, and results replace the
    Lazy nodes in place. Cycles raise an error.
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
        if not path:
            return self._resolve_value((), self.root, resolve_children=False)
        value = _get_path(self.root, path)
        resolved = self._resolve_value(path, value, resolve_children=False)
        if resolved is not value:
            _set_path(self.root, path, resolved)
        return resolved

    def _resolve_value(self, path, value, *, resolve_children: bool):
        if isinstance(value, Lazy):
            if path in self._resolving:
                raise ValueError(f"Lazy cycle detected at {_format_path(path)}")
            self._resolving.append(path)
            try:
                value = value.func(_wrap_proxy(self, (), self.root))
            finally:
                self._resolving.pop()
        if resolve_children and isinstance(value, dict):
            for key in list(value.keys()):
                child = value[key]
                resolved_child = self._resolve_value(
                    path + (key,),
                    child,
                    resolve_children=True,
                )
                if resolved_child is not child:
                    value[key] = resolved_child
        elif resolve_children and isinstance(value, list):
            for index in range(len(value)):
                child = value[index]
                resolved_child = self._resolve_value(
                    path + (index,),
                    child,
                    resolve_children=True,
                )
                if resolved_child is not child:
                    value[index] = resolved_child
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
    _assign_item(parent, last_key, target)


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
