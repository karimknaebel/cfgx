import ast
import os
import re
import runpy
import subprocess
from collections.abc import Callable, Mapping, Sequence
from functools import reduce
from pathlib import Path


class Delete:
    """Sentinel that removes a key from a merged config."""
    pass


class Replace:
    """Sentinel that forces a value to replace a mapping during merge."""

    def __init__(self, value, /):
        self.value = value


class Lazy:
    """Callable wrapper that defers computation until config resolution."""

    def __init__(self, func: Callable | str, /):
        if isinstance(func, str):
            code = compile(func, "<lazy>", "eval")

            def _from_expr(c):
                return eval(code, {}, {"c": c})

            self.func = _from_expr
        else:
            self.func = func


def load(
    path: os.PathLike | Sequence[os.PathLike],
    overrides: Sequence[str] | None = None,
    resolve_lazy: bool = True,
):
    """
    Load config modules from one or more paths, apply overrides, and merge the results.

    Parent configs (via `parents`) are resolved first, then later paths override
    earlier ones. `config` must be a dictionary.
    """

    paths = [path] if isinstance(path, (str, os.PathLike)) else list(path)
    configs = [cfg for p in paths for cfg in _collect_config_specs(Path(p))]
    cfg = reduce(merge, configs)
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


def dump(config: dict, path: os.PathLike):
    """
    Persist a config dictionary to a ruff-formatted Python file.
    """

    config_str = _ruff_format(
        "# Auto-generated config snapshot\nconfig = " + repr(config)
    )

    with open(path, "w") as f:
        f.write("# fmt: off\n")  # prevent auto-formatting
        f.write(config_str)


def format(config: dict) -> str:
    """Return a ruff-formatted string representation of the config dictionary."""
    return _ruff_format(repr(config))


def _ruff_format(source: str) -> str:
    from ruff.__main__ import find_ruff_bin

    result = subprocess.run(
        [find_ruff_bin(), "format", "--isolated", "--stdin-filename=config.py", "-"],
        input=source,
        text=True,
        capture_output=True,
        check=True,
        cwd=Path.cwd(),
    )
    result.check_returncode()
    return result.stdout


def merge(base: dict, override: dict):
    """
    Recursively merge two dictionaries, honoring Delete/Replace sentinels.

    If both sides contain dicts, merge continues down the tree. Delete removes a key
    from the base config, Replace overwrites without further deep merging, and other
    values simply override. Returns a new dictionary without mutating the inputs.
    """
    base = base.copy()
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            base[k] = merge(base[k], v)
        elif isinstance(v, Delete):
            base.pop(k, None)
        elif isinstance(v, Replace):
            base[k] = v.value
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
        if "+=" in override:
            key, value = override.split("+=", 1)
            keys = parse_key_path(key)
            append_to_nested(cfg, keys, infer_type(value))
        elif "!=" in override:
            key, _ = override.split("!=", 1)
            keys = parse_key_path(key)
            delete_nested(cfg, keys)
        elif "-=" in override:
            key, value = override.split("-=", 1)
            keys = parse_key_path(key)
            remove_value_from_list(cfg, keys, infer_type(value))
        else:
            key, value = override.split("=", 1)
            keys = parse_key_path(key)
            set_nested(cfg, keys, infer_type(value))
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
        self._resolve_value((), self.root)

    def resolve_at(self, path):
        if not path:
            return self._resolve_value((), self.root)
        value = _get_path(self.root, path)
        resolved = self._resolve_value(path, value)
        if resolved is not value:
            _set_path(self.root, path, resolved)
        return resolved

    def _resolve_value(self, path, value):
        if isinstance(value, Lazy):
            if path in self._resolving:
                raise ValueError(f"Lazy cycle detected at {_format_path(path)}")
            self._resolving.append(path)
            try:
                value = value.func(_wrap_proxy(self, (), self.root))
            finally:
                self._resolving.pop()
        if isinstance(value, dict):
            for key in list(value.keys()):
                child = value[key]
                resolved_child = self._resolve_value(path + (key,), child)
                if resolved_child is not child:
                    value[key] = resolved_child
        elif isinstance(value, list):
            for index in range(len(value)):
                child = value[index]
                resolved_child = self._resolve_value(path + (index,), child)
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


def set_nested(d: dict, keys, value):
    for i, key in enumerate(keys):
        is_last = i == len(keys) - 1
        if isinstance(key, int):
            while len(d) <= key:
                d.append(None)
            if is_last:
                d[key] = value
            else:
                if d[key] is None:
                    d[key] = {} if isinstance(keys[i + 1], str) else []
                d = d[key]
        else:
            if is_last:
                d[key] = value
            else:
                if key not in d or d[key] is None:
                    d[key] = {} if isinstance(keys[i + 1], str) else []
                d = d[key]


def append_to_nested(d: dict, keys, value):
    for i, key in enumerate(keys):
        is_last = i == len(keys) - 1
        next_key_type = type(keys[i + 1]) if not is_last else None

        if isinstance(key, int):
            while len(d) <= key:
                d.append(None)
            if is_last:
                if d[key] is None:
                    d[key] = []
                if not isinstance(d[key], list):
                    raise ValueError(f"Target at index {key} is not a list")
                d[key].append(value)
            else:
                if d[key] is None:
                    d[key] = {} if next_key_type is str else []
                d = d[key]
        else:
            if is_last:
                if key not in d or not isinstance(d[key], list):
                    d[key] = []
                d[key].append(value)
            else:
                if key not in d or d[key] is None:
                    d[key] = {} if next_key_type is str else []
                d = d[key]


def delete_nested(d: dict, keys):
    for i, key in enumerate(keys[:-1]):
        d = d[key]
    last_key = keys[-1]
    if isinstance(last_key, int):
        if isinstance(d, list) and 0 <= last_key < len(d):
            del d[last_key]
    else:
        d.pop(last_key, None)


def remove_value_from_list(d: dict, keys, value):
    for key in keys:
        d = d[key]
    if isinstance(d, list) and value in d:
        d.remove(value)
    # TODO: this should probably raise if d is not a list


def infer_type(val: str):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val
