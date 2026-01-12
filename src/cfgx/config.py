import ast
import os
import re
import runpy
import subprocess
from collections.abc import Callable, Mapping, Sequence
from functools import reduce
from pathlib import Path
from pprint import pformat


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


def dump(config: dict, path: os.PathLike, formatter: str = "auto"):
    """
    Persist a config dictionary to a formatted Python file.
    """

    if formatter not in {"auto", "ruff", "pprint"}:
        raise ValueError(f"Unknown formatter: {formatter}")

    header = "# Auto-generated config snapshot\n"
    if formatter == "pprint" or (formatter == "auto" and not _ruff_available()):
        config_str = header + "config = " + pformat(config, width=88)
    else:
        config_str = _ruff_format(header + "config = " + repr(config))

    with open(path, "w") as f:
        f.write("# fmt: off\n")  # prevent auto-formatting
        f.write(config_str + "\n")


def format(config: dict, formatter: str = "auto") -> str:
    """Return a formatted string representation of the config dictionary."""
    if formatter not in {"auto", "ruff", "pprint"}:
        raise ValueError(f"Unknown formatter: {formatter}")

    if formatter == "pprint" or (formatter == "auto" and not _ruff_available()):
        return pformat(config, width=88)
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
    return result.stdout.rstrip("\n")


def _ruff_available() -> bool:
    try:
        import ruff
    except ModuleNotFoundError:
        return False
    return ruff is not None


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
        key, op, value = _split_override(override)
        keys = parse_key_path(key)
        if op == "+=":
            append_to_nested(cfg, keys, infer_type(value))
        elif op == "!=":
            delete_nested(cfg, keys)
        elif op == "-=":
            remove_value_from_list(cfg, keys, infer_type(value))
        else:
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
    if isinstance(last_key, int):
        while len(parent) <= last_key:
            parent.append(None)
    parent[last_key] = value


def append_to_nested(d: dict, keys, value):
    parent, last_key = _walk_to_parent(d, keys, create=True)
    if isinstance(last_key, int):
        while len(parent) <= last_key:
            parent.append(None)
        target = parent[last_key]
        if target is None:
            target = []
    else:
        target = parent[last_key] if last_key in parent else []
    if not isinstance(target, list):
        raise ValueError("Target is not a list")
    target.append(value)
    parent[last_key] = target


def delete_nested(d: dict, keys):
    parent, last_key = _walk_to_parent_if_exists(d, keys)
    if parent is None:
        return
    if isinstance(last_key, int):
        if not isinstance(parent, list):
            return
        index = last_key
        if index < 0:
            index += len(parent)
        if 0 <= index < len(parent):
            del parent[index]
    else:
        if not isinstance(parent, dict):
            return
        parent.pop(last_key, None)


def remove_value_from_list(d: dict, keys, value):
    parent, last_key = _walk_to_parent_if_exists(d, keys)
    if parent is None:
        return
    if isinstance(last_key, int):
        if not isinstance(parent, list):
            return
        index = last_key
        if index < 0:
            index += len(parent)
        if index < 0 or index >= len(parent):
            return
        target = parent[index]
    else:
        if not isinstance(parent, dict):
            return
        if last_key not in parent:
            return
        target = parent[last_key]
    if not isinstance(target, list):
        raise ValueError("Target is not a list")
    if value in target:
        target.remove(value)


def _walk_to_parent(d: dict, keys, *, create: bool):
    for i, key in enumerate(keys[:-1]):
        if create:
            if isinstance(key, int):
                while len(d) <= key:
                    d.append(None)
                if d[key] is None:
                    d[key] = {} if isinstance(keys[i + 1], str) else []
            else:
                if key not in d or d[key] is None:
                    d[key] = {} if isinstance(keys[i + 1], str) else []
        d = d[key]
    return d, keys[-1]


def _walk_to_parent_if_exists(d: dict, keys):
    current = d
    for key in keys[:-1]:
        if isinstance(key, int):
            if not isinstance(current, list):
                return None, None
            index = key
            if index < 0:
                index += len(current)
            if index < 0 or index >= len(current):
                return None, None
            current = current[index]
        else:
            if not isinstance(current, dict):
                return None, None
            if key not in current:
                return None, None
            current = current[key]
    return current, keys[-1]


def infer_type(val: str):
    if val.startswith("lazy:"):
        return Lazy(val[len("lazy:") :])
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val
