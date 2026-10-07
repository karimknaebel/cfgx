"""Formatting and Python snapshots of resolved configurations."""

import subprocess
from pathlib import Path
from pprint import pformat
from typing import TextIO


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

    sort_keys orders dict keys throughout nested dict/list/tuple structures.
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

    sort_keys orders dict keys throughout nested dict/list/tuple structures.
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
        config_str = "config = " + pformat(config, width=88, sort_dicts=sort_keys)
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

    sort_keys orders dict keys throughout nested dict/list/tuple structures.
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
    if type(value) is dict:
        return {key: _sort_keys(value[key]) for key in sorted(value)}
    if type(value) in (list, tuple):
        return type(value)(_sort_keys(item) for item in value)
    return value
