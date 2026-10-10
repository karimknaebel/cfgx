"""Config source expansion and the public load operation."""

from __future__ import annotations

import os
import runpy
from dataclasses import dataclass
from pathlib import Path

from .expressions import Expression, _Composition, _Include, _Replacement, include
from .overrides import parse_override
from .resolver import ConfigError, Resolver


@dataclass
class _Source:
    value: object
    directory: Path
    stack: tuple


def load(*sources, overrides=()):
    """Compose sources and overrides into an independent, resolved dictionary.

    Sources are paths, plain dictionaries, expressions producing dictionaries,
    include/compose declarations, replacements, or tuples of sources. Every
    file defines ``config`` using the same syntax. Nested includes and
    compositions bind definitions at their insertion location.
    Includes expand in order relative to the declaring file; repeated includes
    apply repeatedly. Each contribution is a layer; explicit compositions can
    introduce local layers inside a value.

    Exact built-in dictionaries, lists, and tuples are structurally copied.
    Other objects are opaque and retain identity. No source container is mutated.
    Overrides contribute ordinary layers using ``path=value``, ``path!=``, or
    ``expr:layer``. Paths select string dictionary keys. Assignment replaces its
    leaf; dictionary ancestors merge normally. Values can use Python ``expr:``.
    """
    layers = list(_expand(sources, Path.cwd()))
    for item in overrides:
        layers.extend(_expand(parse_override(item), Path.cwd()))
    return Resolver(layers, _expand).resolve()


def _expand(source, directory, stack=()):
    if type(source) is dict or isinstance(source, (Expression, _Replacement)):
        yield _Source(source, directory, stack)
    elif isinstance(source, _Composition):
        yield from _expand(source.sources, directory, stack)
    elif type(source) is tuple:
        for item in source:
            yield from _expand(item, directory, stack)
    elif isinstance(source, (str, os.PathLike)):
        yield from _expand(include(source), directory, stack)
    elif isinstance(source, _Include):
        path = (directory / source.path).resolve()
        if path in stack:
            raise ConfigError(
                "Config include cycle: " + " -> ".join(map(str, (*stack, path)))
            )
        module = runpy.run_path(str(path), run_name="__config__")
        if "config" not in module:
            raise ConfigError(f"Config file {path} must define 'config'")
        yield from _expand(module["config"], path.parent, (*stack, path))
    else:
        raise TypeError(
            "Config sources must be paths, plain dicts, expressions, include/compose "
            "declarations, replacements, or tuples of sources"
        )
