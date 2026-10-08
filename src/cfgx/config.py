"""Config source expansion and the public load operation."""

from __future__ import annotations

import os
import runpy
from pathlib import Path

from .expressions import Expression
from .overrides import parse_override
from .resolver import ConfigError, Resolver


def load(*sources, overrides=()):
    """Compose sources and overrides into an independent, resolved dictionary.

    Sources are paths, plain dictionaries, expressions producing dictionaries,
    or tuples of sources. Every file defines ``config`` using the same syntax.
    Includes expand in order relative to the declaring file; repeated includes
    apply repeatedly. Each contribution and each override is a separate layer.

    Exact built-in dictionaries, lists, and tuples are structurally copied.
    Other objects are opaque and retain identity. No source container is mutated.
    Overrides contribute ordinary layers using ``path=value``, ``path!=``, or
    ``expr:layer``. Paths select string dictionary keys. Assignment replaces its
    leaf; dictionary ancestors merge normally. Values can use Python ``expr:``.
    """
    layers = list(_expand(sources, Path.cwd()))
    layers.extend(parse_override(item) for item in overrides)
    return Resolver(layers).resolve()


def _expand(source, directory, stack=()):
    if type(source) is dict or isinstance(source, Expression):
        yield source
    elif type(source) is tuple:
        for item in source:
            yield from _expand(item, directory, stack)
    elif isinstance(source, (str, os.PathLike)):
        path = (directory / source).resolve()
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
            "Config sources must be paths, plain dicts, expressions, or tuples of sources"
        )
