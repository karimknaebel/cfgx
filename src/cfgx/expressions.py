"""Declarations evaluated by :func:`cfgx.load`."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass


class _Delete:
    def __repr__(self):
        return "delete"


delete = _Delete()
"""Remove a dictionary entry from the composed configuration."""


@dataclass(frozen=True)
class _Replacement:
    value: object


def replace(x, /):
    """Replace inherited contents at this location instead of merging dictionaries."""
    return _Replacement(x)


class Expression:
    """A deferred selection or calculation, resolved within a config location.

    Attributes and subscripts select keys. Calling ``map(fn)``, ``default(x)``,
    ``keys()``, ``len()``, or ``exists()`` on a selection constructs an operation.
    These names can also be selected as ordinary dictionary keys.
    """

    __slots__ = ("_op", "_args")

    def __init__(self, op, *args):
        self._op = op
        self._args = args

    def __getattr__(self, key):
        if key.startswith("__"):
            raise AttributeError(key)
        return self[key]

    def __getitem__(self, key):
        return Expression("select", self, key)

    def __call__(self, *args, **kwargs):
        if self._op == "root":
            if kwargs:
                if args or set(kwargs) != {"parent"}:
                    raise TypeError("References accept only a parent count")
                args = (kwargs["parent"],)
            if len(args) > 1:
                raise TypeError("References accept only a parent count")
            parent = args[0] if args else None
            if parent is not None and (type(parent) is not int or parent < 0):
                raise ValueError("Parent count must be a nonnegative integer or None")
            return Expression("root", self._args[0], parent)
        if self._op == "select" and not kwargs:
            source, operation = self._args
            if operation in ("map", "default") and len(args) == 1:
                if operation == "map":
                    fn = args[0]
                    if isinstance(fn, str):
                        code = compile(fn, "<cfgx map>", "eval")

                        def fn(x):
                            return eval(code, namespace() | {"x": x})

                    if not callable(fn):
                        raise TypeError("map expects a callable or Python expression")
                    return Expression("map", source, fn)
                return Expression("default", source, args[0])
            if operation in ("keys", "len", "exists") and not args:
                return Expression(operation, source)
        raise TypeError(
            "Call map(fn), default(x), keys(), len(), or exists() on expressions"
        )

    def __bool__(self):
        raise TypeError("Expressions have no Python truth value; use computed and get")

    def __iter__(self):
        raise TypeError("Expressions cannot be iterated; use map or computed and get")

    def __repr__(self):
        if self._op == "root":
            name, parent = self._args
            return name if parent is None else f"{name}({parent})"
        if self._op == "select":
            source, key = self._args
            if isinstance(key, str) and key.isidentifier():
                return f"{source!r}.{key}"
            return f"{source!r}[{key!r}]"
        return f"{self._op}({', '.join(repr(x) for x in self._args)})"


def _binary(fn, reflected=False):
    def operation(self, other):
        return (
            Expression("binary", fn, other, self)
            if reflected
            else Expression("binary", fn, self, other)
        )

    return operation


def _unary(fn):
    def operation(self):
        return Expression("unary", fn, self)

    return operation


for _name in (
    "add",
    "sub",
    "mul",
    "truediv",
    "floordiv",
    "mod",
    "pow",
    "matmul",
    "and",
    "or",
    "xor",
    "lshift",
    "rshift",
):
    _fn = getattr(operator, _name + "_" if _name in ("and", "or") else _name)
    setattr(Expression, f"__{_name}__", _binary(_fn))
    setattr(Expression, f"__r{_name}__", _binary(_fn, reflected=True))

for _name in ("eq", "ne", "lt", "le", "gt", "ge"):
    setattr(Expression, f"__{_name}__", _binary(getattr(operator, _name)))

for _name in ("neg", "pos", "abs", "invert"):
    setattr(Expression, f"__{_name}__", _unary(getattr(operator, _name)))

Expression.__hash__ = None

final = Expression("root", "final", None)
"""Read the complete configuration, including later layers and overrides."""
previous = Expression("root", "previous", None)
"""Read definitions contributed before the expression's originating layer."""
value = previous(0)
"""Read the inherited value at the current output location."""


def computed(fn, /):
    """Declare a calculation whose callable receives ``get(expression)``.

    Only reads actually executed by the callable become dependencies. Each
    binding runs at most once per load. The result may contain more declarations.
    """
    if not callable(fn):
        raise TypeError("computed expects a callable receiving get")
    return Expression("computed", fn)


def namespace():
    return {
        "final": final,
        "previous": previous,
        "value": value,
        "computed": computed,
        "replace": replace,
        "delete": delete,
        "math": math,
    }
