"""Resolve ordered definitions without exposing lazy containers to callbacks."""

from __future__ import annotations

from dataclasses import dataclass

from .expressions import Expression, _Replacement, delete

_ABSENT = object()


class ConfigError(ValueError):
    """Invalid config structure or dependency cycle."""


class MissingValueError(KeyError):
    """A selected config value does not exist."""


def _path(path):
    return "$" + "".join(
        f".{key}" if isinstance(key, str) and key.isidentifier() else f"[{key!r}]"
        for key in path
    )


def _missing(x):
    return x is _ABSENT or x is delete or isinstance(x, _Missing)


def _copy(x, ancestors=()):
    if type(x) not in (dict, list, tuple):
        return x
    if id(x) in ancestors:
        raise ConfigError("Cyclic Python containers are not supported")
    ancestors = (*ancestors, id(x))
    if type(x) is dict:
        return {k: _copy(v, ancestors) for k, v in x.items()}
    return type(x)(_copy(v, ancestors) for v in x)


@dataclass
class _Container:
    kind: type
    children: dict | list | tuple


@dataclass
class _Head:
    data: object
    replace: bool = False


@dataclass
class _Failure:
    error: Exception


@dataclass
class _Missing:
    path: tuple


class _Node:
    def __init__(self, resolver, path, layer):
        self.resolver = resolver
        self.path = path
        self.layer = layer

    def cached(self, operation, fn, *args):
        return self.resolver.cached((self, operation, *args), fn)

    def head(self):
        return self.cached("structure", self._head)

    def child(self, key):
        if isinstance(key, slice):
            return self.cached(
                "slice", lambda: self._child(key), key.start, key.stop, key.step
            )
        return self.cached("select", lambda: self._child(key), key)

    def _child(self, key):
        data = self.head().data
        absent = _Literal(self.resolver, (*self.path, key), self.layer, _ABSENT)
        if isinstance(data, _Container):
            if data.kind is dict:
                return data.children.get(key, absent)
            if isinstance(key, slice):
                return _Literal(
                    self.resolver,
                    self.path,
                    self.layer,
                    _Container(data.kind, data.children[key]),
                )
            try:
                return data.children[key]
            except IndexError:
                return absent
        if _missing(data):
            return absent
        try:
            result = data[key]
        except (KeyError, IndexError):
            return absent
        return self.resolver.bind(result, (*self.path, key), self.layer, opaque=True)

    def dict_child(self, key):
        data = self.head().data
        if isinstance(data, _Container) and data.kind is dict:
            return data.children.get(key, self.resolver.absent)
        return self.resolver.absent

    def present(self):
        return self.cached("exists", self._present)

    def _present(self):
        return self.presence() is True

    def presence(self):
        return self.cached("presence", self._presence)

    def _presence(self):
        data = self.head().data
        return None if data is _ABSENT else not _missing(data)

    def keys(self):
        return self.cached("keys", self._keys)

    def _keys(self):
        data = self.head().data
        if isinstance(data, _Container) and data.kind is dict:
            return [key for key, child in data.children.items() if child.present()]
        if _missing(data):
            raise MissingValueError(_path(self.path))
        if isinstance(data, _Container):
            raise TypeError(f"keys() requires a mapping at {_path(self.path)}")
        return list(data.keys())

    def length(self):
        data = self.head().data
        if isinstance(data, _Container):
            return len(self.keys()) if data.kind is dict else len(data.children)
        if _missing(data):
            raise MissingValueError(_path(self.path))
        return len(data)

    def materialize(self):
        return self.cached("value", self._materialize)

    def _materialize(self):
        data = self.head().data
        if isinstance(data, _Missing):
            raise MissingValueError(f"Missing config value at {_path(data.path)}")
        if not isinstance(data, _Container):
            return data
        if data.kind is dict:
            result = {}
            for key, child in data.children.items():
                item = child.materialize()
                if not _missing(item):
                    result[key] = _copy(item)
            return result
        result = []
        for index, child in enumerate(data.children):
            item = child.materialize()
            if _missing(item):
                raise ConfigError(
                    f"delete is not allowed in a sequence at {_path(self.path)}[{index}]"
                )
            result.append(_copy(item))
        return data.kind(result)


class _Literal(_Node):
    def __init__(self, resolver, path, layer, data):
        super().__init__(resolver, path, layer)
        self.data = data

    def _head(self):
        return _Head(self.data)


class _Replace(_Node):
    def __init__(self, target):
        super().__init__(target.resolver, target.path, target.layer)
        self.target = target

    def _head(self):
        return _Head(self.target.head().data, replace=True)

    def _child(self, key):
        return self.target.child(key)

    def _presence(self):
        return self.target.presence()


class _DictChild(_Node):
    def __init__(self, parent, key):
        super().__init__(parent.resolver, (*parent.path, key), parent.layer)
        self.parent = parent
        self.key = key

    def target(self):
        return self.cached("inherited", self._target)

    def _target(self):
        return self.parent.dict_child(self.key)

    def _head(self):
        return self.target().head()

    def _child(self, key):
        return self.target().child(key)

    def dict_child(self, key):
        return self.target().dict_child(key)

    def _presence(self):
        return self.target().presence()


class _Merge(_Node):
    def __init__(self, lower, upper):
        super().__init__(
            upper.resolver,
            lower.path if upper is upper.resolver.absent else upper.path,
            upper.layer,
        )
        self.lower = lower
        self.upper = upper

    def _head(self):
        head = self.upper.head()
        if head.data is _ABSENT:
            return self.lower.head()
        if (
            head.replace
            or not isinstance(head.data, _Container)
            or head.data.kind is not dict
        ):
            return head
        lower = self.lower.head().data
        if not isinstance(lower, _Container) or lower.kind is not dict:
            return head
        keys = dict.fromkeys(lower.children)
        keys.update(dict.fromkeys(head.data.children))
        return _Head(_Container(dict, {key: self.child(key) for key in keys}))

    def _child(self, key):
        head = self.upper.head()
        if head.data is _ABSENT:
            return self.lower.child(key)
        if (
            head.replace
            or not isinstance(head.data, _Container)
            or head.data.kind is not dict
        ):
            return self.upper.child(key)
        return _Merge(
            _DictChild(self.lower, key),
            head.data.children.get(key, self.resolver.absent),
        )

    def _presence(self):
        presence = self.upper.presence()
        return self.lower.presence() if presence is None else presence

    def dict_child(self, key):
        head = self.upper.head()
        if head.data is _ABSENT:
            return self.lower.dict_child(key)
        if head.replace:
            return self.upper.dict_child(key)
        if isinstance(head.data, _Container) and head.data.kind is dict:
            return self.child(key)
        return self.resolver.absent


class _Read(_Node):
    """A value reference preserves source bindings and consumes merge instructions."""

    def __init__(self, target, *, entry=False):
        super().__init__(target.resolver, target.path, target.layer)
        self.target = target
        self.entry = entry

    def _head(self):
        data = self.target.head().data
        if isinstance(data, _Container):
            if data.kind is dict:
                data = _Container(
                    dict,
                    {
                        key: _Read(child, entry=True)
                        for key, child in data.children.items()
                    },
                )
            else:
                data = _Container(
                    data.kind, [_Read(child, entry=True) for child in data.children]
                )
        if _missing(data):
            data = _ABSENT if self.entry else _Missing(self.path)
        return _Head(data)

    def _child(self, key):
        return _Read(self.target.child(key))

    def _materialize(self):
        result = self.target.materialize()
        if _missing(result):
            if self.entry:
                return _ABSENT
            raise MissingValueError(f"Missing config value at {_path(self.path)}")
        return result

    def _presence(self):
        present = self.target.present()
        return None if self.entry and not present else present


class _Evaluation(_Node):
    def __init__(self, resolver, path, layer, expression):
        super().__init__(resolver, path, layer)
        self.expression = expression

    def target(self):
        return self.cached("expression", self._evaluate)

    def _head(self):
        return self.target().head()

    def _child(self, key):
        return self.target().child(key)

    def _presence(self):
        if self.expression._op in ("keys", "len", "exists"):
            return True
        return self.target().presence()

    def _evaluate(self):
        op, args = self.expression._op, self.expression._args

        def bind(x):
            return (
                self.resolver.expression(x, self.path, self.layer)
                if isinstance(x, Expression)
                else self.resolver.bind(x, self.path, self.layer)
            )

        def get(expr):
            if not isinstance(expr, Expression):
                raise TypeError("get expects a cfgx expression")
            return self.resolver.read(bind(expr))

        if op == "root":
            view, parent = args
            if parent is not None and parent > len(self.path):
                raise ConfigError(
                    f"Reference goes above the root at {_path(self.path)}"
                )
            path = () if parent is None else self.path[: len(self.path) - parent]
            node = self.resolver.roots[-1 if view == "final" else self.layer]
            for key in path:
                node = node.child(key)
            return _Read(node)
        if op == "select":
            return bind(args[0]).child(args[1])
        if op == "default":
            source = bind(args[0])
            return source if source.present() else bind(args[1])
        if op == "map":
            return bind(args[1](get(args[0])))
        if op == "computed":
            return bind(args[0](get))
        if op == "binary":
            fn, left, right = args
            return bind(
                fn(self.resolver.read(bind(left)), self.resolver.read(bind(right)))
            )
        if op == "unary":
            return bind(args[0](get(args[1])))
        source = bind(args[0])
        if op == "exists":
            return bind(source.present())
        if op == "keys":
            return bind(source.keys())
        if op == "len":
            return bind(source.length())
        raise AssertionError(op)


class _Definition(_Node):
    def __init__(self, target):
        super().__init__(target.resolver, target.path, target.layer)
        self.target = target

    def _head(self):
        head = self.target.head()
        if isinstance(head.data, _Missing):
            raise MissingValueError(
                f"Missing config value at {_path(head.data.path)} (referenced from {_path(self.path)})"
            )
        return head

    def _child(self, key):
        self.present()
        return self.target.child(key)

    def _presence(self):
        if self.target.present():
            return True
        return not _missing(self.head().data)


class _Root(_Node):
    def __init__(self, target):
        super().__init__(target.resolver, (), target.layer)
        self.target = target

    def _head(self):
        head = self.target.head()
        if not isinstance(head.data, _Container) or head.data.kind is not dict:
            raise ConfigError(
                f"Root contribution in layer {self.layer + 1} must produce a plain dict"
            )
        return head

    def _child(self, key):
        self.head()
        return self.target.child(key)


class Resolver:
    def __init__(self, sources):
        self.cache = {}
        self.active = []
        self.expressions = {}
        self.absent = _Literal(self, (), -1, _ABSENT)
        self.roots = [self.bind({}, (), -1)]
        for layer, source in enumerate(sources):
            self.roots.append(
                _Merge(self.roots[-1], _Root(self.bind(source, (), layer)))
            )

    def cached(self, key, fn):
        if key in self.cache:
            result = self.cache[key]
            if isinstance(result, _Failure):
                raise result.error
            return result
        if key in self.active:
            trace = self.active[self.active.index(key) :] + [key]
            raise ConfigError(
                "Dependency cycle: "
                + " -> ".join(
                    f"{_path(node.path)} (layer {node.layer + 1}, {op})"
                    for node, op, *_ in trace
                )
            )
        self.active.append(key)
        try:
            result = fn()
            self.cache[key] = result
            return result
        except Exception as exc:
            self.cache[key] = _Failure(exc)
            raise
        finally:
            self.active.pop()

    def bind(self, x, path, layer, ancestors=(), opaque=False):
        if not opaque and isinstance(x, Expression):
            return _Definition(self.expression(x, path, layer))
        if not opaque and isinstance(x, _Replacement):
            return _Replace(self.bind(x.value, path, layer, ancestors))
        if type(x) in (dict, list, tuple):
            if id(x) in ancestors:
                raise ConfigError(f"Cyclic Python container at {_path(path)}")
            ancestors = (*ancestors, id(x))
            if type(x) is dict:
                children = {
                    k: self.bind(v, (*path, k), layer, ancestors, opaque)
                    for k, v in x.items()
                }
            else:
                children = [
                    self.bind(v, (*path, k), layer, ancestors, opaque)
                    for k, v in enumerate(x)
                ]
            return _Literal(self, path, layer, _Container(type(x), children))
        return _Literal(self, path, layer, x)

    def expression(self, x, path, layer):
        key = (id(x), path, layer)
        if key not in self.expressions:
            self.expressions[key] = _Evaluation(self, path, layer, x)
        return self.expressions[key]

    def read(self, node):
        result = node.materialize()
        if _missing(result):
            raise MissingValueError(f"Missing config value at {_path(node.path)}")
        return _copy(result)

    def resolve(self):
        return self.read(self.roots[-1])
