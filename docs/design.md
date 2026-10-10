# Resolution contract

This document specifies the new implementation. Configs are ordered definitions;
`load` resolves them once into an ordinary Python dictionary.

## Sources and layers

- Source positions are a file's exported `config` and arguments to `load` and
  `compose`. They accept plain dictionaries, expressions producing dictionaries,
  explicit `include` and `compose` declarations, and dictionary-valued `replace`.
- Source paths are implicit includes; source tuples are implicit compositions
  that flatten recursively in order. `config = {...}` is equivalent to
  `config = compose({...})`. Source lists are invalid.
- Dictionary values and list/tuple elements are value positions: strings, paths,
  and tuples remain data. Explicit includes and compositions contribute
  dictionary definitions at their insertion locations.
- Each contribution is a layer. Files add no boundary. Includes are relative to
  the including file and repeated includes apply again. Include cycles error.
- Nested compositions have local layers. Their previous views contain the
  enclosing layer's earlier config with preceding local contributions applied
  at the composition's location. Surrounding siblings retain the enclosing
  layer's previous view. Dictionary traversal order does not create layers.
- Dictionaries merge recursively. Scalars, opaque objects, lists, and tuples
  replace earlier values. A scalar or deletion between dictionary contributions
  breaks inheritance from the earlier dictionary.
- Mapping keys retain their first-contribution order, including when deleted and
  later reintroduced. Replacement starts a new order. Ordering never forces an
  otherwise overwritten value to execute merely to discover past membership.
- `replace(x)` discards inherited merging at its location. Later layers can
  still patch its result. `delete` removes a dictionary entry. Deletion at the
  root or in a list/tuple is invalid.
- Computations can return dictionaries, expressions, `replace`, `delete`,
  `include`, and `compose`. Returned values retain the producer's layer and bind
  at their output locations. Explicit includes and compositions introduce local
  contributions with their own origins. Returned includes retain the producer's
  declaring-file directory and include chain.
- Expression results do not undergo implicit source conversions. Returned
  strings and tuples remain data. All contributions at the root of a composition
  must produce plain dictionaries.

## Expressions and references

`final` reads all definitions; `previous` reads definitions strictly before the
expression's originating layer. Earlier definitions retain their own origins:
their `final` references still see the complete configuration.
The view includes preceding local contributions for nested compositions, while
other paths still reflect the enclosing layer's earlier config. Intermediate
views do not synthesize values for missing sequence slots; materializing those
slots raises `MissingValueError`.

Both roots accept an optional parent count. `final` equals `final(None)` and
starts at the root. `final(0)` starts at the expression's output location;
`final(1)` starts at its containing container. Each dictionary key or sequence
index counts as one path level. Going above the root errors. `previous` uses
the same addressing rules; `value` is `previous(0)`.

Attributes and indexing select keys, including arbitrary hashable dictionary
keys. Attribute access is key sugar, not Python object attribute lookup.
`final[('a', 'b')]` selects a tuple key. Expressions support arithmetic and
comparisons but cannot be iterated or converted to Python booleans.

- `expr.map(fn)` transforms a fully resolved selected value. The string form
  `expr.map('x * 2')` evaluates Python with `x`, the expression API, and `math`.
- `computed(fn)` calls `fn(get)`. `get(expr)` resolves the selected expression
  completely. Dependencies arise from reads actually executed by the callback.
- `expr.default(fallback)` handles direct absence, including an expression
  returning `delete`. It does not catch callback errors, missing dependencies,
  or cycles. Fallbacks are lazy and can themselves be expressions. Navigation
  after a default remains selective.
- `expr.keys()` returns a list of actual dictionary keys, settling any computed
  deletions but not materializing unrelated child contents.
- `expr.len()` determines container length. Supported sequence lengths do not
  depend on child values; dictionary lengths depend on actual membership.
- `expr.exists()` determines presence without materializing present contents.
  A computation that might return `delete` may need to run to determine presence.

Operation names remain usable as keys: `final.map` selects the key `map`,
whereas `final.map(fn)` transforms the root and `final.map.map(fn)` transforms
that key. Arbitrary Python functions require `map` or `computed`.

## Evaluation and ownership

- Each computation runs at most once per bound expression, output location, and
  originating layer within a load. Structure and value queries share results.
  Reusing an expression at another location creates a separate binding.
- Producers run atomically. cfgx cannot skip statements inside a Python callback.
  Produced structure is retained before resolving its children. Cycles are
  detected across actual operations, with a dependency trace.
- Exact built-in `dict`, `list`, and `tuple` are supported structure. Their
  children are resolved and structurally copied. Source aliases do not create
  output aliases. References such as `final.foo` also produce independent output
  containers. Tuples retain type and contents, without an identity guarantee.
- Callback inputs and `get` results are independent structural copies. Producer
  results are structurally copied when accepted. Raw cyclic supported containers
  are rejected. This is not `deepcopy`.
- Subclasses, sets, custom containers, functions, and other objects are opaque
  leaves and retain identity. Expressions inside opaque objects are not traversed.
  Explicit reads through their indexing interface are allowed, but depend on the
  opaque object as a whole. Its own methods govern keys and length queries.
- Avoid side effects, especially when the types of values are unknown. Mutating
  opaque values can affect sources and other locations. Caching does not promise
  isolation from arbitrary Python effects or an evaluation order for callbacks.

> **Note:** Resolution currently uses Python recursion. Long chains of config
> layers or overrides can raise `RecursionError`, even for flat dictionaries
> without expressions or dependency cycles. The threshold depends on the config
> and Python's recursion limit; there is no fixed supported layer count.

## Loading and overrides

`load(*sources, overrides=())` expands sources, appends override layers, and
resolves once. There is no separate public merge or unresolved-config API.
Reloading sources recomputes expressions. Loading a resolved snapshot cannot
recover its original formulas. Formatting and snapshot helpers remain available.

- `foo.bar=baz` contributes `{'foo': {'bar': replace('baz')}}`.
- `foo.bar!=` contributes `{'foo': {'bar': delete}}`. Declared ancestors remain
  present, including when newly introduced. The same applies to `expr:delete`
  and computed deletions.
- Values use Python literal parsing with unquoted-string fallback. `expr:`
  evaluates Python with `final`, `previous`, `value`, `computed`, `include`,
  `compose`, `replace`, `delete`, `math`, and ordinary builtins. It is trusted
  Python, like config files.
- `model=include:foo.py` contributes `{'model': replace(include('foo.py'))}`.
  Override paths to files resolve relative to the working directory; nested
  includes inside those files resolve relative to their declaring file.
- An argument starting with `expr:` contributes a dictionary or an explicit cfgx
  declaration producing dictionary contributions. It uses ordinary composition
  and can contain arbitrary keys. Explicit compositions can introduce multiple
  layers. A raw string or tuple result is not implicitly expanded.
- Assignment and deletion overrides each contribute a layer; `value` and
  `previous` use the same origins as definitions in config files.
- Shorthand paths select string dictionary keys using dots or quoted subscripts.
  Non-string subscripts, including sequence indices, are rejected. Whole-layer
  expressions support non-string keys; sequence transformations use `value.map`.
- All overrides become ordinary contributions before resolution. Missing
  dictionary ancestors are created; scalar, sequence, or opaque ancestors are
  replaced by the contributed dictionary. Opaque values are not mutated.
- Only assignment and deletion are override operators. List transformations use
  `expr:` rather than separate append/remove syntax.

## Acceptance examples

```python
from cfgx import computed, delete, final, load, previous, replace, value

assert load(
    {'lr': 1, 'backbone_lr': final.lr * 0.1},
    {'lr': value * 2},
) == {'lr': 2, 'backbone_lr': 0.2}

assert load(
    {'branch': computed(lambda get: {'a': 1, 'b': 2})},
    {'branch': {'a': 3}},
) == {'branch': {'a': 3, 'b': 2}}

assert load(
    {'steps': 100, 'cooldown': final.steps // 10},
    {'steps': 200, 'cooldown': value * 2},
) == {'steps': 200, 'cooldown': 40}

assert load({'block': {'width': 64, 'hidden': final(1).width * 2}}) == {
    'block': {'width': 64, 'hidden': 128},
}

assert load({'branch': {'gone': computed(lambda get: delete), 'kept': 1},
             'keys': final.branch.keys()}) == {
    'branch': {'kept': 1}, 'keys': ['kept'],
}
```
