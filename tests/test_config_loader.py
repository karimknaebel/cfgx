import textwrap
from pathlib import Path

import pytest

from cfgx import Lazy, load


def _write(path: Path, code: str):
    path.write_text(textwrap.dedent(code))


def test_parent_precedence(tmp_path):
    """
    parent1  -> parent2  -> child
       lr=0.1    lr=0.01    batch_size=64
    Expect: lr from parent2, plus optim from parent1, plus batch_size.
    """
    p1 = tmp_path / "parent1.py"
    _write(
        p1,
        """
        config = {"lr": 0.1, "optim": "sgd"}
        """,
    )

    p2 = tmp_path / "parent2.py"
    _write(
        p2,
        """
        parents = ["parent1.py"]
        config = {"lr": 0.01}
        """,
    )

    child = tmp_path / "child.py"
    _write(
        child,
        """
        parents = ["parent2.py"]
        config = {"batch_size": 64}
        """,
    )

    cfg = load(child)
    assert cfg == {"lr": 0.01, "optim": "sgd", "batch_size": 64}


def test_key_deletion(tmp_path):
    """
    Child deletes model.dropout.
    """
    parent = tmp_path / "parent.py"
    _write(
        parent,
        """
        config = {"model": {"name": "resnet", "dropout": 0.5}}
        """,
    )

    child = tmp_path / "child.py"
    _write(
        child,
        """
        from cfgx import Delete
        parents = ["parent.py"]
        config  = {"model": {"dropout": Delete()}}
        """,
    )

    cfg = load(child)
    assert cfg == {"model": {"name": "resnet"}}


def test_key_replacement(tmp_path):
    """
    Child replaces model with a new dict.
    """
    parent = tmp_path / "parent.py"
    _write(
        parent,
        """
        config = {"model": {"name": "resnet", "dropout": 0.5}}
        """,
    )

    child = tmp_path / "child.py"
    _write(
        child,
        """
        from cfgx import Replace
        parents = ["parent.py"]
        config  = {"model": Replace({"name": "vit", "activation": "relu"})}
        """,
    )

    cfg = load(child)
    assert cfg == {"model": {"name": "vit", "activation": "relu"}}


def test_load_multiple_configs_order(tmp_path):
    """
    Earlier paths should be overridden by later ones.
    """
    a = tmp_path / "a.py"
    _write(a, "config = {'a': 1, 'b': 2}")

    b = tmp_path / "b.py"
    _write(b, "config = {'b': 3, 'c': 4}")

    merged = load([a, b])
    assert merged == {"a": 1, "b": 3, "c": 4}


def test_load_list_matches_parent_chain(tmp_path):
    base = tmp_path / "base.py"
    _write(base, "config = {'x': 1}")

    mid = tmp_path / "mid.py"
    _write(
        mid,
        """
        parents = ["base.py"]
        config = {"x": 2}
        """,
    )

    prune = tmp_path / "prune.py"
    _write(
        prune,
        """
        from cfgx import Delete
        parents = ["base.py"]
        config = {"x": Delete()}
        """,
    )

    chain = tmp_path / "chain.py"
    _write(
        chain,
        """
        parents = ["mid.py", "prune.py"]
        """,
    )

    chained = load([mid, prune])
    assert chained == load(chain)
    assert chained == {}


def test_lazy_resolution_with_overrides(tmp_path):
    cfg_path = tmp_path / "cfg.py"
    _write(
        cfg_path,
        """
        from cfgx import Lazy
        config = {
            "trainer": {"steps": 1000},
            "warmup_steps": Lazy(lambda cfg: int(cfg["trainer"]["steps"] * 0.1)),
        }
        """,
    )

    cfg = load(cfg_path, overrides=["trainer.steps=5000"])
    assert cfg["trainer"]["steps"] == 5000
    assert cfg["warmup_steps"] == 500


def test_lazy_nested_access(tmp_path):
    cfg_path = tmp_path / "cfg.py"
    _write(
        cfg_path,
        """
        from cfgx import Lazy
        config = {
            "trainer": {"steps": 1000},
            "scheduler": {
                "warmup_steps": Lazy(lambda cfg: int(cfg["trainer"]["steps"] * 0.1))
            },
        }
        """,
    )

    cfg = load(cfg_path)
    assert cfg["scheduler"]["warmup_steps"] == 100


def test_lazy_attribute_access(tmp_path):
    cfg_path = tmp_path / "cfg.py"
    _write(
        cfg_path,
        """
        from cfgx import Lazy
        config = {
            "trainer": {"stages": [{"max_steps": 1000}]},
            "warmup_steps": Lazy(lambda c: int(c.trainer.stages[0].max_steps * 0.1)),
        }
        """,
    )

    cfg = load(cfg_path)
    assert cfg["warmup_steps"] == 100


def test_lazy_expression_shorthand(tmp_path):
    cfg_path = tmp_path / "cfg.py"
    _write(
        cfg_path,
        """
        from cfgx import Lazy
        config = {
            "trainer": {"stages": [{"max_steps": 1000}]},
            "warmup_steps": Lazy("c.trainer.stages[0].max_steps * 0.1"),
        }
        """,
    )

    cfg = load(cfg_path)
    assert cfg["warmup_steps"] == 100


def test_lazy_expression_builtins_available(tmp_path):
    cfg_path = tmp_path / "cfg.py"
    _write(
        cfg_path,
        """
        from cfgx import Lazy
        config = {
            "values": [1, 5, 3],
            "best": Lazy("max(c['values'])"),
        }
        """,
    )

    cfg = load(cfg_path)
    assert cfg["best"] == 5


def test_lazy_expression_math_available(tmp_path):
    cfg_path = tmp_path / "cfg.py"
    _write(
        cfg_path,
        """
        from cfgx import Lazy
        config = {
            "value": 9,
            "root": Lazy("math.sqrt(c['value'])"),
        }
        """,
    )

    cfg = load(cfg_path)
    assert cfg["root"] == 3.0


def test_lazy_same_dict_sibling_access(tmp_path):
    cfg_path = tmp_path / "cfg.py"
    _write(
        cfg_path,
        """
        from cfgx import Lazy
        config = {
            "trainer": {
                "max_steps": 1000,
                "log_every": Lazy("c.trainer.max_steps // 100"),
            },
        }
        """,
    )

    cfg = load(cfg_path)
    assert cfg["trainer"]["log_every"] == 10


def test_load_without_resolve_lazy(tmp_path):
    cfg_path = tmp_path / "cfg.py"
    _write(
        cfg_path,
        """
        from cfgx import Lazy
        config = {
            "steps": 1000,
            "warmup_steps": Lazy(lambda cfg: int(cfg["steps"] * 0.1)),
        }
        """,
    )

    cfg = load(cfg_path, resolve_lazy=False)
    assert isinstance(cfg["warmup_steps"], Lazy)


def test_lazy_cycle_raises(tmp_path):
    cfg_path = tmp_path / "cfg.py"
    _write(
        cfg_path,
        """
        from cfgx import Lazy
        config = {
            "a": Lazy(lambda cfg: cfg["b"]),
            "b": Lazy(lambda cfg: cfg["a"]),
        }
        """,
    )

    with pytest.raises(ValueError, match="Lazy cycle"):
        load(cfg_path)


def test_update_applies_left_to_right_across_parents(tmp_path):
    base = tmp_path / "base.py"
    _write(base, "config = {'x': 1}")

    mid = tmp_path / "mid.py"
    _write(
        mid,
        """
        from cfgx import Update
        parents = ["base.py"]
        config = {"x": Update(lambda v: v + 1)}
        """,
    )

    child = tmp_path / "child.py"
    _write(
        child,
        """
        from cfgx import Update
        parents = ["mid.py"]
        config = {"x": Update(lambda v: v * 10)}
        """,
    )

    cfg = load(child)
    assert cfg["x"] == 20


def test_update_missing_callable_without_default_raises(tmp_path):
    cfg_path = tmp_path / "cfg.py"
    _write(
        cfg_path,
        """
        from cfgx import Update
        config = {"x": Update(lambda v: v + 1)}
        """,
    )

    with pytest.raises(TypeError):
        load(cfg_path)


def test_update_missing_callable_with_default_works(tmp_path):
    cfg_path = tmp_path / "cfg.py"
    _write(
        cfg_path,
        """
        from cfgx import Update
        config = {"x": Update(lambda v=3: v + 1)}
        """,
    )

    cfg = load(cfg_path)
    assert cfg["x"] == 4


def test_update_string_expression_over_existing_value(tmp_path):
    base = tmp_path / "base.py"
    _write(base, "config = {'x': 10}")

    child = tmp_path / "child.py"
    _write(
        child,
        """
        from cfgx import Update
        parents = ["base.py"]
        config = {"x": Update("v * 0.1")}
        """,
    )

    cfg = load(child)
    assert cfg["x"] == 1.0


def test_update_string_expression_missing_value_raises(tmp_path):
    cfg_path = tmp_path / "cfg.py"
    _write(
        cfg_path,
        """
        from cfgx import Update
        config = {"x": Update("v * 0.1")}
        """,
    )

    with pytest.raises(TypeError):
        load(cfg_path)


def test_update_over_lazy_prev_resolves_composed_value(tmp_path):
    base = tmp_path / "base.py"
    _write(
        base,
        """
        from cfgx import Lazy
        config = {"foo": 10, "a": Lazy("c.foo")}
        """,
    )

    child = tmp_path / "child.py"
    _write(
        child,
        """
        from cfgx import Update
        parents = ["base.py"]
        config = {"a": Update(lambda v: v + 1)}
        """,
    )

    cfg = load(child)
    assert cfg["a"] == 11


def test_update_over_lazy_prev_tracks_later_dependency_overrides(tmp_path):
    base = tmp_path / "base.py"
    _write(
        base,
        """
        from cfgx import Lazy
        config = {"foo": 10, "a": Lazy("c.foo")}
        """,
    )

    mid = tmp_path / "mid.py"
    _write(
        mid,
        """
        from cfgx import Update
        parents = ["base.py"]
        config = {"a": Update(lambda v: v + 1)}
        """,
    )

    child = tmp_path / "child.py"
    _write(
        child,
        """
        parents = ["mid.py"]
        config = {"foo": 20}
        """,
    )

    cfg = load(child)
    assert cfg["a"] == 21


def test_update_over_lazy_prev_returning_lazy_resolves(tmp_path):
    base = tmp_path / "base.py"
    _write(
        base,
        """
        from cfgx import Lazy
        config = {"foo": 3, "bar": 7, "a": Lazy("c.foo")}
        """,
    )

    child = tmp_path / "child.py"
    _write(
        child,
        """
        from cfgx import Lazy, Update
        parents = ["base.py"]
        config = {"a": Update(lambda v: Lazy(lambda c: c.bar + v))}
        """,
    )

    cfg = load(child)
    assert cfg["a"] == 10


def test_update_nested_missing_value_in_new_branch_works(tmp_path):
    cfg_path = tmp_path / "cfg.py"
    _write(
        cfg_path,
        """
        from cfgx import Update
        config = {"x": {"y": Update(lambda v=1: v)}}
        """,
    )

    cfg = load(cfg_path)
    assert cfg["x"]["y"] == 1


def test_delete_nested_in_new_branch_does_not_leak_sentinel(tmp_path):
    cfg_path = tmp_path / "cfg.py"
    _write(
        cfg_path,
        """
        from cfgx import Delete
        config = {"x": {"y": Delete(), "z": 1}}
        """,
    )

    cfg = load(cfg_path)
    assert cfg["x"] == {"z": 1}


def test_replace_nested_in_new_branch_unwraps_value(tmp_path):
    cfg_path = tmp_path / "cfg.py"
    _write(
        cfg_path,
        """
        from cfgx import Replace
        config = {"x": {"y": Replace(1)}}
        """,
    )

    cfg = load(cfg_path)
    assert cfg["x"]["y"] == 1


def test_nested_update_under_dict_override_replacing_scalar_branch(tmp_path):
    base = tmp_path / "base.py"
    _write(base, "config = {'x': 1}")

    child = tmp_path / "child.py"
    _write(
        child,
        """
        from cfgx import Update
        parents = ["base.py"]
        config = {"x": {"y": Update(lambda v=2: v * 2)}}
        """,
    )

    cfg = load(child)
    assert cfg["x"] == {"y": 4}
