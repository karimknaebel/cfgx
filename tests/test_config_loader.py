import textwrap
from pathlib import Path

import pytest

from cfgx import Lazy, Update, load


def _write(path: Path, code: str):
    path.write_text(textwrap.dedent(code))


def test_config_reference_precedence(tmp_path):
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
        config = ["parent1.py", {"lr": 0.01}]
        """,
    )

    child = tmp_path / "child.py"
    _write(
        child,
        """
        config = ["parent2.py", {"batch_size": 64}]
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
        config = ["parent.py", {"model": {"dropout": Delete()}}]
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
        config = ["parent.py", {"model": Replace({"name": "vit", "activation": "relu"})}]
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


def test_load_list_matches_config_file(tmp_path):
    base = tmp_path / "base.py"
    _write(base, "config = {'x': 1}")

    mid = tmp_path / "mid.py"
    _write(
        mid,
        """
        config = ["base.py", {"x": 2}]
        """,
    )

    prune = tmp_path / "prune.py"
    _write(
        prune,
        """
        from cfgx import Delete
        config = ["base.py", {"x": Delete()}]
        """,
    )

    chain = tmp_path / "chain.py"
    _write(
        chain,
        """
        config = ["mid.py", "prune.py"]
        """,
    )

    chained = load([mid, prune])
    assert chained == load(chain)
    assert chained == {}


def test_config_list_mixes_files_and_dicts(tmp_path):
    _write(tmp_path / "foo.py", 'config = {"x": 1, "options": {"a": 1}}')
    _write(
        tmp_path / "bar.py",
        """
        from cfgx import Update
        config = {"x": Update("v * 2"), "options": {"b": 2}}
        """,
    )
    _write(
        tmp_path / "config.py",
        """
        from cfgx import Update
        config = ["foo.py", {"x": 3}, "bar.py", {"x": Update("v + 1")}]
        """,
    )

    assert load(tmp_path / "config.py") == {"x": 7, "options": {"a": 1, "b": 2}}


def test_config_lists_expand_before_merging(tmp_path):
    _write(
        tmp_path / "base.py",
        'config = {"x": 10, "debug": True, "model": {"a": 1}}',
    )
    _write(
        tmp_path / "transform.py",
        """
        from cfgx import Delete, Replace, Update
        config = {
            "x": Update("v * 2"),
            "debug": Delete(),
            "model": Replace({"b": 2}),
        }
        """,
    )
    _write(
        tmp_path / "group.py",
        'config = ["transform.py", {"model": {"c": 3}}]',
    )
    _write(tmp_path / "config.py", 'config = ["base.py", "group.py"]')

    assert load(tmp_path / "config.py") == {"x": 20, "model": {"b": 2, "c": 3}}
    assert load(tmp_path / "config.py") == load(
        [tmp_path / "base.py", tmp_path / "transform.py", {"model": {"c": 3}}]
    )


def test_config_list_paths_are_relative_to_declaring_file(tmp_path):
    (tmp_path / "presets").mkdir()
    _write(tmp_path / "presets" / "base.py", 'config = {"x": 5}')
    _write(
        tmp_path / "presets" / "preset.py",
        """
        from pathlib import Path
        config = [Path("base.py"), {"preset": True}]
        """,
    )
    _write(
        tmp_path / "finish.py",
        """
        from cfgx import Update
        config = {"x": Update("v + 1")}
        """,
    )
    _write(
        tmp_path / "config.py",
        """
        from pathlib import Path
        config = [Path("presets/preset.py"), "finish.py"]
        """,
    )

    assert load(tmp_path / "config.py") == {"x": 6, "preset": True}


def test_config_lists_flatten_without_expanding_dictionary_values(tmp_path):
    _write(
        tmp_path / "config.py",
        """
        config = [
            [],
            [{"x": 1}, [{"x": 2}]],
            {
                "items": ["missing.py", {"nested": [1, 2]}],
                "parents": ["also_missing.py"],
                "config": [{"y": 3}],
            },
        ]
        """,
    )

    assert load(tmp_path / "config.py") == {
        "x": 2,
        "items": ["missing.py", {"nested": [1, 2]}],
        "parents": ["also_missing.py"],
        "config": [{"y": 3}],
    }


def test_config_list_repeated_references_apply_each_time(tmp_path):
    _write(
        tmp_path / "increment.py",
        """
        from cfgx import Update
        config = {"x": Update("v + 1")}
        """,
    )
    _write(
        tmp_path / "config.py",
        'config = [{"x": 1}, "increment.py", "increment.py"]',
    )

    assert load(tmp_path / "config.py") == {"x": 3}


def test_shared_config_references_are_reapplied(tmp_path):
    _write(tmp_path / "base.py", 'config = {"x": 1}')
    _write(tmp_path / "left.py", 'config = ["base.py", {"x": 2}]')
    _write(tmp_path / "right.py", 'config = ["base.py", {"y": 3}]')
    _write(tmp_path / "config.py", 'config = ["left.py", "right.py"]')

    assert load(tmp_path / "config.py") == {"x": 1, "y": 3}


@pytest.mark.parametrize("declaration", ["config = {}", "config = []"])
def test_empty_config(tmp_path, declaration):
    _write(tmp_path / "config.py", declaration)

    assert load(tmp_path / "config.py") == {}


@pytest.mark.parametrize("declaration", ["", "configs = {'x': 1}"])
def test_config_declaration_is_required(tmp_path, declaration):
    _write(tmp_path / "missing.py", declaration)
    _write(tmp_path / "config.py", 'config = ["missing.py", {"y": 2}]')

    with pytest.raises(ValueError, match="must define 'config'") as error:
        load(tmp_path / "config.py")
    assert str(tmp_path / "missing.py") in str(error.value)


def test_load_only_reads_config_attribute(tmp_path):
    _write(
        tmp_path / "config.py",
        'parents = ["missing.py"]\nconfig = {"x": 1}',
    )

    assert load(tmp_path / "config.py") == {"x": 1}


def test_load_variadic_sources(tmp_path):
    _write(
        tmp_path / "double.py",
        """
        from cfgx import Update
        config = {"x": Update("v * 2")}
        """,
    )

    assert load({"x": 3}, tmp_path / "double.py", {"y": 4}) == {
        "x": 6,
        "y": 4,
    }
    assert load([{"x": 3}, tmp_path / "double.py"], {"y": 4}) == load(
        [{"x": 3}, tmp_path / "double.py", {"y": 4}]
    )
    assert load({"x": 3}, tmp_path / "double.py", overrides=["x=7"]) == {"x": 7}
    assert load() == {}


def test_load_accepts_inline_configs(tmp_path):
    _write(
        tmp_path / "increment.py",
        """
        from cfgx import Update
        config = {"x": Update("v + 1")}
        """,
    )

    assert load([{"x": 1}, tmp_path / "increment.py", {"x": Update("v * 2")}]) == {
        "x": 4
    }
    assert load({"x": 1}, overrides=["x=2"]) == {"x": 2}
    assert load([]) == {}


def test_config_list_lazy_resolves_after_overrides(tmp_path):
    _write(
        tmp_path / "base.py",
        """
        from cfgx import Lazy
        config = {"x": 2, "scaled": Lazy("c.x * 2")}
        """,
    )
    _write(
        tmp_path / "config.py",
        """
        from cfgx import Update
        config = ["base.py", {"scaled": Update("v + 1")}, {"x": 3}]
        """,
    )

    cfg = load(tmp_path / "config.py", resolve_lazy=False)
    assert cfg["x"] == 3
    assert isinstance(cfg["scaled"], Lazy)
    assert load(tmp_path / "config.py", overrides=["x=5"]) == {"x": 5, "scaled": 11}


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


def test_update_applies_left_to_right_across_files(tmp_path):
    base = tmp_path / "base.py"
    _write(base, "config = {'x': 1}")

    mid = tmp_path / "mid.py"
    _write(
        mid,
        """
        from cfgx import Update
        config = ["base.py", {"x": Update(lambda v: v + 1)}]
        """,
    )

    child = tmp_path / "child.py"
    _write(
        child,
        """
        from cfgx import Update
        config = ["mid.py", {"x": Update(lambda v: v * 10)}]
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
        config = ["base.py", {"x": Update("v * 0.1")}]
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
        config = ["base.py", {"a": Update(lambda v: v + 1)}]
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
        config = ["base.py", {"a": Update(lambda v: v + 1)}]
        """,
    )

    child = tmp_path / "child.py"
    _write(
        child,
        """
        config = ["mid.py", {"foo": 20}]
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
        config = ["base.py", {"a": Update(lambda v: Lazy(lambda c: c.bar + v))}]
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
        config = ["base.py", {"x": {"y": Update(lambda v=2: v * 2)}}]
        """,
    )

    cfg = load(child)
    assert cfg["x"] == {"y": 4}
