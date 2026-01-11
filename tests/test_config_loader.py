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
