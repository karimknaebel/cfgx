from pathlib import Path

import pytest

from cfgx import ConfigError, computed, final, load, value


def test_empty_and_nested_sources():
    assert load() == {}
    assert load((), ({"a": 1}, ({"b": 2},)), {"a": 3}) == {"a": 3, "b": 2}


@pytest.mark.parametrize("source", [[], ["base.py"], 1, None, object()])
def test_invalid_source(source):
    with pytest.raises(TypeError, match="sources"):
        load(source)


def test_computed_root():
    assert load(computed(lambda get: {"a": 1}), {"b": 2}) == {"a": 1, "b": 2}


@pytest.mark.parametrize("result", [[], (), 1, None])
def test_invalid_computed_root(result):
    with pytest.raises(ConfigError, match="plain dict"):
        load(computed(lambda get: result))


def test_file_includes_expand_in_place(tmp_path):
    (tmp_path / "base.py").write_text("config = {'x': 1, 'nested': {'a': 1}}")
    (tmp_path / "patch.py").write_text(
        "from cfgx import value\nconfig = {'x': value * 2}"
    )
    (tmp_path / "child.py").write_text(
        "config = ('base.py', 'patch.py', {'nested': {'b': 2}})"
    )
    assert load(tmp_path / "child.py") == {"x": 2, "nested": {"a": 1, "b": 2}}


def test_repeated_include_is_not_deduplicated(tmp_path):
    (tmp_path / "increment.py").write_text(
        "from cfgx import value\nconfig = {'x': value.default(0) + 1}"
    )
    assert load(tmp_path / "increment.py", tmp_path / "increment.py") == {"x": 2}


def test_nested_relative_includes(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "base.py").write_text("config = {'x': 1}")
    (tmp_path / "sub" / "child.py").write_text("config = '../base.py', {'y': 2}")
    assert load(tmp_path / "sub" / "child.py") == {"x": 1, "y": 2}


def test_include_cycle(tmp_path):
    (tmp_path / "a.py").write_text("config = 'b.py'")
    (tmp_path / "b.py").write_text("config = 'a.py'")
    with pytest.raises(ConfigError, match="include cycle.*a.py.*b.py.*a.py"):
        load(tmp_path / "a.py")


def test_file_must_define_config(tmp_path):
    (tmp_path / "empty.py").write_text("x = 1")
    with pytest.raises(ConfigError, match="must define 'config'"):
        load(tmp_path / "empty.py")


def test_each_load_executes_files_again(tmp_path):
    path = tmp_path / "config.py"
    path.write_text("config = {'x': 1}")
    assert load(path) == {"x": 1}
    path.write_text("config = {'x': 2}")
    assert load(path) == {"x": 2}


def test_later_file_and_override_are_visible_to_earlier_expression(tmp_path):
    path = tmp_path / "base.py"
    path.write_text("from cfgx import final\nconfig = {'x': 1, 'y': final.x * 2}")
    assert load(path, {"x": value * 3}, overrides=["x=4"]) == {"x": 4, "y": 8}


def test_snapshot_has_no_formulas():
    source = {"x": 1, "y": final.x * 2}
    assert load(source, overrides=["x=3"]) == {"x": 3, "y": 6}
    assert load(load(source), overrides=["x=3"]) == {"x": 3, "y": 2}


def test_pathlike_source(tmp_path):
    path = Path(tmp_path, "cfg.py")
    path.write_text("config = {'a': [1, (2, 3)]}")
    assert load(path) == {"a": [1, (2, 3)]}
