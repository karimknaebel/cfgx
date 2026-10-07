import pytest

from cfgx import ConfigError, final, load
from cfgx.overrides import parse_path


@pytest.mark.parametrize(
    "path, expected",
    [
        ("a.b[0].c", ("a", "b", 0, "c")),
        ("a[-1]", ("a", -1)),
        ('a["literal.key"]', ("a", "literal.key")),
        ("a[(1, 2)]", ("a", (1, 2))),
        ("[None].a", (None, "a")),
    ],
)
def test_paths(path, expected):
    assert parse_path(path) == expected


@pytest.mark.parametrize(
    "path", ["", ".a", "a.", "a..b", "a.[0]", "a[", "a[0]b", "a[]"]
)
def test_invalid_paths(path):
    with pytest.raises((ValueError, SyntaxError)):
        parse_path(path)


def test_assignments_replace_leaf_dict_and_preserve_siblings():
    assert load({"a": {"b": {"x": 1}, "c": 2}}, overrides=["a.b={'y': 3}"]) == {
        "a": {"b": {"y": 3}, "c": 2},
    }


def test_override_values_and_expression_namespace():
    assert load(
        {"lr": 2, "derived": final.lr * 0.1},
        overrides=[
            "lr=expr:value * math.sqrt(4)",
            "lr=expr:value + 1",
            "text=hello",
            "yes=True",
            "none=None",
            "other=expr:previous.lr",
            "n=expr:final.lr * 2",
            "items=expr:computed(lambda get: [get(final.lr)])",
        ],
    ) == {
        "lr": 5,
        "derived": 0.5,
        "text": "hello",
        "yes": True,
        "none": None,
        "other": 5,
        "n": 10,
        "items": [5],
    }


def test_overrides_do_not_split_operators_inside_values_or_quoted_keys():
    assert load(overrides=["a=foo+=1", "b=bar!=2", '["x=y"]=3']) == {
        "a": "foo+=1",
        "b": "bar!=2",
        "x=y": 3,
    }


def test_sequence_edits_preserve_unresolved_sibling_definitions():
    assert load(
        {"items": [{"x": 1, "y": final.items[0].x * 2}, {"x": 3}]},
        overrides=["items[0].x=4"],
    ) == {"items": [{"x": 4, "y": 8}, {"x": 3}]}
    assert load({"items": (1, 2)}, overrides=["items[-1]=expr:value * 3"]) == {
        "items": (1, 6)
    }


def test_create_missing_dictionary_ancestors():
    assert load(overrides=["a.b.c=1"]) == {"a": {"b": {"c": 1}}}


def test_deletion_and_computed_deletion():
    assert load({"a": {"b": 1, "c": 2}}, overrides=["a.b!=", "a.c=expr:delete"]) == {
        "a": {}
    }
    assert load(overrides=["missing.deep!="]) == {}


@pytest.mark.parametrize("override", ["x[0]!=", "x[0]=expr:delete"])
def test_sequence_deletion_rejected(override):
    with pytest.raises(ConfigError, match="sequence"):
        load({"x": [1]}, overrides=[override])


@pytest.mark.parametrize("override", ["x[2]=3", "x[-3]=3"])
def test_sequence_override_bounds(override):
    with pytest.raises(IndexError):
        load({"x": [1, 2]}, overrides=[override])


def test_scalar_ancestor_rejected():
    with pytest.raises(TypeError, match="scalar"):
        load({"x": 1}, overrides=["x.y=2"])


def test_list_transform_with_map():
    assert load({"tags": ["a"]}, overrides=["tags=expr:value.map('x + [\"b\"]')"]) == {
        "tags": ["a", "b"],
    }


@pytest.mark.parametrize("override", ["x!=3", "x+=3", "x-=3", "x"])
def test_invalid_override(override):
    with pytest.raises(ValueError):
        load(overrides=[override])
