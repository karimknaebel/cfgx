from pprint import pformat

import pytest

from cfgx.config import (
    Lazy,
    Update,
    apply_overrides,
    dump,
    dumps,
    format,
    infer_type,
    parse_key_path,
    resolve_lazy,
    set_nested,
    update_nested,
)

### --- parse_key_path tests --- ###


@pytest.mark.parametrize(
    "input_path, expected",
    [
        ("a.b.c", ["a", "b", "c"]),
        ("a[0].b", ["a", 0, "b"]),
        ("x[1][2].y", ["x", 1, 2, "y"]),
        ("x[-1].y", ["x", -1, "y"]),
        ("a", ["a"]),
        ("a[0]", ["a", 0]),
    ],
)
def test_parse_key_path(input_path, expected):
    assert parse_key_path(input_path) == expected


### --- infer_type tests --- ###


@pytest.mark.parametrize(
    "input_val, expected",
    [
        ("123", 123),
        ("3.14", 3.14),
        ("3e-4", 3e-4),
        ("True", True),
        ("None", None),
        ("[1, 2]", [1, 2]),
        ("{'x': 5}", {"x": 5}),
        ("{'x': [{'a': 1}, {'b': 2}]}", {"x": [{"a": 1}, {"b": 2}]}),
        ("'hello'", "hello"),
        ("unquoted_string", "unquoted_string"),  # fallback
    ],
)
def test_infer_type(input_val, expected):
    assert infer_type(input_val) == expected


def test_infer_type_lazy():
    value = infer_type("lazy:c.trainer.max_steps * 0.1")
    assert isinstance(value, Lazy)


def test_infer_type_update():
    value = infer_type("update:v * 0.1")
    assert isinstance(value, Update)


def test_infer_type_lazy_expression_resolves():
    cfg = {"trainer": {"max_steps": 1000}}
    apply_overrides(cfg, ["warmup_steps=lazy:c.trainer.max_steps * 0.1"])
    resolve_lazy(cfg)
    assert cfg["warmup_steps"] == 100


def test_infer_type_update_expression_resolves():
    cfg = {"trainer": {"max_steps": 1000}}
    apply_overrides(cfg, ["trainer.max_steps=update:v * 0.1"])
    assert cfg["trainer"]["max_steps"] == 100.0


### --- resolve_lazy tests --- ###


def test_resolve_lazy_skips_tuples():
    cfg = {"a": (Lazy("1 + 1"),), "b": Lazy("1 + 1")}
    resolve_lazy(cfg)
    assert isinstance(cfg["a"][0], Lazy)
    assert cfg["b"] == 2


### --- set_nested tests --- ###


def test_set_nested_dict():
    cfg = {}
    set_nested(cfg, ["a", "b", "c"], 42)
    assert cfg == {"a": {"b": {"c": 42}}}


def test_set_nested_list():
    cfg = {}
    set_nested(cfg, ["a", "b", 0, "x"], "hi")
    assert cfg == {"a": {"b": [{"x": "hi"}]}}


def test_set_nested_expand_list():
    cfg = []
    set_nested(cfg, [2], "foo")
    assert cfg == [None, None, "foo"]


def test_set_nested_last_list_index():
    cfg = ["a", "b"]
    set_nested(cfg, [-1], "c")
    assert cfg == ["a", "c"]


### --- apply_overrides tests --- ###


def test_apply_overrides_basic():
    cfg = {"a": {"b": 1}}
    overrides = ["a.b=42"]
    updated = apply_overrides(cfg, overrides)
    assert updated == {"a": {"b": 42}}


def test_apply_overrides_nested_list():
    cfg = {}
    overrides = ["x.y[0].name=conv", "x.y[1].attrs.out_channels=64"]
    updated = apply_overrides(cfg, overrides)
    assert updated == {"x": {"y": [{"name": "conv"}, {"attrs": {"out_channels": 64}}]}}


def test_apply_overrides_mix_types():
    cfg = {}
    overrides = ["flag=True", "threshold=0.75", "num_layers=5", "name=model_v1"]
    updated = apply_overrides(cfg, overrides)
    assert updated == {
        "flag": True,
        "threshold": 0.75,
        "num_layers": 5,
        "name": "model_v1",
    }


def test_apply_overrides_value_with_operator_tokens():
    cfg = {}
    overrides = ["foo=bar+=3", "baz=qux-=1", "zip=zot!=2"]
    updated = apply_overrides(cfg, overrides)
    assert updated == {"foo": "bar+=3", "baz": "qux-=1", "zip": "zot!=2"}


def test_apply_overrides_append():
    cfg = {"existing_list": ["a", "b"], "nested": {"items": []}}
    overrides = [
        "existing_list+=c",
        "new_list+=1",
        "new_list+=2",
        "new_list+=3",
        "nested.items+=foo",
        "nested.items+=bar",
    ]
    updated = apply_overrides(cfg, overrides)
    assert updated == {
        "existing_list": ["a", "b", "c"],
        "new_list": [1, 2, 3],
        "nested": {"items": ["foo", "bar"]},
    }


def test_apply_overrides_append_non_list_raises():
    cfg = {"a": {"b": 1}}
    with pytest.raises(ValueError, match="not a list"):
        apply_overrides(cfg, ["a.b+=2"])


def test_apply_overrides_append_none_target_raises():
    cfg = {"items": [None]}
    with pytest.raises(ValueError, match="not a list"):
        apply_overrides(cfg, ["items[0]+=1"])


def test_apply_overrides_set_negative_index():
    cfg = {"layers": ["conv1", "conv2"]}
    updated = apply_overrides(cfg, ["layers[-1]=conv3"])
    assert updated == {"layers": ["conv1", "conv3"]}


def test_apply_overrides_set_negative_index_out_of_bounds_raises():
    cfg = {"layers": ["conv1", "conv2"]}
    with pytest.raises(IndexError):
        apply_overrides(cfg, ["layers[-3]=conv3"])


def test_apply_overrides_append_negative_index():
    cfg = {"pipelines": [["a"], ["b"]]}
    updated = apply_overrides(cfg, ["pipelines[-1]+=c"])
    assert updated == {"pipelines": [["a"], ["b", "c"]]}


def test_apply_overrides_append_negative_index_out_of_bounds_raises():
    cfg = {"pipelines": []}
    with pytest.raises(IndexError):
        apply_overrides(cfg, ["pipelines[-1]+=c"])


def test_update_nested_missing_without_default_raises():
    cfg = {}
    with pytest.raises(TypeError):
        update_nested(cfg, ["a"], Update(lambda v: v + 1))


def test_apply_overrides_update_missing_string_raises():
    cfg = {}
    with pytest.raises(TypeError):
        apply_overrides(cfg, ["a=update:v + [1]"])


def test_update_nested_missing_with_default():
    cfg = {}
    update_nested(cfg, ["items"], Update(lambda v=[]: v + ["x"]))
    assert cfg["items"] == ["x"]


def test_update_nested_returning_update_is_unchecked():
    cfg = {"a": 1}
    update_nested(cfg, ["a"], Update(lambda v: Update(lambda w: w + 1)))
    assert isinstance(cfg["a"], Update)


def test_delete_dict_key():
    cfg = {"a": {"b": {"c": 123, "d": 456}}}
    overrides = ["a.b.c!="]
    updated = apply_overrides(cfg, overrides)
    assert updated == {"a": {"b": {"d": 456}}}


def test_delete_with_value_raises():
    cfg = {"a": {"b": 1}}
    with pytest.raises(ValueError, match="Delete overrides must not include a value"):
        apply_overrides(cfg, ["a.b!=2"])


def test_delete_list_index():
    cfg = {"layers": ["conv1", "conv2", "conv3"]}
    overrides = ["layers[1]!="]
    updated = apply_overrides(cfg, overrides)
    assert updated == {"layers": ["conv1", "conv3"]}


def test_delete_list_negative_index():
    cfg = {"layers": ["conv1", "conv2", "conv3"]}
    overrides = ["layers[-1]!="]
    updated = apply_overrides(cfg, overrides)
    assert updated == {"layers": ["conv1", "conv2"]}


def test_delete_missing_path_noop():
    cfg = {"a": {"b": 1}}
    overrides = ["a.c!=", "missing!="]
    updated = apply_overrides(cfg, overrides)
    assert updated == {"a": {"b": 1}}


@pytest.mark.parametrize(
    "cfg, override",
    [
        ({"a": {"b": 1}}, "a.b.c!="),
        ({"items": ("a",)}, "items[0]!="),
    ],
)
def test_delete_type_mismatch_raises(cfg, override):
    with pytest.raises(TypeError):
        apply_overrides(cfg, [override])


def test_delete_list_value():
    cfg = {"tags": ["debug", "train", "final"]}
    overrides = ['tags-="train"']
    updated = apply_overrides(cfg, overrides)
    assert updated == {"tags": ["debug", "final"]}


def test_delete_list_value_non_list_raises():
    cfg = {"tags": "train"}
    with pytest.raises(ValueError, match="not a list"):
        apply_overrides(cfg, ['tags-="train"'])


def test_remove_missing_path_noop():
    cfg = {"lists": [["a"]]}
    overrides = ["missing.path-='a'", "lists[1]-='a'"]
    updated = apply_overrides(cfg, overrides)
    assert updated == {"lists": [["a"]]}


def test_delete_int_key_in_dict():
    cfg = {"items": {0: "a", 1: "b"}}
    updated = apply_overrides(cfg, ["items[0]!="])
    assert updated == {"items": {1: "b"}}


def test_remove_int_key_in_dict():
    cfg = {"items": {0: ["a", "b"]}}
    updated = apply_overrides(cfg, ["items[0]-='b'"])
    assert updated == {"items": {0: ["a"]}}


def test_set_special_key():
    cfg = {"a": {"b": 1}}
    overrides = ["a.*=[1,2,3]"]
    updated = apply_overrides(cfg, overrides)
    assert updated == {"a": {"*": [1, 2, 3], "b": 1}}


def test_set_through_tuple_item():
    cfg = {"items": ({"a": 1},)}
    updated = apply_overrides(cfg, ["items[0].a=2"])
    assert updated == {"items": ({"a": 2},)}


def test_set_int_key_in_dict():
    cfg = {"items": {}}
    updated = apply_overrides(cfg, ["items[0]=1"])
    assert updated == {"items": {0: 1}}


def test_combined_deletes_and_adds():
    cfg = {
        "model": {
            "layers": ["conv", "bn", "relu"],
            "dropout": 0.5,
        },
        "tags": ["baseline"],
    }
    overrides = [
        "model.dropout!=",  # delete dict key
        "model.layers[1]!=",  # delete list index
        "model.layers+=maxpool",  # append
        "tags+=debug",  # append
        'tags-="baseline"',  # remove value
    ]
    updated = apply_overrides(cfg, overrides)
    assert updated == {
        "model": {"layers": ["conv", "relu", "maxpool"]},
        "tags": ["debug"],
    }


### --- format tests --- ###


def test_format_simple_dict():
    cfg = {
        "model": {
            "encoder": {"channels": 64},
            "head": {"in_channels": 64, "out_channels": 10},
        },
        "optimizer": {"type": "adam", "lr": 3e-4},
        "trainer": {"max_steps": 50_000},
    }
    formatted = format(cfg)
    assert formatted == pformat(cfg, width=88, sort_dicts=False)


def test_format_pretty():
    cfg = {"b": 2, "a": 1}
    formatted = format(cfg, format="pretty")
    assert formatted == "{'b': 2, 'a': 1}"


def test_format_sort_keys():
    cfg = {"b": 2, "a": 1}
    formatted = format(cfg, format="raw", sort_keys=True)
    assert formatted == "{'a': 1, 'b': 2}"


def test_format_sort_keys_skips_tuples():
    cfg = {"a": ({"b": 2, "a": 1},)}
    formatted = format(cfg, format="raw", sort_keys=True)
    assert formatted == "{'a': ({'b': 2, 'a': 1},)}"


def test_dump_simple_dict(tmp_path):
    cfg = {
        "model": {
            "encoder": {"channels": 64},
            "head": {"in_channels": 64, "out_channels": 10},
        },
        "optimizer": {"type": "adam", "lr": 3e-4},
        "trainer": {"max_steps": 50_000},
    }
    snapshot_path = tmp_path / "config_snapshot.py"
    with open(snapshot_path, "w") as f:
        dump(cfg, f)
    with open(snapshot_path, "r") as f:
        content = f.read()
    expected = "config = " + pformat(cfg, width=88, sort_dicts=False) + "\n"
    assert content == expected


def test_dumps_simple_dict():
    cfg = {"a": 1}
    expected = "config = " + pformat(cfg, width=88, sort_dicts=False) + "\n"
    assert dumps(cfg) == expected
