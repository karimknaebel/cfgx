import pytest

from cfgx import ConfigError, computed, delete, final, load, replace
from cfgx.overrides import parse_path


@pytest.mark.parametrize(
    "path, expected",
    [
        ("a.b.c", ("a", "b", "c")),
        ('a["literal.key"]', ("a", "literal.key")),
        ('["root.key"].a', ("root.key", "a")),
        ('a["0"]', ("a", "0")),
        ('a["x[y]"]', ("a", "x[y]")),
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


@pytest.mark.parametrize("path", ["", "branch."])
@pytest.mark.parametrize(
    "overrides", [["x=3", "y=4"], ["y=4", "x=3"], ["x=1", "y=4", "x=3"]]
)
def test_computed_container_can_read_override_contributions(path, overrides):
    calls = []
    ref = final.branch if path else final

    def produce(get):
        calls.append(None)
        return {"answer": get(ref.x) + get(ref.y), "kept": 5}

    source = computed(produce)
    expected = {"answer": 7, "kept": 5, "x": 3, "y": 4}
    if path == "branch.":
        source, expected = {"branch": source}, {"branch": expected}
    assert load(source, overrides=[path + item for item in overrides]) == expected
    assert calls == [None]


def test_computed_root_can_read_expression_override():
    assert load(
        computed(lambda get: {"answer": get(final.y)}),
        overrides=["y=expr:final.x * 2", "x=3"],
    ) == {"answer": 6, "y": 6, "x": 3}


def test_overrides_preserve_real_dependency_cycles():
    with pytest.raises(ConfigError, match="Dependency cycle"):
        load(
            computed(lambda get: {"answer": get(final.x)}),
            overrides=["x=expr:final.answer"],
        )


@pytest.mark.parametrize("result", [0, [], ()])
def test_dictionary_override_replaces_computed_non_dictionary_ancestor(result):
    def produce(get):
        assert get(final.branch.x) == 3
        return result

    assert load({"branch": computed(produce)}, overrides=["branch.x=3"]) == {
        "branch": {"x": 3}
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


@pytest.mark.parametrize("key", ["0", "-1", "None", "True", "(1, 2)"])
@pytest.mark.parametrize("operation", ["=3", "!="])
def test_non_string_override_paths_rejected(key, operation):
    with pytest.raises(ValueError, match="string dictionary keys"):
        load(overrides=[f"items[{key}]{operation}"])


def test_create_missing_dictionary_ancestors():
    assert load(overrides=["a.b.c=1"]) == {"a": {"b": {"c": 1}}}


def test_deletion_and_computed_deletion():
    assert load({"a": {"b": 1, "c": 2}}, overrides=["a.b!=", "a.c=expr:delete"]) == {
        "a": {}
    }


@pytest.mark.parametrize(
    "override",
    [
        "a.b.c!=",
        "a.b.c=expr:delete",
        "a.b.c=expr:computed(lambda get: delete)",
    ],
)
def test_deletion_establishes_dictionary_ancestors(override):
    assert load(overrides=[override]) == {"a": {"b": {}}}
    assert load({"a": {"keep": 1}}, overrides=[override]) == {"a": {"keep": 1, "b": {}}}


def test_deletion_does_not_evaluate_the_inherited_leaf():
    assert load(
        {
            "a": {
                "b": computed(lambda get: 1 / 0),
                "kept": final.a.b.exists(),
            }
        },
        overrides=["a.b!="],
    ) == {"a": {"kept": False}}


@pytest.mark.parametrize("override", ["x=expr:[delete]", "expr:{'x': (delete,)}"])
def test_sequence_deletion_rejected(override):
    with pytest.raises(ConfigError, match="sequence"):
        load(overrides=[override])


@pytest.mark.parametrize("ancestor", [1, None, [1, 2], (1, 2)])
def test_dictionary_override_replaces_non_dictionary_ancestor(ancestor):
    assert load({"x": ancestor}, overrides=["x.y=2"]) == {"x": {"y": 2}}


@pytest.mark.parametrize(
    "override, layer",
    [
        ("a.b={'new': 3}", {"a": {"b": replace({"new": 3})}}),
        ("a.b!=", {"a": {"b": delete}}),
        ("a.b=expr:delete", {"a": {"b": replace(delete)}}),
    ],
)
def test_shorthand_matches_ordinary_contributions(override, layer):
    source = {"a": {"b": {"old": 1}, "keep": 2}}
    assert load(source, overrides=[override]) == load(source, layer)


def test_whole_layer_expressions_merge_and_preserve_layer_boundaries():
    assert load(
        {"x": 1, "options": {"a": 1}},
        overrides=[
            "expr:dict(x=value + 1, before=previous.x)",
            "x=expr:value * 3",
            "expr:{'options': {'b': final.x}}",
        ],
    ) == {"x": 6, "before": 1, "options": {"a": 1, "b": 6}}


def test_whole_layer_expressions_support_non_string_keys():
    assert load(
        {"options": {0: {"a": 1}, (1, 2): 3}},
        overrides=["expr:{'options': {0: {'b': 2}, (1, 2): delete, None: True}}"],
    ) == {"options": {0: {"a": 1, "b": 2}, None: True}}


def test_computed_whole_layer_can_read_later_override():
    assert load(
        {"kept": 1},
        overrides=["expr:computed(lambda get: {'answer': get(final.x)})", "x=3"],
    ) == {"kept": 1, "answer": 3, "x": 3}


@pytest.mark.parametrize("expression", ["[]", "()", "1", "None", "delete", "'cfg.py'"])
def test_invalid_whole_layer_value(expression):
    with pytest.raises(TypeError, match="whole-layer expression"):
        load(overrides=[f"expr:{expression}"])


def test_invalid_computed_whole_layer_value():
    with pytest.raises(ConfigError, match="plain dict"):
        load(overrides=["expr:computed(lambda get: [])"])


@pytest.mark.parametrize(
    "override",
    ["model=include:model.py", "model=expr:include('model.py')"],
)
def test_include_override_replaces_and_uses_working_directory(
    tmp_path, monkeypatch, override
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "defaults.py").write_text("config = {'width': 3}")
    (tmp_path / "model.py").write_text(
        "from cfgx import final, value\n"
        "config = 'defaults.py', {'width': value * 2, 'derived': final.model.width * 2}"
    )
    assert load({"model": {"old": 1}}, overrides=[override, "model.width=10"]) == {
        "model": {"width": 10, "derived": 20}
    }


def test_whole_layer_include_and_compose_overrides_merge(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "model.py").write_text("config = {'width': 3}")
    assert load(
        {"model": {"old": 1}, "count": 2},
        overrides=[
            "expr:{'model': include('model.py')}",
            "expr:compose({'count': value * 2}, {'count': value + 1})",
        ],
    ) == {"model": {"old": 1, "width": 3}, "count": 5}
    assert load(overrides=["expr:include('model.py')"]) == {"width": 3}
    assert load({"old": 1}, overrides=["expr:replace({'new': 2})"]) == {"new": 2}


def test_compose_assignment_replaces_target_but_reads_previous():
    assert load(
        {"model": {"old": 1, "width": 3}},
        overrides=["model=expr:compose({'width': value * 2}, {'width': value + 1})"],
    ) == {"model": {"width": 7}}


def test_include_prefix_can_be_quoted_as_string_data():
    assert load(overrides=["path='include:missing.py'"]) == {
        "path": "include:missing.py"
    }


def test_list_transform_with_map():
    assert load({"tags": ["a"]}, overrides=["tags=expr:value.map('x + [\"b\"]')"]) == {
        "tags": ["a", "b"],
    }


def test_sequence_transform_preserves_untouched_elements_and_source():
    source = {"items": [{"width": 64, "keep": 1}, {"width": 32}]}
    assert load(
        source,
        overrides=['items=expr:value.map(lambda x: [{**x[0], "width": 128}, *x[1:]])'],
    ) == {"items": [{"width": 128, "keep": 1}, {"width": 32}]}
    assert source == {"items": [{"width": 64, "keep": 1}, {"width": 32}]}
    assert load(
        {"items": (1, 2)},
        overrides=["items=expr:value.map(lambda x: (*x[:-1], x[-1] * 3))"],
    ) == {"items": (1, 6)}


@pytest.mark.parametrize("override", ["x!=3", "x+=3", "x-=3", "x"])
def test_invalid_override(override):
    with pytest.raises(ValueError):
        load(overrides=[override])
