from pathlib import Path

import pytest

from cfgx import (
    ConfigError,
    MissingValueError,
    compose,
    computed,
    delete,
    final,
    include,
    load,
    previous,
    replace,
    value,
)
from cfgx.config import _expand
from cfgx.resolver import Resolver


def test_source_tuple_and_explicit_composition_are_equivalent():
    sources = ({"x": 2}, {"x": value * 3}, {"y": previous.x})
    assert load(sources) == load(compose(*sources)) == {"x": 6, "y": 6}
    assert load(compose(compose(*sources[:2]), sources[2])) == {"x": 6, "y": 6}
    assert load(compose()) == {}


@pytest.mark.parametrize("reverse", [False, True])
def test_nested_layers_advance_locally_without_advancing_siblings(reverse):
    patch = {
        "model": compose(
            {"width": value * 2, "batch_before": previous.batch_size},
            {"width": value + 1, "seen": previous.model.width},
        ),
        "batch_size": value * 2,
        "model_before": previous.model.width,
    }
    if reverse:
        patch = dict(reversed(patch.items()))
    assert load({"model": {"width": 3, "kept": True}, "batch_size": 8}, patch) == {
        "model": {"width": 7, "kept": True, "batch_before": 8, "seen": 6},
        "batch_size": 16,
        "model_before": 3,
    }


def test_sibling_compositions_have_independent_previous_views():
    assert load(
        {"a": {"x": 1}, "b": {"x": 2}},
        {
            "a": compose({"x": 3}, {"seen": previous.b.x}),
            "b": compose({"x": 4}, {"seen": previous.a.x}),
        },
    ) == {"a": {"x": 3, "seen": 2}, "b": {"x": 4, "seen": 1}}


def test_sibling_compositions_keep_metadata_growth_linear():
    entries = []
    for count in (100, 200):
        source = {i: {"settings": compose({"a": 1})} for i in range(count)}
        resolver = Resolver(_expand(source, Path.cwd()), _expand)
        assert resolver.resolve() == {i: {"settings": {"a": 1}} for i in range(count)}
        entries.append(sum(len(layer.containers) for layer in resolver.layers))
    assert entries[1] < 3 * entries[0]


def test_nested_previous_root_includes_prior_local_contributions():
    assert load(
        {"outer": 1, "model": {"old": 2}},
        {
            "outer": 9,
            "model": compose(
                {"new": 3},
                {"snapshot": previous, "local": previous(1)},
            ),
        },
    ) == {
        "outer": 9,
        "model": {
            "old": 2,
            "new": 3,
            "snapshot": {"outer": 1, "model": {"old": 2, "new": 3}},
            "local": {"old": 2, "new": 3},
        },
    }


def test_deeply_nested_compositions_share_final_and_keep_origins():
    assert load(
        {"scale": 2},
        {
            "model": compose(
                {"block": {"width": 3}},
                {
                    "block": compose(
                        {"width": value * previous.scale},
                        {
                            "derived": final(1).width * final.scale,
                            "inherited_width": previous(1).width,
                        },
                    )
                },
            ),
        },
        {"scale": 4, "model": {"block": {"width": 10}}},
    ) == {
        "scale": 4,
        "model": {"block": {"width": 10, "derived": 40, "inherited_width": 6}},
    }


def test_value_positions_preserve_strings_paths_and_tuples():
    assert load(
        {
            "string": "does-not-exist.py",
            "path": Path("does-not-exist.py"),
            "tuple": ({"x": 1}, {"x": 2}),
            "computed": computed(lambda get: ("does-not-exist.py", {"x": 1})),
        }
    ) == {
        "string": "does-not-exist.py",
        "path": Path("does-not-exist.py"),
        "tuple": ({"x": 1}, {"x": 2}),
        "computed": ("does-not-exist.py", {"x": 1}),
    }


@pytest.mark.parametrize("container", [list, tuple])
def test_composition_in_sequence_elements(container):
    assert load(
        {"items": container([{"x": 2}, {"x": 9}])},
        {
            "items": container(
                [
                    compose({"x": value * 3}, {"seen": previous.items[0].x}),
                    {"x": value + 1},
                ]
            )
        },
    ) == {"items": container([{"x": 6, "seen": 6}, {"x": 10}])}


def test_composition_under_new_sequence_ancestors():
    assert load(
        {"items": [0, compose({"x": 2}, {"x": value * 3, "seen": previous.items[1].x})]}
    ) == {"items": [0, {"x": 6, "seen": 2}]}


def test_local_previous_does_not_fill_missing_sequence_siblings():
    assert load(
        {"items": [0, compose({"x": 2}, {"sibling": previous.items[0].exists()})]}
    ) == {"items": [0, {"x": 2, "sibling": False}]}
    with pytest.raises(MissingValueError, match=r"items\[0\]"):
        load({"items": [0, compose({"x": 2}, {"snapshot": previous.items})]})


def test_replacements_inside_composition_discard_inherited_fields():
    assert load(
        {"model": {"old": 1}},
        {
            "model": compose(
                {"a": 2},
                replace({"b": 3}),
                {"c": 4, "keys_before": previous.model.keys()},
            )
        },
        {"model": {"d": 5}},
    ) == {"model": {"b": 3, "c": 4, "keys_before": ["b"], "d": 5}}


@pytest.mark.parametrize("barrier", [0, None, [], (), delete, replace({})])
def test_composition_preserves_nested_inheritance_barriers(barrier):
    base = {"model": {"nested": {"old": 1}, "kept": True}}
    first, second = {"nested": barrier}, {"nested": {"new": 2}}
    assert (
        load(base, {"model": compose(first, second)})
        == load(base, {"model": first}, {"model": second})
        == {"model": {"nested": {"new": 2}, "kept": True}}
    )


def test_replacement_around_composition_can_still_read_inherited_values():
    assert load(
        {"model": {"width": 3, "old": 1}},
        {"model": replace(compose({"width": value * 2}, {"width": value + 1}))},
    ) == {"model": {"width": 7}}


def test_source_replacements_and_empty_compositions():
    assert load({"old": 1}, replace({"new": 2}), {"later": 3}) == {
        "new": 2,
        "later": 3,
    }
    assert load({"x": {"old": 1}}, {"x": compose()}) == {"x": {"old": 1}}
    assert load({"x": {"old": 1}}, {"x": replace(compose())}) == {"x": {}}
    assert load({"x": compose()}) == {"x": {}}


def test_compositions_preserve_deletions_and_reference_consumption():
    assert load(
        {"model": {"gone": 1}},
        {"model": compose({"gone": delete}, {"kept": 2})},
        {"copy": replace(final.model)},
        {"copy": {"extra": 3}},
    ) == {"model": {"kept": 2}, "copy": {"kept": 2, "extra": 3}}


def test_computed_can_return_compositions_without_reinterpreting_data():
    assert load(
        computed(lambda get: compose({"x": 2}, {"x": value * 3})),
        {
            "model": computed(
                lambda get: compose({"width": final.x}, {"width": value + 1})
            )
        },
    ) == {"x": 6, "model": {"width": 7}}


@pytest.mark.parametrize("source", [1, [], delete, replace(1)])
def test_composition_sources_must_produce_dicts(source):
    with pytest.raises((TypeError, ConfigError), match="sources|plain dict"):
        load({"model": compose(source)})


@pytest.mark.parametrize("result", ["missing.py", ({"x": 1},), [], delete])
def test_expression_results_are_not_implicit_sources(result):
    with pytest.raises(ConfigError, match="plain dict"):
        load({"model": compose(computed(lambda get: result))})


@pytest.mark.parametrize(
    "wrapper", [lambda x: x, replace, lambda x: computed(lambda get: x)]
)
def test_local_computed_dictionary_reads_later_contributions_once(wrapper):
    calls = []

    def produce(get):
        calls.append(1)
        return {"derived": get(final.model.width) * 2}

    assert load({"model": wrapper(compose(computed(produce), {"width": 3}))}) == {
        "model": {"derived": 6, "width": 3}
    }
    assert calls == [1]


@pytest.mark.parametrize(
    ("query", "expected"),
    [(final.model.exists(), True), (final.keys(), ["model"]), (final.len(), 1)],
)
def test_composed_presence_does_not_evaluate_earlier_contributions(query, expected):
    calls = []

    def produce(get):
        calls.append(1)
        return {"seen": get(query)}

    assert load({"model": compose(computed(produce), {})}) == {
        "model": {"seen": expected}
    }
    assert calls == [1]


def test_composed_presence_dependency_cycles_have_traces():
    with pytest.raises(ConfigError, match=r"Dependency cycle:.*layer.*->"):
        load(
            {
                "model": compose(
                    computed(lambda get: {"seen": get(final.model.exists())})
                )
            }
        )


@pytest.mark.parametrize("wrapper", [lambda x: x, lambda x: compose(x)])
def test_structural_queries_do_not_evaluate_present_child_contents(wrapper):
    assert load(
        {"x": {"a": 1}},
        {
            "x": wrapper(
                {
                    "count": final.x.len(),
                    "keys": final.x.keys(),
                    "exists": final.x.exists(),
                }
            )
        },
    ) == {
        "x": {
            "a": 1,
            "count": 4,
            "keys": ["a", "count", "keys", "exists"],
            "exists": True,
        }
    }


def test_nested_composition_dependency_cycles_have_traces():
    with pytest.raises(ConfigError, match=r"Dependency cycle:.*layer.*->"):
        load(
            {
                "model": compose(
                    {"x": final.model.y}, {"y": value.default(final.model.x)}
                )
            }
        )


def test_expression_return_cycles_have_traces():
    expression = computed(lambda get: expression)
    with pytest.raises(ConfigError, match=r"Dependency cycle:.*layer.*->"):
        load({"x": expression})


def test_shared_compositions_bind_independently():
    source = compose({"width": 2}, {"derived": final(1).width * 2, "items": []})
    result = load({"a": source, "b": source}, {"b": {"width": 3}})
    assert result["a"]["derived"] == 4
    assert result["b"]["derived"] == 6
    result["a"]["items"].append(1)
    assert result["b"]["items"] == []


def test_raw_container_cycles_through_composition_are_rejected():
    source = {}
    source["nested"] = compose(source)
    with pytest.raises(ConfigError, match="Cyclic Python container"):
        load(source)


def test_opaque_containers_are_not_expanded(tmp_path):
    class CustomList(list):
        pass

    source = CustomList([include(tmp_path / "missing.py"), compose({"x": 1})])
    assert load({"opaque": source})["opaque"] is source


def test_explicit_and_implicit_includes_are_equivalent(tmp_path):
    path = tmp_path / "base.py"
    path.write_text("from cfgx import value\nconfig = ({'x': 2}, {'x': value * 3})")
    assert (
        load(path, {"x": value + 1})
        == load(include(path), {"x": value + 1})
        == {"x": 7}
    )
    assert load({"model": include(path)}) == {"model": {"x": 6}}


def test_nested_includes_bind_relative_references_at_destination(tmp_path):
    path = tmp_path / "model.py"
    path.write_text(
        "from cfgx import final\n"
        "config = {'width': 3, 'derived': final(1).width * final.scale}\n"
    )
    assert load(
        {"scale": 2, "a": include(path), "b": include(path)},
        {"b": {"width": 5}},
    ) == {"scale": 2, "a": {"width": 3, "derived": 6}, "b": {"width": 5, "derived": 10}}


def test_nested_and_computed_includes_keep_declaring_file_directory(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "base.py").write_text("config = {'width': 3}")
    (sub / "model.py").write_text(
        "from cfgx import compose, value\n"
        "config = compose('base.py', {'width': value * 2})\n"
    )
    (sub / "run.py").write_text(
        "from cfgx import compose, computed, include, final\n"
        "config = {\n"
        "    'file': 'model.py',\n"
        "    'a': include('model.py'),\n"
        "    'b': computed(lambda get: include(get(final.file))),\n"
        "    'c': computed(lambda get: compose('model.py', {'extra': 1})),\n"
        "}\n"
    )
    assert load(sub / "run.py") == {
        "file": "model.py",
        "a": {"width": 6},
        "b": {"width": 6},
        "c": {"width": 6, "extra": 1},
    }


def test_nested_repeated_includes_apply_in_order_and_execute_again(tmp_path):
    path = tmp_path / "increment.py"
    path.write_text("from cfgx import value\nconfig = {'x': value.default(0) + 1}")
    source = {"model": compose(include(path), include(path))}
    assert load(source) == {"model": {"x": 2}}
    path.write_text("from cfgx import value\nconfig = {'x': value.default(0) + 2}")
    assert load(source) == {"model": {"x": 4}}


@pytest.mark.parametrize("dynamic", [False, True])
def test_nested_include_cycles_report_file_chain(tmp_path, dynamic):
    (tmp_path / "a.py").write_text(
        "from cfgx import include\nconfig = {'x': include('b.py')}"
    )
    (tmp_path / "b.py").write_text(
        "from cfgx import computed, include\nconfig = "
        + ("computed(lambda get: include('a.py'))" if dynamic else "include('a.py')")
    )
    with pytest.raises(ConfigError, match="include cycle.*a.py.*b.py.*a.py"):
        load(tmp_path / "a.py")


def test_computed_include_can_use_a_later_file_override(tmp_path):
    path = tmp_path / "model.py"
    path.write_text("config = {'width': 3}")
    assert load(
        {"model": computed(lambda get: include(get(final.model_path)))},
        {"model_path": path},
    ) == {"model": {"width": 3}, "model_path": path}


def test_computed_include_runs_once_across_structural_and_value_reads(tmp_path):
    path = tmp_path / "model.py"
    path.write_text("config = {'width': 3}")
    calls = []

    def produce(get):
        calls.append(1)
        return include(path)

    assert load(
        {
            "keys": final.model.keys(),
            "model": computed(produce),
            "size": final.model.len(),
            "copy": final.model,
        }
    ) == {"keys": ["width"], "model": {"width": 3}, "size": 1, "copy": {"width": 3}}
    assert calls == [1]


def test_include_in_unused_default_is_not_loaded(tmp_path):
    assert load(
        {"model": {}, "copy": final.model.default(include(tmp_path / "missing.py"))}
    ) == {
        "model": {},
        "copy": {},
    }


def test_missing_previous_inside_composition_is_still_an_error():
    with pytest.raises(MissingValueError):
        load({"model": compose({"x": previous.model.missing})})
