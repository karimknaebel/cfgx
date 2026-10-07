import math

import pytest

from cfgx import (
    ConfigError,
    Expression,
    MissingValueError,
    computed,
    delete,
    final,
    load,
    previous,
    replace,
    value,
)


def test_computed_dictionary_merges_with_later_patch():
    assert load(
        {"branch": computed(lambda get: {"a": 1, "b": 2})},
        {"branch": {"a": 3}},
    ) == {"branch": {"a": 3, "b": 2}}


def test_computed_dictionary_merges_with_earlier_dictionary():
    assert load({"x": {"a": 1}}, {"x": computed(lambda get: {"b": 2})}) == {
        "x": {"a": 1, "b": 2},
    }


@pytest.mark.parametrize("barrier", [0, [], (), delete, replace({})])
def test_non_mapping_contribution_breaks_inheritance(barrier):
    assert load({"x": {"a": 1}}, {"x": barrier}, {"x": {"b": 2}}) == {"x": {"b": 2}}


def test_replace_can_read_previous_and_be_patched_later():
    assert load(
        {"x": {"a": 1, "b": 2}},
        {"x": replace(value.map(lambda x: {"a": x["a"] + 1}))},
        {"x": {"c": 3}},
    ) == {"x": {"a": 2, "c": 3}}


def test_callback_can_return_merge_instructions():
    assert load(
        {"x": {"a": 1}, "y": 2},
        {
            "x": computed(lambda get: replace({"b": 3})),
            "y": computed(lambda get: delete),
        },
    ) == {"x": {"b": 3}}
    assert load({"x": 1}, {"x": replace(delete)}) == {}


def test_previous_definitions_remain_bound_to_final():
    assert load(
        {"steps": 100, "cooldown": final.steps // 10},
        {"steps": 200, "cooldown": value * 2},
    ) == {"steps": 200, "cooldown": 40}


def test_previous_reads_before_whole_layer():
    assert load(
        {"x": 1, "y": 2},
        {"x": previous.y, "y": previous.x},
    ) == {"x": 2, "y": 1}


def test_relative_references_and_reused_declarations():
    block = {"width": 64, "hidden": final(1).width * 2}
    assert load({"a": block, "b": block}, {"b": {"width": 128}}) == {
        "a": {"width": 64, "hidden": 128},
        "b": {"width": 128, "hidden": 256},
    }
    assert load({"x": {"a": 2}}, {"x": {"a": 3, "b": previous(1).a}}) == {
        "x": {"a": 3, "b": 2},
    }


def test_relative_paths_count_sequences_and_do_not_count_operators():
    assert load(
        {"x": [{"a": 2, "b": final(1).a.map("x + 1") * 2}], "y": 4, "z": [final(2).y]}
    ) == {"x": [{"a": 2, "b": 6}], "y": 4, "z": [4]}


def test_nested_returned_expressions_bind_at_output_locations():
    assert load({"branch": computed(lambda get: {"x": 4, "y": final(1).x * 2})}) == {
        "branch": {"x": 4, "y": 8},
    }
    assert load({"x": {"a": 2}}, {"x": computed(lambda get: {"a": value * 3})}) == {
        "x": {"a": 6},
    }


def test_parent_errors():
    with pytest.raises(ConfigError, match="above the root"):
        load({"x": final(2)})
    for parent in [-1, 1.5, True, "x"]:
        with pytest.raises(ValueError):
            final(parent)
    assert load({"x": 1, "y": final(parent=None).x}) == {"x": 1, "y": 1}


def test_keys_can_include_method_names_and_non_strings():
    assert load(
        {
            "map": 2,
            "default": 3,
            "keys": 4,
            "len": 5,
            "exists": 6,
            ("a", "b"): 7,
            None: 8,
            "result": [
                final.map.map("x * 2"),
                final.default,
                final.keys,
                final.len,
                final.exists,
                final[("a", "b")],
                final[None],
            ],
        }
    )["result"] == [4, 3, 4, 5, 6, 7, 8]


def test_arithmetic_comparisons_and_string_maps():
    assert load(
        {
            "a": 4,
            "b": [
                10 - final.a,
                2**final.a,
                -final.a,
                abs(-final.a),
                final.a >= 3,
                final.a == 4,
                final.a.map("math.sqrt(x)") * 2,
            ],
        }
    )["b"] == [6, 16, -4, 4, True, True, 4]
    assert isinstance(final.a.map("x * 2"), Expression)


def test_python_truth_and_iteration_are_rejected():
    with pytest.raises(TypeError, match="truth value"):
        bool(final.x)
    with pytest.raises(TypeError, match="iterated"):
        list(final.x)
    with pytest.raises(TypeError):
        len(final.x)


def test_default_is_lazy_and_handles_only_absence():
    assert load(
        {
            "a": None,
            "b": 0,
            "c": False,
            "gone": computed(lambda get: delete),
            "out": [
                final.a.default(1),
                final.b.default(1),
                final.c.default(1),
                final.gone.default(2),
                final.missing.default(final.b),
                final.a.default(final.nonexistent),
            ],
        }
    )["out"] == [None, 0, False, 2, 0, None]
    assert load({"x": computed(lambda get: delete).default(3)}) == {"x": 3}


def test_default_during_navigation():
    assert load({"x": final.missing.default({"a": {"b": 3}}).a.b}) == {"x": 3}
    assert load({"a": {}, "x": final.a.default({"b": 3}).b.default(4)})["x"] == 4
    assert load({"a": {"b": 2}, "x": final.a.default({"b": final.x}).b})["x"] == 2


def test_default_does_not_hide_missing_dependencies_or_callback_errors():
    with pytest.raises(MissingValueError):
        load({"a": computed(lambda get: get(final.missing)).default(3)})
    with pytest.raises(ZeroDivisionError):
        load({"a": computed(lambda get: 1 / 0).default(3)})
    with pytest.raises(ConfigError, match="cycle"):
        load({"a": final.a.default(3)})


def test_dynamic_conditional_dependencies():
    assert load(
        {
            "enabled": False,
            "result": computed(
                lambda get: get(final.missing) if get(final.enabled) else 1
            ),
        }
    ) == {"enabled": False, "result": 1}


def test_map_and_computed_composition():
    assert (
        load({"x": 3, "out": computed(lambda get: get(final.x.map("x + 1")) * 2)})[
            "out"
        ]
        == 8
    )
    assert load({"x": computed(lambda get: final.y * 2), "y": 3})["x"] == 6


def test_structural_reads_avoid_child_contents():
    assert load(
        {
            "x": {
                "count": final.x.len(),
                "names": final.x.keys(),
                "exists": final.x.exists(),
            }
        }
    ) == {
        "x": {"count": 3, "names": ["count", "names", "exists"], "exists": True},
    }
    assert load({"x": [final.x.len()]}) == {"x": [1]}
    assert load({"x": (final.x.len(),)}) == {"x": (1,)}


def test_presence_is_narrower_than_all_keys():
    assert load(
        {
            "x": {
                "a": {},
                "b": computed(lambda get: delete if get(final.x.a.exists()) else 1),
            }
        }
    ) == {
        "x": {"a": {}},
    }


def test_computed_deletions_affect_actual_membership():
    assert load(
        {
            "x": {"a": computed(lambda get: delete), "b": {"nested": 1}},
            "out": [
                final.x.keys(),
                final.x.len(),
                final.x.a.exists(),
                final.x.b.exists(),
            ],
        }
    )["out"] == [
        ["b"],
        1,
        False,
        True,
    ]
    with pytest.raises(ConfigError, match="cycle"):
        load({"x": {"a": computed(lambda get: delete if get(final.x.len()) else 1)}})


def test_reference_consumes_source_deletions_instead_of_replaying_them():
    assert load(
        {"source": {"a": delete}, "dest": {"a": 1}}, {"dest": final.source}
    ) == {
        "source": {},
        "dest": {"a": 1},
    }


def test_producer_runs_once_for_keys_presence_length_and_values():
    calls = []

    def produce(get):
        calls.append(1)
        return {"a": len(calls), "b": computed(lambda get: delete)}

    cfg = load(
        {
            "keys": final.x.keys(),
            "exists": final.x.exists(),
            "length": final.x.len(),
            "x": computed(produce),
            "copy": final.x,
        }
    )
    assert cfg == {
        "keys": ["a"],
        "exists": True,
        "length": 1,
        "x": {"a": 1},
        "copy": {"a": 1},
    }
    assert calls == [1]


def test_computation_cache_distinguishes_layers_and_locations():
    calls = []

    def increment(get):
        calls.append(1)
        return get(value.default(0)) + 1

    expr = computed(increment)
    assert load({"a": expr, "b": expr}, {"a": expr}) == {"a": 2, "b": 1}
    assert len(calls) == 3
    assert load({"a": expr}) == {"a": 1}
    assert len(calls) == 4


@pytest.mark.parametrize(
    "source",
    [
        {"a": final.a},
        {"a": final.b, "b": final.a},
        {"a": final(0)},
        {"a": computed(lambda get: get(final))},
    ],
)
def test_dependency_cycles_have_trace(source):
    with pytest.raises(ConfigError, match=r"Dependency cycle:.*layer.*->"):
        load(source)


def test_previous_can_cycle_through_final():
    with pytest.raises(ConfigError, match="cycle"):
        load({"a": final.b}, {"b": previous.a})


@pytest.mark.parametrize("container", [list, tuple])
@pytest.mark.parametrize(
    "item", [delete, computed(lambda get: delete), replace(delete)]
)
def test_delete_in_sequences_errors(container, item):
    with pytest.raises(ConfigError, match="delete.*sequence"):
        load({"x": container([item])})


def test_overwritten_scalar_producer_is_not_evaluated():
    assert load({"x": computed(lambda get: 1 / 0)}, {"x": 3}) == {"x": 3}
    assert load({"x": computed(lambda get: 1 / 0)}, {"x": replace({})}) == {"x": {}}


def test_learning_rate_checkpoint_and_augmentation_patterns():
    cfg = load(
        {
            "steps": 96000,
            "cooldown": final.steps // 10,
            "lr": 3e-4,
            "backbone_lr": final.lr * 0.1,
            "checkpoints": [],
            "augmentations": ["color", "crop", "flip"],
        },
        {
            "steps": 48000,
            "lr": value * math.sqrt(0.5),
            "checkpoints": value.map(lambda x: [*x, final.steps - final.cooldown]),
            "augmentations": value.map(
                lambda x: [item for item in x if item != "color"]
            ),
        },
    )
    assert cfg["lr"] == pytest.approx(3e-4 * math.sqrt(0.5))
    assert cfg["backbone_lr"] == pytest.approx(cfg["lr"] * 0.1)
    assert cfg["checkpoints"] == [43200]
    assert cfg["augmentations"] == ["crop", "flip"]


def test_reweight_arbitrary_dataset_names_without_resolving_derived_fields():
    def reweight(get):
        weights = {name: get(value[name].weight) for name in get(value.keys())}
        synthetic = {
            name for name in weights if get(value[name].label_type) == "synthetic"
        }
        total = sum(weights.values())
        subtotal = sum(weights[name] for name in synthetic)
        return {
            name: {
                "weight": weight
                * (
                    0.8 * total / subtotal
                    if name in synthetic
                    else 0.2 * total / (total - subtotal)
                )
            }
            for name, weight in weights.items()
        }

    cfg = load(
        {
            "datasets": {
                "unknown-render": {
                    "weight": 1,
                    "label_type": "synthetic",
                    "fraction": final(1).weight / 2,
                },
                "unknown-camera": {"weight": 1, "label_type": "sfm"},
            }
        },
        {"datasets": computed(reweight)},
    )
    assert cfg["datasets"]["unknown-render"] == {
        "weight": 1.6,
        "label_type": "synthetic",
        "fraction": 0.8,
    }
    assert cfg["datasets"]["unknown-camera"]["weight"] == 0.4


def test_broad_transform_reports_real_reweight_cycle():
    with pytest.raises(ConfigError, match="cycle"):
        load(
            {"datasets": {"a": {"weight": 1, "fraction": final.datasets.a.weight}}},
            {"datasets": value.map(lambda x: x)},
        )


def test_missing_reference_errors_instead_of_deleting_or_inheriting():
    with pytest.raises(MissingValueError):
        load({"a": final.missing})
    with pytest.raises(MissingValueError):
        load({"a": 1}, {"a": final.missing})
    with pytest.raises(MissingValueError):
        load({"b": final.a.default(3), "a": final.missing})
    assert load({"a": final.missing.default(3)}) == {"a": 3}


def test_failing_producer_is_not_retried():
    calls = []

    def fail(get):
        calls.append(1)
        raise RuntimeError("expected")

    expression = computed(fail)

    def catch(get):
        for _ in range(2):
            try:
                get(expression)
            except RuntimeError:
                pass
        return 1

    assert load({"x": computed(catch)}) == {"x": 1}
    assert calls == [1]


def test_nested_default_does_not_hide_a_broken_inherited_reference():
    with pytest.raises(MissingValueError, match="missing"):
        load(
            {"a": final.missing},
            {"a": replace({"b": previous.a.foo.default(3)})},
        )


def test_structural_reads_remain_precise_through_value_references():
    assert load(
        {
            "a": final.options,
            "options": {
                "c": {},
                "b": computed(lambda get: delete if get(final.a.c.exists()) else 1),
            },
        }
    ) == {"a": {"c": {}}, "options": {"c": {}}}


def test_referenced_absence_does_not_hide_inherited_membership():
    cfg = load(
        {"source": {"a": delete}, "dest": {"a": 1}},
        {"dest": final.source},
        {"keys": final.dest.keys(), "length": final.dest.len()},
    )
    assert cfg == {"source": {}, "dest": {"a": 1}, "keys": ["a"], "length": 1}


def test_first_contribution_order_does_not_evaluate_overwritten_values():
    cfg = load(
        {"a": computed(lambda get: 1 / 0), "b": 2},
        {"a": delete},
        {"a": 3},
    )
    assert list(cfg) == ["a", "b"]
    assert cfg == {"a": 3, "b": 2}
