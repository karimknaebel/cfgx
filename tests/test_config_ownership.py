from collections import UserDict

import pytest

from cfgx import ConfigError, computed, final, load, replace, value


def test_source_aliases_and_output_references_are_independent():
    shared = {"items": [{"x": 1}]}
    source = {"a": shared, "b": shared, "c": final.a}
    result = load(source, {"a": {"items": [{"x": 2}]}})
    result["a"]["items"][0]["x"] = 3
    assert source["a"] == {"items": [{"x": 1}]}
    assert result["b"] == {"items": [{"x": 1}]}
    assert result["c"] == {"items": [{"x": 2}]}
    assert result["a"] is not result["c"]
    assert result["b"] is not shared


def test_repeated_sequence_items_are_independent():
    shared = {"x": []}
    result = load({"items": [shared, shared], "tuple": (shared, shared)})
    result["items"][0]["x"].append(1)
    assert result["items"][1] == {"x": []}
    assert result["tuple"] == ({"x": []}, {"x": []})
    assert result["tuple"][0] is not result["tuple"][1]
    assert shared == {"x": []}


def test_repeated_and_transitive_references_are_independent():
    result = load(
        {"source": {"items": []}, "a": final.source, "b": final.source, "c": final.a}
    )
    result["a"]["items"].append(1)
    assert result["source"] == result["b"] == result["c"] == {"items": []}


def test_reusing_sources_recomputes_without_mutation():
    source = {"x": 2, "items": [final.x]}
    assert load(source, overrides=["x=3"])["items"] == [3]
    assert load(source, overrides=["x=4"])["items"] == [4]
    assert load(source)["items"] == [2]


def test_callback_inputs_are_copies():
    def mutate(get):
        first = get(final.a)
        first.append(2)
        assert get(final.a) == [1]
        return first

    source = {"a": [1], "b": computed(mutate)}
    assert load(source) == {"a": [1], "b": [1, 2]}
    assert source["a"] == [1]


def test_previous_map_input_is_copied():
    original = {"items": [1]}

    def mutate(x):
        x.append(2)
        return x

    assert load(original, {"items": value.map(mutate)}) == {"items": [1, 2]}
    assert original == {"items": [1]}


def test_producer_output_is_captured_before_external_mutation():
    shared = {"a": [1]}

    def mutate(get):
        get(final.produced.keys())
        shared["a"].append(2)
        shared["b"] = 3
        return get(final.produced)

    assert load(
        {"copy": computed(mutate), "produced": computed(lambda get: shared)}
    ) == {
        "copy": {"a": [1]},
        "produced": {"a": [1]},
    }


def test_opaque_leaves_keep_identity():
    class CustomList(list):
        pass

    expression = final.x
    custom = CustomList([expression])
    mapping = UserDict({"a": [1]})
    opaque_set = {1, 2}
    result = load({"x": 2, "custom": custom, "mapping": mapping, "set": opaque_set})
    assert result["custom"] is custom
    assert result["mapping"] is mapping
    assert result["set"] is opaque_set
    assert custom[0] is expression


def test_explicit_opaque_reads_and_copied_supported_results():
    mapping = UserDict({"x": [3]})
    result = load(
        {
            "opaque": mapping,
            "read": final.opaque.x,
            "size": final.opaque.len(),
            "keys": final.opaque.keys(),
        }
    )
    assert result["read"] == [3]
    assert result["size"] == 1
    assert result["keys"] == ["x"]
    result["read"].append(4)
    assert mapping["x"] == [3]


def test_dictionary_contributions_and_overrides_replace_opaque_without_mutating_it():
    mapping = UserDict({"x": 1})
    assert load({"a": mapping}, {"a": {"y": 2}}) == {"a": {"y": 2}}
    assert load({"a": mapping}, overrides=["a.y=2"]) == {"a": {"y": 2}}
    assert mapping == {"x": 1}


def test_replace_does_not_preserve_supported_container_identity():
    source = {"x": []}
    result = load({"branch": replace(source)})
    assert result["branch"] is not source
    assert result["branch"]["x"] is not source["x"]


def test_raw_container_cycles_error():
    source = []
    source.append(source)
    with pytest.raises(ConfigError, match="Cyclic Python container"):
        load({"x": source})
