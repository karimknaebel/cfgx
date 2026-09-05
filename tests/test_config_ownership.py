import sys
from collections import UserDict, UserList
from collections.abc import Mapping, Sequence
from types import ModuleType

import pytest

from cfgx import (
    Delete,
    Lazy,
    Replace,
    Update,
    apply_overrides,
    load,
    merge,
    resolve_lazy,
)


class Resource:
    def __init__(self):
        self.values = [Lazy("c.steps")]

    def __copy__(self):
        raise AssertionError("Resource must not be copied")

    def __deepcopy__(self, memo):
        raise AssertionError("Resource must not be deep-copied")


def make_config():
    return {"steps": 10, "tags": ["base"], "rows": [{"value": Lazy("c.steps * 2")}]}


def test_merge_copies_only_merged_dictionary_branches():
    base = {"untouched": {"items": []}, "changed": {"old": [], "remove": True}}
    override = {"changed": {"new": [], "remove": Delete()}, "added": {"items": []}}

    result = merge(base, override)

    assert result is not base
    assert result is not override
    assert result["untouched"] is base["untouched"]
    assert result["changed"] is not base["changed"]
    assert result["changed"] is not override["changed"]
    assert result["changed"]["old"] is base["changed"]["old"]
    assert result["changed"]["new"] is override["changed"]["new"]
    assert "remove" not in result["changed"]
    assert base["changed"]["remove"] is True
    assert result["added"] is not override["added"]
    assert result["added"]["items"] is override["added"]["items"]


def test_merge_creates_plain_dicts_without_subclass_copy_hooks():
    class ConfigDict(dict):
        def copy(self):
            raise AssertionError("Merge must not delegate ownership to a copy hook")

    base = ConfigDict(branch=ConfigDict(value=1), untouched=ConfigDict(value=2))
    override = ConfigDict(branch=ConfigDict(value=3), added=ConfigDict(value=4))

    result = merge(base, override)

    assert type(result) is dict
    assert type(result["branch"]) is dict
    assert type(result["added"]) is dict
    assert result["untouched"] is base["untouched"]
    assert base["branch"]["value"] == 1
    assert result["branch"]["value"] == 3


def test_load_can_split_dictionary_aliases_while_preserving_list_aliases():
    branch = {"value": 1}
    rows = [branch]
    source = {"left": branch, "right": branch, "rows": rows, "other_rows": rows}

    cfg = load(source, overrides=["left.value=2", "rows[0].value=3"])

    assert cfg["left"] == {"value": 2}
    assert cfg["right"] == {"value": 1}
    assert cfg["left"] is not cfg["right"]
    assert cfg["rows"] is cfg["other_rows"] is rows
    assert cfg["rows"][0] is branch
    assert source["left"] is source["right"] is branch
    assert branch == {"value": 3}


@pytest.mark.parametrize("operation", [Replace, lambda value: Update(lambda: value)])
def test_replace_and_update_results_are_installed_without_copying_or_merging(operation):
    value = {"remove": Delete(), "items": []}

    cfg = load({"value": operation(value)})

    assert cfg["value"] is value
    assert isinstance(value["remove"], Delete)
    apply_overrides(cfg, ["value.items+=extra"])
    assert value["items"] == ["extra"]


def test_dict_and_list_subclasses_are_resolved_in_place():
    writes = []

    class ConfigDict(dict):
        label = "dict metadata"

        def __setitem__(self, key, value):
            writes.append(("dict", key, value))
            super().__setitem__(key, value)

    class ConfigList(list):
        label = "list metadata"

        def __setitem__(self, key, value):
            writes.append(("list", key, value))
            super().__setitem__(key, value)

    branch = ConfigDict(value=Lazy("c.steps"))
    values = ConfigList([Lazy("c.steps * 2")])
    cfg = load({"steps": 10, "branch": Replace(branch), "values": values})

    assert cfg["branch"] is branch
    assert cfg["values"] is values
    assert branch.label == "dict metadata"
    assert values.label == "list metadata"
    assert writes == [("dict", "value", 10), ("list", 0, 20)]


@pytest.mark.parametrize(
    "wrap", [tuple, UserList, lambda values: UserDict(value=values[0])]
)
def test_other_containers_are_shared_and_not_resolved(wrap):
    lazy = Lazy("c.steps")
    value = wrap([lazy])

    cfg = load({"steps": 10, "value": value, "reference": Lazy("c.value")})

    assert cfg["value"] is value
    assert cfg["reference"] is value
    assert value["value" if isinstance(value, UserDict) else 0] is lazy


def test_overrides_can_explicitly_traverse_other_container_types():
    value = UserDict(rows=UserList([{"value": 1}]))
    cfg = load({"value": value})

    assert apply_overrides(cfg, ["value.rows[0].value=2"]) is cfg
    assert value["rows"][0] == {"value": 2}


@pytest.mark.parametrize(
    "override, expected",
    [
        ("left.items+=c", ["a", "b", "c"]),
        ("left.items-=a", ["b"]),
        ("left.items[0]!=", ["b"]),
        ("left.items[0]=c", ["c", "b"]),
        ("left.items=update:v.append('c') or v", ["a", "b", "c"]),
    ],
)
def test_overrides_mutate_shared_targets(override, expected):
    items = ["a", "b"]
    cfg = {"left": {"items": items}, "right": {"items": items}}

    assert apply_overrides(cfg, [override]) is cfg
    assert cfg["left"]["items"] is cfg["right"]["items"] is items
    assert items == expected


def test_override_assignment_rebinds_a_key_without_mutating_its_old_value():
    items = ["a"]
    cfg = {"left": items, "right": items}

    apply_overrides(cfg, ["left=['b']"])

    assert cfg["left"] == ["b"]
    assert cfg["left"] is not items
    assert cfg["right"] is items
    assert items == ["a"]


@pytest.mark.parametrize("existing", [False, True])
def test_append_only_assigns_a_target_when_creating_it(existing):
    assignments = []

    class ConfigDict(dict):
        def __setitem__(self, key, value):
            assignments.append(key)
            super().__setitem__(key, value)

    cfg = ConfigDict(items=[]) if existing else ConfigDict()

    apply_overrides(cfg, ["items+=a"])

    assert cfg["items"] == ["a"]
    assert assignments == ([] if existing else ["items"])


@pytest.mark.parametrize("deferred", [False, True])
def test_update_receives_previous_objects_directly(deferred):
    items = ["base"]
    received = []

    def update(value):
        received.append(value)
        value.append("extra")
        return value

    cfg = load(
        {"items": Lazy(lambda c: items) if deferred else items}, resolve_lazy=False
    )
    cfg = merge(cfg, {"items": Update(update)})
    if deferred:
        assert received == []
    resolve_lazy(cfg)

    assert received == [items]
    assert received[0] is items
    assert cfg["items"] is items
    assert items == ["base", "extra"]


def test_update_returning_new_values_preserves_previous_values():
    items = ["base"]
    base = {"items": items}

    first = load(base, {"items": Update("v + ['first']")})
    second = load(base, {"items": Update("v + ['second']")})

    assert first["items"] == ["base", "first"]
    assert second["items"] == ["base", "second"]
    assert items == ["base"]


def test_missing_update_does_not_copy_callable_defaults():
    def append(items=[]):
        items.append("extra")
        return items

    source = {"items": Update(append)}
    first = load(source)
    second = load(source)

    assert first["items"] is second["items"] is append.__defaults__[0]
    assert first["items"] == ["extra", "extra"]


def test_arbitrary_objects_are_shared_and_callbacks_can_mutate_them():
    resource = Resource()

    def update(value):
        assert value is resource
        value.values.append("update")
        return value

    def lazy(c):
        assert c.resource is resource
        c.resource.values.append("lazy")
        return c.resource

    cfg = load(
        {"resource": resource},
        {"resource": Update(update), "reference": Lazy(lazy)},
    )

    assert cfg["resource"] is cfg["reference"] is resource
    assert isinstance(resource.values[0], Lazy)
    assert resource.values[1:] == ["update", "lazy"]


@pytest.mark.parametrize("reference", ["c.branch", "c.tags"])
def test_lazy_container_references_remain_live_proxies(reference):
    cfg = load({"branch": {"value": 1}, "tags": [1], "reference": Lazy(reference)})
    proxy = cfg["reference"]

    assert isinstance(proxy, Mapping if reference == "c.branch" else Sequence)
    assert proxy is not cfg[reference[2:]]
    with pytest.raises(TypeError):
        proxy["value" if reference == "c.branch" else 0] = 3

    apply_overrides(cfg, ["branch={'value': 2}", "tags=[2]"])

    assert proxy["value" if reference == "c.branch" else 0] == 2
    assert resolve_lazy(cfg) is cfg
    assert cfg["reference"] is proxy


def test_update_over_lazy_reference_receives_the_proxy():
    received = []

    def update(value):
        received.append(value)
        return [*value, "extra"]

    cfg = load(
        {"tags": ["base"], "extended": Lazy("c.tags")}, {"extended": Update(update)}
    )

    assert isinstance(received[0], Sequence)
    assert not isinstance(received[0], list)
    assert cfg["extended"] == ["base", "extra"]
    assert cfg["tags"] == ["base"]


def test_lazy_returned_containers_and_nested_proxies_are_not_copied():
    returned = {"value": Lazy("c.steps")}
    cfg = load(
        {
            "steps": 10,
            "branch": Lazy(lambda c: returned),
            "references": Lazy(lambda c: [c.branch]),
        }
    )

    assert cfg["branch"] is returned
    assert returned == {"value": 10}
    assert isinstance(cfg["references"][0], Mapping)
    apply_overrides(cfg, ["branch={'value': 20}"])
    assert cfg["references"][0]["value"] == 20


@pytest.mark.parametrize("read_first", [False, True])
@pytest.mark.parametrize("container_type", [dict, list])
def test_lazy_commits_returned_container_before_resolving_its_children(
    read_first, container_type
):
    returned = []

    def make_branch(c):
        if container_type is dict:
            branch = {"value": len(returned) + 1, "derived": Lazy("c.branch.value * 2")}
        else:
            branch = [len(returned) + 1, Lazy("c.branch[0] * 2")]
        returned.append(branch)
        return branch

    source = {"branch": Lazy(make_branch)}
    if read_first:
        source = {
            "read": Lazy(
                "c.branch.derived" if container_type is dict else "c.branch[1]"
            ),
            **source,
        }

    cfg = load(source)

    assert len(returned) == 1
    assert cfg["branch"] is returned[0]
    assert cfg["branch"] == (
        {"value": 1, "derived": 2} if container_type is dict else [1, 2]
    )
    if read_first:
        assert cfg["read"] == 2


def test_lazy_in_aliased_container_is_replaced_once():
    calls = []

    def compute(c):
        calls.append(c.steps)
        return c.steps * 2

    values = [Lazy(compute)]
    cfg = {"steps": 10, "read": Lazy("c.right[0]"), "left": values, "right": values}

    assert resolve_lazy(cfg) is cfg
    assert resolve_lazy(cfg) is cfg
    assert calls == [10]
    assert cfg["read"] == 20
    assert cfg["left"] is cfg["right"] is values
    assert values == [20]


def test_lazy_wrapper_itself_does_not_cache_across_slots_or_loads():
    calls = []

    def compute(c):
        calls.append(c.steps)
        return c.steps * 2

    lazy = Lazy(compute)
    source = {"steps": 10, "left": lazy, "right": lazy}

    assert load(source)["left"] == 20
    assert load(source, overrides=["steps=20"])["right"] == 40
    assert calls == [10, 10, 20, 20]
    assert source["left"] is source["right"] is lazy


def test_lazy_cycle_in_a_returned_container_raises():
    cfg = {"branch": Lazy(lambda c: {"value": Lazy("c.branch.value")})}

    with pytest.raises(ValueError, match="Lazy cycle detected at branch.value"):
        resolve_lazy(cfg)
    assert isinstance(cfg["branch"], dict)
    assert isinstance(cfg["branch"]["value"], Lazy)


@pytest.mark.parametrize("source_kind", ["file", "inline", "factory"])
def test_fresh_sources_produce_independent_variants(tmp_path, source_kind):
    path = tmp_path / "base.py"
    code = (
        "from cfgx import Lazy\n"
        "config = {'steps': 10, 'tags': ['base'], 'rows': [{'value': Lazy('c.steps * 2')}]}\n"
    )
    path.write_text(code)

    def source():
        if source_kind == "file":
            return path
        if source_kind == "factory":
            return make_config()
        return {"steps": 10, "tags": ["base"], "rows": [{"value": Lazy("c.steps * 2")}]}

    first = load(source(), overrides=["steps=20", "tags+=first"])
    second = load(source(), overrides=["steps=30", "tags+=second"])

    assert first == {"steps": 20, "tags": ["base", "first"], "rows": [{"value": 40}]}
    assert second == {"steps": 30, "tags": ["base", "second"], "rows": [{"value": 60}]}
    assert first["tags"] is not second["tags"]
    assert first["rows"][0] is not second["rows"][0]
    assert path.read_text() == code


@pytest.mark.parametrize(
    "prepare", [lambda base: base, dict.copy, lambda base: merge(base, {})]
)
def test_reusing_a_base_or_shallow_copy_shares_mutation_and_resolved_lazies(prepare):
    base = make_config()

    first = load(prepare(base), overrides=["steps=20", "tags+=first"])
    second = load(prepare(base), overrides=["steps=30", "tags+=second"])

    assert first["tags"] is second["tags"] is base["tags"]
    assert first["tags"] == ["base", "first", "second"]
    assert first["rows"] is second["rows"] is base["rows"]
    assert second["rows"] == [{"value": 40}]
    assert base["steps"] == 10


def test_variants_of_an_unresolved_loaded_base_still_share_values():
    base = load(make_config(), resolve_lazy=False)
    first = merge(base, {"steps": 20})
    second = merge(base, {"steps": 30})

    resolve_lazy(first)
    resolve_lazy(second)

    assert base["rows"] is first["rows"] is second["rows"]
    assert second["rows"] == [{"value": 40}]


def test_explicit_copies_of_known_containers_allow_reusing_a_template():
    base = make_config()
    resource = Resource()

    def fresh_base():
        return {
            **base,
            "tags": list(base["tags"]),
            "rows": [dict(row) for row in base["rows"]],
            "resource": resource,
        }

    first = load(fresh_base(), overrides=["steps=20", "tags+=first"])
    second = load(fresh_base(), overrides=["steps=30", "tags+=second"])

    assert first["rows"] == [{"value": 40}]
    assert second["rows"] == [{"value": 60}]
    assert first["tags"] == ["base", "first"]
    assert second["tags"] == ["base", "second"]
    assert base["tags"] == ["base"]
    assert isinstance(base["rows"][0]["value"], Lazy)
    assert first["resource"] is second["resource"] is resource


def test_fresh_file_loads_can_consume_imported_mutable_values(tmp_path, monkeypatch):
    shared = ModuleType("_cfgx_shared_config")
    shared.rows = [{"value": Lazy("c.steps * 2")}]
    shared.tags = ["base"]
    shared.resource = Resource()
    monkeypatch.setitem(sys.modules, shared.__name__, shared)
    path = tmp_path / "config.py"
    path.write_text(
        "from _cfgx_shared_config import rows, tags, resource\n"
        "config = {'steps': 10, 'rows': rows, 'tags': tags, 'resource': resource}\n"
    )

    first = load(path, overrides=["steps=20", "tags+=first"])
    second = load(path, overrides=["steps=30", "tags+=second"])

    assert first["rows"] is second["rows"] is shared.rows
    assert shared.rows == [{"value": 40}]
    assert first["tags"] is second["tags"] is shared.tags
    assert shared.tags == ["base", "first", "second"]
    assert first["resource"] is second["resource"] is shared.resource


def test_file_importing_a_factory_creates_independent_variants(tmp_path, monkeypatch):
    shared = ModuleType("_cfgx_config_factory")
    shared.make_config = make_config
    monkeypatch.setitem(sys.modules, shared.__name__, shared)
    path = tmp_path / "config.py"
    path.write_text(
        "from _cfgx_config_factory import make_config\nconfig = make_config()\n"
    )

    first = load(path, overrides=["steps=20", "tags+=first"])
    second = load(path, overrides=["steps=30", "tags+=second"])

    assert first["rows"] == [{"value": 40}]
    assert second["rows"] == [{"value": 60}]
    assert first["tags"] == ["base", "first"]
    assert second["tags"] == ["base", "second"]


def test_failed_operations_do_not_roll_back_mutations():
    items = []

    def fail(value):
        value.append("before error")
        raise RuntimeError("update failed")

    with pytest.raises(RuntimeError, match="update failed"):
        merge({"items": items}, {"items": Update(fail)})
    assert items == ["before error"]

    cfg = {"items": items, "resolved": Lazy("1 + 1"), "failed": Lazy("1 / 0")}
    with pytest.raises(ValueError, match="not a list"):
        apply_overrides(cfg, ["items+=extra", "resolved+=invalid"])
    assert items == ["before error", "extra"]

    with pytest.raises(ZeroDivisionError):
        resolve_lazy(cfg)
    assert cfg["resolved"] == 2
    assert isinstance(cfg["failed"], Lazy)
