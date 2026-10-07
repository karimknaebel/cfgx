from pprint import pformat

from cfgx import dump, dumps, format


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


def test_format_sort_keys_includes_tuples():
    cfg = {"a": ({"b": 2, "a": 1},)}
    formatted = format(cfg, format="raw", sort_keys=True)
    assert formatted == "{'a': ({'a': 1, 'b': 2},)}"


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
