from cfgx.cli import main
from cfgx.config import dumps, format as format_config, load


def test_render_basic(capsys, tmp_path):
    cfg_path = tmp_path / "cfg.py"
    cfg_path.write_text("config = {'a': 1}\n")

    exit_code = main(["render", str(cfg_path)])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert out == f"{format_config(load(cfg_path), format='pprint')}\n"


def test_render_overrides_list(capsys, tmp_path):
    cfg_path = tmp_path / "cfg.py"
    cfg_path.write_text("config = {'a': {'b': 1}}\n")

    exit_code = main(["render", str(cfg_path), "-o", "a.b=2", "c=3"])

    out = capsys.readouterr().out
    assert exit_code == 0
    expected = format_config(
        load(cfg_path, overrides=["a.b=2", "c=3"]),
        format="pprint",
    )
    assert out == f"{expected}\n"


def test_dump_basic(capsys, tmp_path):
    cfg_path = tmp_path / "cfg.py"
    cfg_path.write_text("config = {'a': 1}\n")

    exit_code = main(["dump", str(cfg_path)])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert out == dumps(load(cfg_path))


def test_render_no_resolve_lazy(capsys, tmp_path):
    cfg_path = tmp_path / "cfg.py"
    cfg_path.write_text("from cfgx import Lazy\nconfig = {'a': Lazy('1 + 1')}\n")

    exit_code = main(["render", str(cfg_path), "--no-resolve-lazy"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Lazy(" in out


def test_render_no_pretty(capsys, tmp_path):
    cfg_path = tmp_path / "cfg.py"
    cfg_path.write_text("config = {'a': 1}\n")

    exit_code = main(["render", str(cfg_path), "--no-pretty"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert out == f"{format_config(load(cfg_path))}\n"
