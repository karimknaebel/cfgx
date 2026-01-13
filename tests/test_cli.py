from cfgx.cli import main
from cfgx.config import _format_snapshot, format as format_config, load


def test_render_basic(capsys, tmp_path):
    cfg_path = tmp_path / "cfg.py"
    cfg_path.write_text("config = {'a': 1}\n")

    exit_code = main(["render", str(cfg_path)])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert out == f"{format_config(load(cfg_path))}\n"


def test_render_overrides_list(capsys, tmp_path):
    cfg_path = tmp_path / "cfg.py"
    cfg_path.write_text("config = {'a': {'b': 1}}\n")

    exit_code = main(["render", str(cfg_path), "-o", "a.b=2", "c=3"])

    out = capsys.readouterr().out
    assert exit_code == 0
    expected = format_config(load(cfg_path, overrides=["a.b=2", "c=3"]))
    assert out == f"{expected}\n"


def test_dump_basic(capsys, tmp_path):
    cfg_path = tmp_path / "cfg.py"
    cfg_path.write_text("config = {'a': 1}\n")

    exit_code = main(["dump", str(cfg_path)])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert out == _format_snapshot(load(cfg_path))
