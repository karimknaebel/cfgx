from cfgx.cli import main
from cfgx import dumps, format as format_config, load


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
    expected = format_config(
        load(cfg_path, overrides=["a.b=2", "c=3"]),
    )
    assert out == f"{expected}\n"


def test_dump_basic(capsys, tmp_path):
    cfg_path = tmp_path / "cfg.py"
    cfg_path.write_text("config = {'a': 1}\n")

    exit_code = main(["dump", str(cfg_path)])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert out == dumps(load(cfg_path))


def test_render_computed_override(capsys, tmp_path):
    cfg_path = tmp_path / "cfg.py"
    cfg_path.write_text("from cfgx import final\nconfig = {'a': 2, 'b': final.a * 2}\n")
    assert main(["render", str(cfg_path), "-o", "a=expr:value * 3"]) == 0
    assert capsys.readouterr().out == "{'a': 6, 'b': 12}\n"


def test_render_whole_layer_and_deletion_overrides(capsys, tmp_path):
    cfg_path = tmp_path / "cfg.py"
    cfg_path.write_text("config = {'a': 2, 'nested': {'kept': 1}}\n")
    assert (
        main(
            [
                "render",
                str(cfg_path),
                "-o",
                "expr:{'nested': {'derived': final.a * 2}}",
                "missing.deep!=",
                "-o",
                "a=expr:value * 3",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out == (
        "{'a': 6, 'nested': {'kept': 1, 'derived': 12}, 'missing': {}}\n"
    )


def test_render_raw(capsys, tmp_path):
    cfg_path = tmp_path / "cfg.py"
    cfg_path.write_text("config = {'a': 1}\n")

    exit_code = main(["render", str(cfg_path), "--format", "raw"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert out == f"{format_config(load(cfg_path), format='raw')}\n"
