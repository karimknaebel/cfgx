---
icon: lucide/terminal
---

# CLI

The CLI is an optional convenience for when you want a quick render or snapshot.
Most workflows can just call the Python API. All commands load the provided
paths as an implicit parent chain and apply overrides in order.

## Commands

### `cfgx render` / `cfgx print`

Pretty-print the config dictionary to stdout. Use `--no-pretty` to print the
raw `repr(config)` output.

```bash
cfgx render configs/base.py configs/finetune.py -o trainer.max_steps=12000 trainer.hooks+=wandb
```

### `cfgx dump` / `cfgx freeze`

Print a Python snapshot (`config = ...`) to stdout.
This is a best-effort snapshot based on `repr(config)`; you are responsible for
ensuring it is valid Python that can recreate the config. Otherwise formatting
can raise (for example, on a syntax error) or the output may fail to load
because required imports are missing.

```bash
cfgx dump configs/finetune.py -o trainer.max_steps=12000 > runs/finetune_config.py
```

## Options

- `-o, --overrides`: One or more override strings, e.g. `key=value`. You can pass
  multiple values after a single flag or repeat the flag.
- `--format {pprint,ruff}`: Optional formatter to apply (dump only; default: no formatting).
- `--sort-keys`: Sort dict keys throughout nested dict/list/tuple structures (including
  dict subclasses) before formatting (dump only; default: false).
- `--no-resolve-lazy`: Print `Lazy` values without resolving them (render only).
- `--no-pretty`: Print raw `repr(config)` output (render only).
