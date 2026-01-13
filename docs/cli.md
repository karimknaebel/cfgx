---
icon: lucide/terminal
---

# CLI

The CLI is an optional convenience for when you want a quick render or snapshot.
Most workflows can just call the Python API. All commands load the provided
paths as an implicit parent chain and apply overrides in order.

## Commands

### `cfgx render`

Print the formatted config dictionary to stdout.

```bash
cfgx render configs/base.py configs/finetune.py -o trainer.max_steps=12000 trainer.hooks+=wandb
```

### `cfgx dump`

Print a Python snapshot (with headers and `config =`) to stdout.

```bash
cfgx dump configs/finetune.py -o trainer.max_steps=12000 > runs/finetune_config.py
```

## Options

- `-o, --overrides`: One or more override strings, e.g. `key=value`. You can pass
  multiple values after a single flag or repeat the flag.
- `--formatter {auto,ruff,pprint}`: Output formatter.
