# Composing a training config

A compact depth-training example with separate training, model, and dataset
settings. The fields illustrate configuration patterns adapted from real training
configs. Model and dataset entries are descriptive metadata; loading or rendering
these files requires only cfgx.

Start with `run.py` to see the composition order:

| File | Contribution |
| --- | --- |
| `base.py` | Training defaults, optimizer groups, scheduler, and a checkpoint before cooldown |
| `model.py` | Backbone and decoder, with the decoder width derived from the backbone |
| `data.py` | Two synthetic datasets and one captured dataset, with sampling weights and optional augmentations |
| `48k_bs64.py` | Smaller batch, shorter run, and inherited learning-rate scaling |
| `reweight_synthetic80p.py` | Adjust the dataset mix to 80% synthetic while preserving total weight |
| `light_augmentation.py` | Filter global and per-dataset augmentations while preserving other fields |

The basic dependencies use `final`, while the smaller-batch preset uses `value`
to transform inherited settings. Dataset transformations use `computed` to read
selected fields and return partial patches. They discover dataset names through
`.keys()`; augmentation filtering uses `.exists()` for optional settings.

The preset treats the inherited learning rate as the rate for batch size 128 and
scales it by `sqrt(final.batch_size / 128)`. Overriding the batch size therefore
also updates the learning rate and both optimizer groups.

```sh
uv run cfgx render examples/training/run.py
uv run cfgx render examples/training/run.py -o 'batch_size=32'
uv run cfgx render examples/training/run.py -o 'steps=24000' 'lr=expr:value * 0.5'
uv run cfgx render examples/training/run.py -o 'model.backbone.width=768'
```

The composed run has batch size 64, 48,000 steps, a 4,800-step cooldown, and a
checkpoint at step 43,200, selected with `[final.steps - final.cooldown_steps]`.
Changing `steps` also changes the schedule and checkpoint; changing the backbone
width updates the decoder's input channels.

Hypersim and Structured3D start with weights 1 and 2; BlendedMVS starts with weight
1. Reweighting preserves the synthetic datasets' 1:2 ratio and gives them a
combined weight of 3.2, leaving 0.8 for BlendedMVS. The total stays at 4, with 80%
allocated to synthetic data.

To explore fewer transformations, load just the initial files:

```sh
uv run cfgx render examples/training/base.py examples/training/model.py examples/training/data.py
```
