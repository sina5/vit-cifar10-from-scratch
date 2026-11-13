# vit-cifar10-from-scratch

A small Vision Transformer (ViT) trained from scratch on CIFAR-10, and a look at what its attention heads learn. Each project trains on one consumer GPU (an RTX 3080 Ti with 12 GB).

## Projects

| # | Project | GPU time |
|---|---------|----------|
| 1 | [ViT from scratch on CIFAR-10, with the failures](projects/01_vit_from_scratch/) | About 8 hours (3 seeds plus two side studies) |
| 2 | [What the attention heads look at](projects/02_attention_maps/) | About 1 minute |

Project 2 reads the checkpoints that Project 1 writes, so run Project 1 first.

## Setup

You need [uv](https://docs.astral.sh/uv/) and an NVIDIA driver that supports CUDA 12.8.

```bash
uv sync --group dev
uv run pytest
```

`uv sync` installs PyTorch with CUDA 12.8 from the PyTorch package index. The scripts download the dataset to `data/` on first use. They write results, charts and checkpoints to `runs/`. The repository includes the results and charts of the full runs in `runs/`. Git ignores `data/`, the model weights and the `smoke/` folders.

## Layout

```
vitcifar/              shared code
  constants.py         every default value (epochs, learning rates, paths)
  models/              from-scratch ViT, CIFAR ResNet-18
  data.py              CIFAR-10 pipeline
  training.py          training loop, LR schedule, result files
  attention.py         attention extraction, rollout, demo export
  attention_stats.py   mean attention distance and entropy per head
  plotting.py          chart helpers
  experiments/         the ablation ladder and its charts
projects/NN_name/      entry scripts and the README for each project
tests/                 pytest suite (CPU only, no downloads)
```

The training scripts have a `--smoke` flag. It runs a few steps on a small subset and writes to a separate `smoke/` folder, so you can check the full pipeline in about a minute before a long run.

## License

The code is licensed under the [Apache License 2.0](LICENSE).
