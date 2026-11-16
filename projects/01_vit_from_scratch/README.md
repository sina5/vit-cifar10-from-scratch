# Project 1: ViT from scratch on CIFAR-10, with the failures

Small ViTs are known to lose to a ResNet when the dataset is small. A ViT has no built-in idea that nearby pixels belong together, so it must learn that from the data. CIFAR-10 has only 50,000 training images.

This project shows the gap, then closes part of it with one fix at a time.

## The ladder

Each ViT step adds one change to the step before it. The difference in accuracy between two neighboring steps is what that change buys. Every step runs with 3 seeds (0, 1, 2), and the charts show the mean and the standard deviation.

| Step | Model | Change from the step before |
|------|-------|-----------------------------|
| `resnet18_noaug` | ResNet-18 | baseline, no augmentation |
| `vit_p8_naive` | ViT | patch 8 (16 tokens), no augmentation, no warmup |
| `vit_p8_aug` | ViT | + crop, flip, RandAugment |
| `vit_p4_aug` | ViT | + patch 4 (64 tokens) |
| `vit_p4_aug_warmup` | ViT | + 10-epoch linear warmup |
| `vit_p4_ls` | ViT | + label smoothing 0.1 |
| `vit_p4_mix` | ViT | + MixUp (alpha 0.8) or CutMix (alpha 1.0), one per batch |
| `vit_p4_droppath` | ViT | + stochastic depth, up to 0.1 |
| `vit_p4_long` | ViT | + 2x epochs (200) |
| `resnet18_aug` | ResNet-18 | crop, flip, RandAugment (same as the ViT) |

Shared settings (see `vitcifar/constants.py`):

- ViT: width 384, depth 7, 6 heads, MLP ratio 4 (12.5 M parameters, close to the 11.2 M of ResNet-18).
- ViT optimizer: AdamW, learning rate 1e-3, weight decay 0.05, gradient clipping at 1.0.
- ResNet-18 optimizer: SGD with Nesterov momentum 0.9, learning rate 0.1, weight decay 5e-4.
- All runs: 100 epochs unless the step says otherwise, batch size 512, cosine learning-rate decay, bfloat16 mixed precision.

If a run's loss becomes NaN, training stops and the result file marks it `diverged`. The charts show this.

## Side studies

- **Learning rate and warmup** (`run_lr_sweep.py`): the `vit_p4_aug` step at peak learning rates 5e-4, 1e-3, 2e-3 and 4e-3, each with no warmup and with a 5-epoch warmup. 30 epochs per run, one seed.
- **Data efficiency** (`run_data_efficiency.py`): `resnet18_aug` and `vit_p4_droppath` on 10%, 25%, 50% and 100% of the training set. Each class keeps the same share. Smaller sets train for more epochs, so every run takes the same number of optimizer steps. One seed.

## Run

Each script skips runs that already have a result file, so you can stop it and start it again.

```bash
# Check each pipeline (about 1 minute each)
uv run python projects/01_vit_from_scratch/run_ablation.py --smoke
uv run python projects/01_vit_from_scratch/run_lr_sweep.py --smoke
uv run python projects/01_vit_from_scratch/run_data_efficiency.py --smoke

# Full studies (about 5.5 h, 35 min and 1.7 h on an RTX 3080 Ti)
uv run python projects/01_vit_from_scratch/run_ablation.py
uv run python projects/01_vit_from_scratch/run_lr_sweep.py
uv run python projects/01_vit_from_scratch/run_data_efficiency.py

# Some steps or seeds only; redraw charts from saved results
uv run python projects/01_vit_from_scratch/run_ablation.py --steps vit_p4_mix --seeds 0
uv run python projects/01_vit_from_scratch/run_ablation.py --plot-only
```

## Output

In `runs/01_vit_from_scratch/`:

- `<step>_s<seed>.json`: per-epoch loss and accuracy, wall time, peak VRAM, parameter count.
- `<step>.pt`: weights of each ViT step for seed 0. Project 2 reads `vit_p4_long.pt`.
- `accuracy_curves.png`, `fix_ladder.png`: the ladder charts.
- `lr_sweep/lr_sweep.png`, `data_efficiency/data_efficiency.png`: the side-study charts.

Model weights are not stored in Git.

## Results

Final-epoch test accuracy, mean ± sample standard deviation over 3 seeds. Time and VRAM are from seed 0 on an RTX 3080 Ti.

| Step | Test accuracy | Change | Time | Peak VRAM |
|------|---------------|--------|------|-----------|
| `resnet18_noaug` | 85.43 ± 0.34% | | 10.6 min | 1,354 MiB |
| `vit_p8_naive` | 57.34 ± 1.20% | | 4.6 min | 1,083 MiB |
| `vit_p8_aug` | 61.58 ± 2.16% | +4.2 | 5.7 min | 1,083 MiB |
| `vit_p4_aug` | 77.73 ± 0.69% | +16.1 | 14.1 min | 3,524 MiB |
| `vit_p4_aug_warmup` | 88.60 ± 0.39% | +10.9 | 14.0 min | 3,523 MiB |
| `vit_p4_ls` | 88.73 ± 0.46% | +0.1 | 14.1 min | 3,526 MiB |
| `vit_p4_mix` | 88.66 ± 0.59% | -0.1 | 14.1 min | 3,526 MiB |
| `vit_p4_droppath` | 88.31 ± 0.57% | -0.4 | 14.6 min | 3,526 MiB |
| `vit_p4_long` | 92.94 ± 0.25% | +4.6 | 29.2 min | 3,526 MiB |
| `resnet18_aug` | 94.93 ± 0.13% | | 10.7 min | 1,357 MiB |

Observations:

- **The naive ViT memorizes.** With patch 8 and no augmentation, it reaches 100% training accuracy and 57% test accuracy. ResNet-18 without augmentation reaches 85%.
- **Smaller patches and warmup give the largest gains.** Patch 4 (64 tokens instead of 16) adds 16 points. A 10-epoch warmup adds another 11 points at the same learning rate.
- **The three regularizers do not help at 100 epochs.** Label smoothing, MixUp/CutMix and stochastic depth change the accuracy by less than one standard deviation. The accuracy curves show why: with MixUp/CutMix and stochastic depth, the test accuracy is still rising at epoch 100.
- **Doubling the epochs adds 4.6 points.** This step keeps all three regularizers. The ladder does not include a 200-epoch run without them, so it does not show how much of the 4.6 points needs the regularizers.
- **ResNet-18 stays ahead.** With the same augmentation, it reaches 94.9% in 100 epochs. The best ViT reaches 92.9% in 200 epochs, a gap of 2 points.

Side studies (one seed each):

- **Learning rate and warmup** (30 epochs): without warmup, accuracy falls from 80.5% at 5e-4 to 31% at 2e-3 and 4e-3. With a 5-epoch warmup, it falls from 83.0% to 57.1%. No run diverged to NaN. The high-learning-rate runs stall at a high loss.
- **Data efficiency** (equal optimizer steps): ResNet-18 leads at every size. The gap is 14.4 points with 10% of the data (84.2% versus 69.8%) and 5.8 points with all of it (94.8% versus 89.0%).
