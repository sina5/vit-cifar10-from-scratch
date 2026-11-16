# Project 2: What the attention heads look at

This project loads the best ViT from Project 1 and shows where each attention head looks, layer by layer.

## Prerequisite

Train Project 1 first, or at least the final ViT step for seed 0:

```bash
uv run python projects/01_vit_from_scratch/run_ablation.py --steps vit_p4_long --seeds 0
```

## Run

```bash
uv run python projects/02_attention_maps/draw_attention.py
```

The script uses the first test image of each CIFAR-10 class, so the set is the same on every run.

## Head statistics

```bash
uv run python projects/02_attention_maps/analyze_attention.py
```

This script measures two numbers for every head of every saved Project 1 ViT (seed 0), averaged over the first 1,000 test images (`vitcifar/attention_stats.py`):

- **Mean attention distance**: for each patch query, the pixel distance to each patch key, weighted by the attention (Dosovitskiy et al., 2021). The CLS key is left out and the patch weights are re-normalized. If a query attended to every patch equally, the value would be 16.5 px for patch 4 and 16.1 px for patch 8.
- **Entropy**: of each patch query's attention over all tokens, in nats. Uniform attention gives log(65) = 4.17 for patch 4 and log(17) = 2.83 for patch 8.

It runs in less than a minute on the GPU.

## Output

- `runs/02_attention_maps/head_statistics.png`: distance and entropy per head for the final ViT (`vit_p4_long`).
- `runs/02_attention_maps/distance_by_model.png`: distance per layer (mean of the heads) for each ViT step of Project 1.
- `runs/02_attention_maps/head_statistics.json`: the numbers behind both charts.
- `runs/02_attention_maps/layers_<i>_<class>.png`: for 4 images, a grid of layers (rows) by heads (columns). Each panel shows how much the CLS token attends to each patch. The last two columns show the mean of the heads and the attention rollout over all layers (Abnar and Zuidema, 2020).
- `projects/02_attention_maps/demo/data.js`: data for the browser demo (about 3 MB).

## Browser demo

Open `projects/02_attention_maps/demo/index.html` in a browser. No server is necessary.

- Move the pointer over a patch to see which patches it attends to.
- Click a patch to pin it. Move the pointer off the image to see the CLS token's attention.
- Select the layer and the head (or the mean of the heads).
- The strip below shows the same query at every layer, so you can see attention spread or focus with depth.

Each map is scaled to its own maximum. The demo shows the shape of the attention, not its absolute strength.

## Results

The final ViT (`vit_p4_long`, 92.9% test accuracy) classifies all 10 demo images correctly.

**The final ViT learns local heads early and global heads late.** In layer 0, the heads look almost everywhere (mean distance 14.7 px). In layer 1, the mean distance drops to 8.1 px, and three heads look at patches about 5 to 6 px away, which is about one patch. From layer 2 to layer 6, the distance rises again until it is close to uniform (15.9 px). The entropy follows the same shape: it is lowest in layer 1 (2.6 nats) and highest in the last layers (3.8 nats). Nothing in the architecture asks for this: the ViT has no convolution and learned position embeddings. It learns a local, convolution-like stage from the data.

**The ladder steps that raise accuracy also make the early layers more local.** The mean distance in layer 1, by step:

| Step | Test accuracy (3 seeds) | Layer 1 mean distance |
|------|-------------------------|-----------------------|
| `vit_p8_naive` | 57.3% | 15.8 px (uniform: 16.1) |
| `vit_p8_aug` | 61.6% | 15.3 px (uniform: 16.1) |
| `vit_p4_aug` | 77.7% | 14.0 px |
| `vit_p4_aug_warmup` | 88.6% | 12.0 px |
| `vit_p4_ls` | 88.7% | 12.5 px |
| `vit_p4_mix` | 88.7% | 10.1 px |
| `vit_p4_droppath` | 88.3% | 9.6 px |
| `vit_p4_long` | 92.9% | 8.1 px |

- The two patch-8 models stay close to uniform in every layer. Their entropy is low (0.3 to 0.65 nats in layer 1, out of 2.83), and only 1% to 15% of a patch's attention goes to CLS, so each query looks hard at a few patches that are not near it.
- `vit_p4_aug`, trained without warmup, has one local layer (layer 0, 8.5 px). Every later layer is close to uniform. The same model with warmup moves the local stage to layers 1 to 3.
- MixUp/CutMix and stochastic depth did not change the accuracy at 100 epochs, but they did make layer 1 more local (from 12.5 px to 10.1 px and 9.6 px). After 200 epochs, the model is the most local and the most accurate.

These are correlations from one checkpoint (seed 0) per step. They do not show that local heads cause the higher accuracy.

**In the browser demo**, a patch pinned on the bird in layer 1 attends to the patches next to it. In layer 6, the same patch attends to a few patches across the image. In the last layer, the CLS token of the airplane image attends to only a few patches: one on the airplane and one on the sign behind it. One image does not show whether CLS usually picks the object.
