"""Project 2: mean attention distance and entropy per head, for every saved Project 1 ViT.

Uses the first 1,000 CIFAR-10 test images. Writes a JSON file and two charts.

Examples:
    uv run python projects/02_attention_maps/analyze_attention.py
    uv run python projects/02_attention_maps/analyze_attention.py --images 200
"""

import argparse
import json
from pathlib import Path

import torch

from vitcifar import constants as C
from vitcifar.attention import AttentionExtractor
from vitcifar.attention_stats import AttentionStatistics, AttentionStatsPlotter, HeadStatistics
from vitcifar.data import Augmentation, Cifar10Data, CifarTransforms
from vitcifar.experiments.ablation import LadderCatalog, ModelKind, VitCheckpoint
from vitcifar.models.vit import VisionTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-dir", type=Path, default=C.PROJECT1_RUNS_DIR)
    parser.add_argument("--output-dir", type=Path, default=C.PROJECT2_RUNS_DIR)
    parser.add_argument("--images", type=int, default=C.P2_STATS_IMAGES)
    return parser.parse_args()


def measure(model: VisionTransformer, images: torch.Tensor, device: torch.device) -> HeadStatistics:
    """Head statistics of one model over normalized images, in batches."""
    stats = AttentionStatistics(model.config.grid_size, model.config.patch_size)
    extractor = AttentionExtractor(model, device)
    for start in range(0, len(images), C.P2_STATS_BATCH):
        stats.add(extractor.extract(images[start:start + C.P2_STATS_BATCH]).weights)
    return stats.result()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw, _ = Cifar10Data(C.DATA_DIR, 1, 0, Augmentation.NONE).load().raw_split(train=False)
    normalize = CifarTransforms().test()
    images = torch.stack([normalize(img) for img in raw[:args.images]])
    checkpoints = VitCheckpoint(args.checkpoint_dir)
    steps = [s for s in LadderCatalog().select(None) if s.model is ModelKind.VIT and checkpoints.path_for(s.name).exists()]
    series: dict[str, HeadStatistics] = {}
    for step in steps:
        stats = measure(checkpoints.load(step.name, device), images, device)
        series[step.label] = stats
        print(f"{step.name:20s} distance by layer: {[round(float(d), 1) for d in stats.layer_distance]}")
    final = next(s for s in steps if s.name == C.P1_FINAL_VIT_STEP)
    plotter, out = AttentionStatsPlotter(), args.output_dir
    print("wrote", plotter.plot_heads(series[final.label], "How far and how widely each head looks (final ViT, all fixes, 200 epochs)",
                                      out / "head_statistics.png"))
    print("wrote", plotter.plot_models(series, out / "distance_by_model.png"))
    report = {label: {"distance": s.distance.tolist(), "entropy": s.entropy.tolist(), "images": s.images,
                      "max_entropy": s.max_entropy, "uniform_distance": s.uniform_distance} for label, s in series.items()}
    (out / "head_statistics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
