"""Project 2: draw the attention of the Project 1 ViT and export data for the hover demo.

Picks the first test image of each CIFAR-10 class, so the set is the same on every run.

Examples:
    uv run python projects/02_attention_maps/draw_attention.py
    uv run python projects/02_attention_maps/draw_attention.py --checkpoint-dir runs/01_vit_from_scratch/smoke
"""

import argparse
from pathlib import Path

import torch

from vitcifar import constants as C
from vitcifar.attention import AttentionExtractor, AttentionMapPlotter, DemoExporter
from vitcifar.data import Augmentation, Cifar10Data, CifarTransforms
from vitcifar.experiments.ablation import VitCheckpoint


class ClassSampler:
    """Selects the first image of each class from a labeled split."""

    def __init__(self, count: int) -> None:
        self.count = count

    def indices(self, labels: torch.Tensor) -> list[int]:
        """Return up to `count` indices, one per class, in class order."""
        picks = [int(torch.nonzero(labels == c)[0]) for c in labels.unique().tolist()]
        return picks[: self.count]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-dir", type=Path, default=C.PROJECT1_RUNS_DIR)
    parser.add_argument("--step", default=C.P1_FINAL_VIT_STEP, help="Project 1 step whose checkpoint to load")
    parser.add_argument("--output-dir", type=Path, default=C.PROJECT2_RUNS_DIR)
    parser.add_argument("--demo-file", type=Path, default=C.PROJECT2_DEMO_DIR / "data.js")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VitCheckpoint(args.checkpoint_dir).load(args.step, device)
    data = Cifar10Data(C.DATA_DIR, batch_size=1, num_workers=0, augmentation=Augmentation.NONE).load()
    images, labels = data.raw_split(train=False)
    picks = ClassSampler(C.P2_NUM_IMAGES).indices(labels)
    raw, chosen = images[picks], labels[picks]
    normalize = CifarTransforms().test()
    maps = AttentionExtractor(model, device).extract(torch.stack([normalize(img) for img in raw]))
    pictures = raw.permute(0, 2, 3, 1).numpy()
    plotter = AttentionMapPlotter()
    for i in range(min(C.P2_PLOT_IMAGES, len(picks))):
        name = C.CIFAR10_CLASSES[int(chosen[i])]
        print("wrote", plotter.plot_layers(pictures[i], maps, i, int(chosen[i]), args.output_dir / f"layers_{i}_{name}.png"))
    print("wrote", DemoExporter().export(pictures, chosen.tolist(), maps, args.demo_file))


if __name__ == "__main__":
    main()
