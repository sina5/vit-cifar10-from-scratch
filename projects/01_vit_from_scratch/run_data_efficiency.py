"""Project 1 side study: test accuracy against training-set size, ViT versus ResNet-18.

Each model trains on 10%, 25%, 50% and 100% of CIFAR-10 (the same share of every
class) with the same number of optimizer steps. Finished runs are skipped.

Examples:
    uv run python projects/01_vit_from_scratch/run_data_efficiency.py --smoke
    uv run python projects/01_vit_from_scratch/run_data_efficiency.py
    uv run python projects/01_vit_from_scratch/run_data_efficiency.py --plot-only
"""

import argparse

import torch

from vitcifar import constants as C
from vitcifar.experiments.ablation import AblationRunner, DataEfficiencyPlanner, LadderCatalog, LadderSettings
from vitcifar.experiments.ablation_plots import DataEfficiencyPlotter
from vitcifar.training import ResultStore

RESNET_LABEL = "ResNet-18"
VIT_LABEL = "ViT (all fixes, 1x epochs)"
LINE_COLORS = {RESNET_LABEL: C.RESNET_COLOR, VIT_LABEL: C.VIT_COLOR}

CIFAR10_TRAIN_IMAGES = 50_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=C.P1_EPOCHS, help="epochs at 100%% of the data")
    parser.add_argument("--workers", type=int, default=C.DEFAULT_NUM_WORKERS)
    parser.add_argument("--force", action="store_true", help="retrain runs that already have results")
    parser.add_argument("--smoke", action="store_true", help="tiny run on 2,048 images, to a separate folder")
    parser.add_argument("--plot-only", action="store_true", help="redraw the chart from saved results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = LadderSettings(epochs=args.epochs, num_workers=args.workers)
    output_dir, total = C.PROJECT1_RUNS_DIR / "data_efficiency", CIFAR10_TRAIN_IMAGES
    if args.smoke:
        settings, output_dir, total = settings.smoke(), output_dir / "smoke", C.P1_SMOKE_SUBSET
    steps = LadderCatalog().select(list(C.P1_DATA_STEPS))
    specs = DataEfficiencyPlanner(C.P1_DATA_FRACTIONS, settings.epochs).plan(steps)
    if not args.plot_only:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        AblationRunner(settings, C.DATA_DIR, output_dir, device).run(specs, force=args.force)
    store = ResultStore(output_dir)
    series: dict[str, list] = {}
    for spec in specs:
        if store.path_for(spec.run_name).exists():
            label = RESNET_LABEL if spec.step.name.startswith("resnet") else VIT_LABEL
            series.setdefault(label, []).append((round(total * spec.train_fraction), store.load(spec.run_name)))
    print("wrote", DataEfficiencyPlotter().plot(series, output_dir / "data_efficiency.png", LINE_COLORS))


if __name__ == "__main__":
    main()
