"""Project 1 side study: peak learning rate with and without warmup.

Trains the ViT p4 step with augmentation at several learning rates, once with no
warmup and once with a short warmup. Finished runs are skipped.

Examples:
    uv run python projects/01_vit_from_scratch/run_lr_sweep.py --smoke
    uv run python projects/01_vit_from_scratch/run_lr_sweep.py
    uv run python projects/01_vit_from_scratch/run_lr_sweep.py --plot-only
"""

import argparse

import torch

from vitcifar import constants as C
from vitcifar.experiments.ablation import AblationRunner, LadderCatalog, LadderSettings, SweepPlanner
from vitcifar.experiments.ablation_plots import SweepPlotter
from vitcifar.training import ResultStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=C.P1_SWEEP_EPOCHS)
    parser.add_argument("--workers", type=int, default=C.DEFAULT_NUM_WORKERS)
    parser.add_argument("--force", action="store_true", help="retrain runs that already have results")
    parser.add_argument("--smoke", action="store_true", help="2 epochs on 2,048 images, to a separate folder")
    parser.add_argument("--plot-only", action="store_true", help="redraw the chart from saved results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = LadderSettings(num_workers=args.workers)
    epochs, output_dir = args.epochs, C.PROJECT1_RUNS_DIR / "lr_sweep"
    if args.smoke:
        settings, epochs, output_dir = settings.smoke(), C.P1_SWEEP_SMOKE_EPOCHS + 1, output_dir / "smoke"
    warmup = min(C.P1_SWEEP_WARMUP_EPOCHS, epochs - 1)
    planner = SweepPlanner(C.P1_SWEEP_LRS, warmup, epochs)
    specs = planner.plan(LadderCatalog().get(C.P1_SWEEP_STEP))
    if not args.plot_only:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        AblationRunner(settings, C.DATA_DIR, output_dir, device).run(specs, force=args.force)
    store = ResultStore(output_dir)
    results = {(s.lr, s.warmup_epochs): store.load(s.run_name) for s in specs if store.path_for(s.run_name).exists()}
    print("wrote", SweepPlotter().plot(results, output_dir / "lr_sweep.png"))


if __name__ == "__main__":
    main()
