"""Project 1: train the ablation ladder on CIFAR-10 over several seeds and chart it.

Finished runs are skipped, so you can stop the script and start it again.

Examples:
    uv run python projects/01_vit_from_scratch/run_ablation.py --smoke
    uv run python projects/01_vit_from_scratch/run_ablation.py
    uv run python projects/01_vit_from_scratch/run_ablation.py --steps vit_p4_mix --seeds 0
    uv run python projects/01_vit_from_scratch/run_ablation.py --plot-only
"""

import argparse

import torch

from vitcifar import constants as C
from vitcifar.experiments.ablation import AblationRunner, LadderCatalog, LadderPlanner, LadderSettings, SeedAggregator
from vitcifar.experiments.ablation_plots import LadderPlotter
from vitcifar.training import ResultStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", nargs="+", choices=LadderCatalog().names(), help="run only these steps")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(C.P1_SEEDS))
    parser.add_argument("--epochs", type=int, default=C.P1_EPOCHS, help="base epochs per step")
    parser.add_argument("--batch-size", type=int, default=C.P1_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=C.DEFAULT_NUM_WORKERS)
    parser.add_argument("--force", action="store_true", help="retrain runs that already have results")
    parser.add_argument("--smoke", action="store_true", help="2 epochs on 2,048 images, to a separate folder")
    parser.add_argument("--plot-only", action="store_true", help="redraw the charts from saved results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = LadderSettings(args.epochs, args.batch_size, args.workers, seeds=tuple(args.seeds))
    output_dir = C.PROJECT1_RUNS_DIR
    if args.smoke:
        settings, output_dir = settings.smoke(), output_dir / "smoke"
    catalog = LadderCatalog()
    if not args.plot_only:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        specs = LadderPlanner(settings.seeds).plan(catalog.select(args.steps))
        AblationRunner(settings, C.DATA_DIR, output_dir, device).run(specs, force=args.force)
    summaries = SeedAggregator(ResultStore(output_dir), settings.seeds).summarize(catalog.select(None))
    plotter = LadderPlotter()
    print("wrote", plotter.plot_curves(summaries, output_dir / "accuracy_curves.png"))
    print("wrote", plotter.plot_ladder(summaries, output_dir / "fix_ladder.png"))
    for summary in summaries:
        print(f"{summary.name:20s} {summary.mean * 100:6.2f} ± {summary.std * 100:4.2f}  (n={len(summary.finals)})")


if __name__ == "__main__":
    main()
