"""Charts for Project 1: the ladder (over seeds), the learning-rate sweep, and data efficiency."""

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from vitcifar import constants as C
from vitcifar.experiments.ablation import StepSummary
from vitcifar.plotting import ANNOTATION_SIZE, HEADROOM, LABEL_SIZE, PERCENT, TICK_SIZE, TITLE_SIZE
from vitcifar.plotting import apply_style, require, save_png
from vitcifar.training import RunResult

BAND_ALPHA = 0.18
LINE_WIDTH = 2.2
SUBTITLE_COLOR = "dimgray"


def _is_resnet(name: str) -> bool:
    return name.startswith("resnet")


class LadderPlotter:
    """Accuracy curves and the what-each-fix-buys bar chart, as mean and spread over seeds."""

    def plot_curves(self, summaries: list[StepSummary], path: Path) -> Path:
        """Mean test accuracy per epoch for each step, with a band of one standard deviation."""
        require(summaries)
        apply_style()
        fig = Figure(figsize=C.FIG_SIZE, layout="constrained")
        ax = fig.subplots()
        for color, summary in zip(C.PALETTE, summaries):
            if summary.epochs.size == 0:
                continue
            mean, std = summary.curve_mean * PERCENT, summary.curve_std * PERCENT
            style = "--" if _is_resnet(summary.name) else "-"
            ax.plot(summary.epochs, mean, style, color=color, linewidth=LINE_WIDTH, label=summary.label)
            ax.fill_between(summary.epochs, mean - std, mean + std, color=color, alpha=BAND_ALPHA, linewidth=0)
        ax.set(xlabel="Epoch", ylabel="Test accuracy (%)")
        fig.suptitle("CIFAR-10 test accuracy as fixes are added to a small ViT", fontsize=TITLE_SIZE)
        ax.set_title(self._subtitle(summaries), fontsize=LABEL_SIZE, color=SUBTITLE_COLOR)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=TICK_SIZE, frameon=False)
        return save_png(fig, path)

    def plot_ladder(self, summaries: list[StepSummary], path: Path) -> Path:
        """Mean final accuracy per step with error bars and the gain over the previous ViT step."""
        require(summaries)
        apply_style()
        fig = Figure(figsize=C.FIG_SIZE, layout="constrained")
        ax = fig.subplots()
        means = [s.mean * PERCENT for s in summaries]
        stds = [s.std * PERCENT for s in summaries]
        colors = [C.RESNET_COLOR if _is_resnet(s.name) else C.VIT_COLOR for s in summaries]
        x = np.arange(len(summaries))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, width=0.68, error_kw={"elinewidth": 1.5})
        for bar, std, text in zip(bars, stds, self.bar_labels(summaries)):
            top = bar.get_height() + std
            ax.annotate(text, (bar.get_x() + bar.get_width() / 2, top), xytext=(0, 5),
                        textcoords="offset points", ha="center", fontsize=ANNOTATION_SIZE, fontweight="bold")
        ax.set_xticks(x, [s.label for s in summaries], rotation=20, ha="right")
        ax.set_ylim(0, max(means) * HEADROOM)
        ax.set_ylabel("Final test accuracy (%)")
        fig.suptitle("What each fix buys a small ViT on CIFAR-10", fontsize=TITLE_SIZE)
        ax.set_title(
            f"Red: ViT steps, each adds one fix to the one before. Blue: ResNet-18. {self._seed_text(summaries)}",
            fontsize=LABEL_SIZE, color=SUBTITLE_COLOR,
        )
        return save_png(fig, path)

    @staticmethod
    def bar_labels(summaries: list[StepSummary]) -> list[str]:
        """Mean accuracy per bar; ViT bars after the first also show the change in points."""
        labels, previous = [], None
        for summary in summaries:
            mean = summary.mean * PERCENT
            text = f"{mean:.1f}%"
            if not _is_resnet(summary.name):
                if previous is not None:
                    text += f"\n({mean - previous:+.1f})"
                previous = mean
            labels.append(text)
        return labels

    @staticmethod
    def _seed_text(summaries: list[StepSummary]) -> str:
        counts = sorted({len(s.finals) for s in summaries})
        seeds = "/".join(str(c) for c in counts)
        return f"Mean ± std over {seeds} seed(s)."

    def _subtitle(self, summaries: list[StepSummary]) -> str:
        longest = max(int(s.epochs.max()) for s in summaries if s.epochs.size)
        return f"{len(summaries)} steps, up to {longest} epochs. {self._seed_text(summaries)} Dashed: ResNet-18."


class SweepPlotter:
    """The learning-rate sweep: final accuracy against learning rate, and early training loss."""

    def plot(self, results: dict[tuple[float, int], RunResult], path: Path) -> Path:
        """Draw both panels from results keyed by (learning rate, warmup epochs)."""
        require(list(results))
        apply_style()
        fig = Figure(figsize=C.FIG_SIZE, layout="constrained")
        left, right = fig.subplots(1, 2)
        lrs = sorted({lr for lr, _ in results})
        warmups = sorted({w for _, w in results})
        colors = dict(zip(lrs, C.PALETTE))
        for warmup, marker in zip(warmups, ("o", "s")):
            points = [(lr, results[(lr, warmup)]) for lr in lrs if (lr, warmup) in results]
            accs = [(r.final_test_acc or 0.0) * PERCENT for _, r in points]
            label = "no warmup" if warmup == 0 else f"{warmup}-epoch warmup"
            left.plot([lr for lr, _ in points], accs, marker=marker, markersize=9, linewidth=LINE_WIDTH,
                      linestyle="--" if warmup == 0 else "-", color="black", label=label)
            for (lr, result), acc in zip(points, accs):
                note = "diverged" if result.diverged else f"{acc:.1f}%"
                left.annotate(note, (lr, acc), xytext=(0, 10 if warmup else -18), textcoords="offset points",
                              ha="center", fontsize=ANNOTATION_SIZE, color=colors[lr])
        left.set(xscale="log", xlabel="Peak learning rate", ylabel="Final test accuracy (%)",
                 title="Final accuracy")
        left.set_xticks(lrs, [f"{lr:g}" for lr in lrs])
        left.minorticks_off()
        left.legend(loc="lower left", fontsize=TICK_SIZE, frameon=False)
        self._loss_panel(right, results, colors)
        epochs = max(len(r.history) for r in results.values())
        fig.suptitle("Peak learning rate with and without warmup",
                     fontsize=TITLE_SIZE)
        fig.supxlabel(f"ViT p4 with augmentation, {epochs} epochs per run, one seed.",
                      fontsize=LABEL_SIZE, color=SUBTITLE_COLOR)
        return save_png(fig, path)

    @staticmethod
    def _loss_panel(ax, results: dict[tuple[float, int], RunResult], colors: dict[float, str]) -> None:
        for (lr, warmup), result in sorted(results.items()):
            epochs = [r.epoch for r in result.history]
            losses = [r.train_loss for r in result.history]
            ax.plot(epochs, losses, linestyle="--" if warmup == 0 else "-", color=colors[lr],
                    linewidth=LINE_WIDTH, label=f"lr {lr:g}" if warmup == 0 else None)
        ax.set(xlabel="Epoch", ylabel="Training loss", title="Training loss (dashed: no warmup)")
        ax.legend(loc="upper right", fontsize=TICK_SIZE, frameon=False)


class DataEfficiencyPlotter:
    """Test accuracy against the number of training images, one line per model."""

    def plot(self, series: dict[str, list[tuple[int, RunResult]]], path: Path,
             colors: dict[str, str] | None = None) -> Path:
        """Draw one line per label from (number of images, result) pairs; colors maps label to color."""
        require(list(series))
        apply_style()
        fig = Figure(figsize=C.FIG_SIZE, layout="constrained")
        ax = fig.subplots()
        all_sizes: set[int] = set()
        colors = colors or {}
        for default, (label, points) in zip(C.PALETTE, series.items()):
            color = colors.get(label, default)
            sizes = [n for n, _ in points]
            accs = [(r.final_test_acc or 0.0) * PERCENT for _, r in points]
            all_sizes.update(sizes)
            ax.plot(sizes, accs, marker="o", markersize=9, linewidth=LINE_WIDTH, color=color, label=label)
            for n, acc in zip(sizes, accs):
                ax.annotate(f"{acc:.1f}%", (n, acc), xytext=(8, -4), textcoords="offset points",
                            fontsize=ANNOTATION_SIZE, color=color)
        ticks = sorted(all_sizes)
        ax.set(xscale="log", xlabel="Training images", ylabel="Final test accuracy (%)")
        ax.set_xticks(ticks, [f"{n:,}" for n in ticks])
        ax.minorticks_off()
        ax.legend(loc="lower right", fontsize=TICK_SIZE, frameon=False)
        fig.suptitle("ViT versus ResNet-18 as the training set grows", fontsize=TITLE_SIZE)
        ax.set_title("Same number of optimizer steps at every size (smaller sets train for more epochs). One seed.",
                     fontsize=LABEL_SIZE, color=SUBTITLE_COLOR)
        return save_png(fig, path)
