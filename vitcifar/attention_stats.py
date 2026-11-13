"""Project 2 statistics: how far, and how widely, each attention head looks.

Two numbers per head, averaged over many test images:

- Mean attention distance (Dosovitskiy et al., 2021, figure 7): for each patch query, the
  distance in pixels to each patch key, weighted by the attention. CLS is left out and the
  patch weights are re-normalized. A small value means a local head; a large value a global one.
- Entropy (in nats) of each patch query's attention over all tokens. A small value means the
  head focuses on a few tokens; log(T) is the maximum, for uniform attention.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure

from vitcifar import constants as C
from vitcifar.attention import CLS_INDEX
from vitcifar.errors import ConfigError
from vitcifar.plotting import LABEL_SIZE, TICK_SIZE, TITLE_SIZE, apply_style, require, save_png

EPS = 1e-12
SCATTER_SIZE = 50
LINE_WIDTH = 2.5
REFERENCE_HEADROOM = 0.06  # share of the axis kept above the reference line for its label
JITTER = 0.08  # horizontal offset between heads in the scatter, in layer units


class PatchDistances:
    """Pixel distances between patch centers on a square grid."""

    def __init__(self, grid_size: int, patch_size: int) -> None:
        if grid_size < 1 or patch_size < 1:
            raise ConfigError("grid_size and patch_size must be positive")
        rows, cols = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing="ij")
        centers = torch.stack([rows.flatten(), cols.flatten()], dim=1).float() * patch_size
        self.matrix = torch.cdist(centers, centers)  # (P, P)


@dataclass(frozen=True)
class HeadStatistics:
    """Per-head averages. Both arrays have shape (layers, heads)."""

    distance: np.ndarray
    entropy: np.ndarray
    images: int
    max_entropy: float
    uniform_distance: float  # mean distance if a query attended to every patch equally

    @property
    def layer_distance(self) -> np.ndarray:
        """Mean over heads, per layer."""
        return self.distance.mean(axis=1)


class AttentionStatistics:
    """Accumulates distance and entropy over batches of attention weights."""

    def __init__(self, grid_size: int, patch_size: int) -> None:
        self.distances = PatchDistances(grid_size, patch_size).matrix
        self._distance_sum: torch.Tensor | None = None
        self._entropy_sum: torch.Tensor | None = None
        self._images = 0
        self._tokens = grid_size * grid_size + 1

    def add(self, weights: torch.Tensor) -> None:
        """Add one batch of weights shaped (layers, batch, heads, T, T)."""
        if weights.shape[-1] != self._tokens:
            raise ConfigError(f"expected {self._tokens} tokens, got {weights.shape[-1]}")
        patch_queries = weights[..., CLS_INDEX + 1:, :]  # drop the CLS query
        to_patches = patch_queries[..., CLS_INDEX + 1:]
        to_patches = to_patches / to_patches.sum(dim=-1, keepdim=True).clamp_min(EPS)
        distance = (to_patches * self.distances).sum(dim=-1).mean(dim=-1).sum(dim=1)  # (layers, heads)
        entropy = -(patch_queries * patch_queries.clamp_min(EPS).log()).sum(dim=-1).mean(dim=-1).sum(dim=1)
        self._distance_sum = distance if self._distance_sum is None else self._distance_sum + distance
        self._entropy_sum = entropy if self._entropy_sum is None else self._entropy_sum + entropy
        self._images += weights.shape[1]

    def result(self) -> HeadStatistics:
        """The averages over every image added so far."""
        if not self._images:
            raise ConfigError("add at least one batch first")
        return HeadStatistics((self._distance_sum / self._images).numpy(), (self._entropy_sum / self._images).numpy(),
                              self._images, math.log(self._tokens), float(self.distances.mean()))


class AttentionStatsPlotter:
    """Charts for the statistics: one model per head, and several models per layer."""

    def plot_heads(self, stats: HeadStatistics, title: str, path: Path) -> Path:
        """Two panels: mean distance and entropy, one dot per head, a line for the head mean."""
        apply_style()
        fig = Figure(figsize=C.FIG_SIZE, layout="constrained")
        left, right = fig.subplots(1, 2)
        layers = np.arange(stats.distance.shape[0])
        for ax, values, label in ((left, stats.distance, "Mean attention distance (pixels)"),
                                  (right, stats.entropy, "Attention entropy (nats)")):
            for head in range(values.shape[1]):
                offset = (head - (values.shape[1] - 1) / 2) * JITTER
                ax.scatter(layers + offset, values[:, head], s=SCATTER_SIZE, color=C.PALETTE[head % len(C.PALETTE)],
                           label=f"head {head}", zorder=3)
            ax.plot(layers, values.mean(axis=1), color="black", linewidth=LINE_WIDTH, label="mean of heads")
            ax.set(xlabel="Layer", ylabel=label, xticks=layers)
        for ax, reference in ((left, stats.uniform_distance), (right, stats.max_entropy)):
            self._reference(ax, reference, layers[0])
        handles, labels = left.get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside right center", fontsize=TICK_SIZE, frameon=False)
        fig.suptitle(title, fontsize=TITLE_SIZE)
        left.set_title(f"Averaged over {stats.images:,} test images", fontsize=LABEL_SIZE)
        return save_png(fig, path)

    def plot_models(self, series: dict[str, HeadStatistics], path: Path) -> Path:
        """Mean attention distance per layer (mean of heads), one line per model."""
        require(list(series))
        apply_style()
        fig = Figure(figsize=C.FIG_SIZE, layout="constrained")
        ax = fig.subplots()
        for color, (label, stats) in zip(C.PALETTE, series.items()):
            layers = np.arange(len(stats.layer_distance))
            ax.plot(layers, stats.layer_distance, marker="o", linewidth=LINE_WIDTH, color=color, label=label)
        self._reference(ax, max(s.uniform_distance for s in series.values()), 0)
        ax.set(xlabel="Layer", ylabel="Mean attention distance (pixels, mean of heads)")
        fig.legend(loc="outside right center", fontsize=TICK_SIZE, frameon=False)
        fig.suptitle("Mean attention distance per layer for each Project 1 ViT", fontsize=TITLE_SIZE)
        return save_png(fig, path)

    @staticmethod
    def _reference(ax, value: float, left_edge: float) -> None:
        """A dotted line for uniform attention, labeled above it at the left, where no data lies."""
        ax.axhline(value, color="gray", linestyle=":", linewidth=LINE_WIDTH)
        ax.annotate("uniform attention", (left_edge, value), xytext=(0, 4), textcoords="offset points",
                    ha="left", va="bottom", fontsize=TICK_SIZE, color="gray")
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom, max(top, value + REFERENCE_HEADROOM * (value - bottom)))
