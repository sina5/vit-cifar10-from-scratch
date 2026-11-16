"""Project 2: read, combine, draw and export the attention of a trained ViT."""

import base64
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image

from vitcifar import constants as C
from vitcifar.errors import ConfigError
from vitcifar.models.vit import VisionTransformer

CLS_INDEX = 0
OVERLAY_ALPHA = 0.55
HEATMAP_CMAP = "inferno"
GRID_FIG_SCALE = 1.25  # 16:9 figure, enlarged so a 7x8 grid of panels stays readable


@dataclass(frozen=True)
class AttentionMaps:
    """Attention for a batch of images.

    Attributes:
        weights: float tensor (layers, batch, heads, tokens, tokens). Token 0 is CLS.
        predictions: predicted class index per image.
        grid_size: patches along one side.
    """

    weights: torch.Tensor
    predictions: torch.Tensor
    grid_size: int

    @property
    def num_layers(self) -> int:
        """Number of Transformer blocks."""
        return self.weights.shape[0]

    @property
    def num_heads(self) -> int:
        """Heads per block."""
        return self.weights.shape[2]

    def cls_to_patches(self, image: int) -> torch.Tensor:
        """CLS-query attention on each patch: (layers, heads, grid, grid)."""
        row = self.weights[:, image, :, CLS_INDEX, 1:]
        return row.reshape(self.num_layers, self.num_heads, self.grid_size, self.grid_size)


class AttentionExtractor:
    """Runs a ViT in eval mode and collects every layer's attention weights."""

    def __init__(self, model: VisionTransformer, device: torch.device) -> None:
        self.model = model
        self.device = device

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> AttentionMaps:
        """Return the attention for normalized images of shape (B, C, H, W)."""
        self.model.eval()
        logits, maps = self.model.forward_with_attention(images.to(self.device))
        weights = torch.stack(maps).float().cpu()
        return AttentionMaps(weights, logits.argmax(dim=1).cpu(), self.model.config.grid_size)


class AttentionRollout:
    """Attention rollout (Abnar & Zuidema, 2020).

    Averages heads, adds the residual path as identity, re-normalizes rows, and
    multiplies the layers together. The CLS row of the product estimates how much
    each patch flows into the final CLS token.
    """

    def __init__(self, residual: float = C.P2_ROLLOUT_RESIDUAL) -> None:
        if not 0.0 <= residual <= 1.0:
            raise ConfigError("residual must be in [0, 1]")
        self.residual = residual

    def compute(self, weights: torch.Tensor) -> torch.Tensor:
        """Map (layers, batch, heads, T, T) weights to (batch, T, T) rollout."""
        mean = weights.mean(dim=2)
        eye = torch.eye(mean.shape[-1]).expand_as(mean[0])
        rollout = eye.clone()
        for layer in mean:
            mixed = self.residual * eye + (1.0 - self.residual) * layer
            mixed = mixed / mixed.sum(dim=-1, keepdim=True)
            rollout = mixed @ rollout
        return rollout

    def cls_map(self, weights: torch.Tensor, grid_size: int) -> torch.Tensor:
        """CLS-row rollout reshaped to (batch, grid, grid)."""
        rollout = self.compute(weights)[:, CLS_INDEX, 1:]
        return rollout.reshape(-1, grid_size, grid_size)


class AttentionMapPlotter:
    """Draws per-layer, per-head CLS attention over the input image."""

    def __init__(self, class_names: tuple[str, ...] = C.CIFAR10_CLASSES) -> None:
        self.class_names = class_names

    def plot_layers(self, image: np.ndarray, maps: AttentionMaps, index: int, label: int, path: Path) -> Path:
        """Grid of layers (rows) by heads plus mean and rollout (columns).

        Args:
            image: uint8 (H, W, 3) picture shown under the heatmaps.
            maps: attention for the batch that holds this image.
            index: position of the image in that batch.
            label: true class index.
            path: PNG output path.
        """
        cls = maps.cls_to_patches(index).numpy()
        rollout = AttentionRollout().cls_map(maps.weights[:, index : index + 1], maps.grid_size)[0].numpy()
        cols = maps.num_heads + 2
        width, height = C.FIG_SIZE
        fig = Figure(figsize=(width * GRID_FIG_SCALE, height * GRID_FIG_SCALE), layout="constrained")
        axes = fig.subplots(maps.num_layers, cols, squeeze=False)
        for layer in range(maps.num_layers):
            panels = [*cls[layer], cls[layer].mean(axis=0)]
            titles = [f"head {h}" for h in range(maps.num_heads)] + ["mean"]
            for col, (panel, title) in enumerate(zip(panels, titles)):
                self._overlay(axes[layer, col], image, panel, title if layer == 0 else None)
            axes[layer, 0].set_ylabel(f"layer {layer}", fontsize=11)
            rollout_ax = axes[layer, cols - 1]
            if layer == 0:
                self._overlay(rollout_ax, image, rollout, "rollout (all layers)")
            else:
                rollout_ax.set_axis_off()
        true, pred = self.class_names[label], self.class_names[int(maps.predictions[index])]
        fig.suptitle(f"CLS attention per layer and head. True: {true}, predicted: {pred}", fontsize=16)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=C.FIG_DPI, format="png")
        return path

    @staticmethod
    def _overlay(ax, image: np.ndarray, heat: np.ndarray, title: str | None) -> None:
        size = image.shape[0]
        ax.imshow(image, extent=(0, size, size, 0))
        ax.imshow(heat / max(heat.max(), 1e-12), cmap=HEATMAP_CMAP, alpha=OVERLAY_ALPHA,
                  extent=(0, size, size, 0), interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        if title:
            ax.set_title(title, fontsize=11)


class RowQuantizer:
    """Scales each attention row to its own maximum and stores it as uint8.

    The demo only draws relative strength inside one row, so per-row scaling keeps
    the useful precision and cuts the file size by four.
    """

    def __init__(self, levels: int = C.QUANT_LEVELS) -> None:
        self.levels = levels

    def quantize(self, rows: np.ndarray) -> np.ndarray:
        """Map float rows (..., T) to uint8 with each row's max at `levels`."""
        peak = rows.max(axis=-1, keepdims=True)
        scaled = rows / np.maximum(peak, 1e-12) * self.levels
        return np.clip(np.rint(scaled), 0, self.levels).astype(np.uint8)


class DemoExporter:
    """Writes `data.js` for the hover demo in projects/02_attention_maps/demo.

    The file sets `window.ATTENTION_DATA`, so the demo works from file:// with no server.
    Per image it stores the picture as a PNG data URL and a uint8 array of shape
    (layers, heads + 1, T, T), base64-encoded; head index `heads` is the head mean.
    """

    def __init__(self, class_names: tuple[str, ...] = C.CIFAR10_CLASSES) -> None:
        self.class_names = class_names
        self.quantizer = RowQuantizer()

    def export(self, images: np.ndarray, labels: list[int], maps: AttentionMaps, path: Path) -> Path:
        """Write the demo data file.

        Args:
            images: uint8 array (B, H, W, 3) of the original pictures.
            labels: true class per image.
            maps: attention for the same images, in the same order.
            path: output `.js` path.
        """
        if images.shape[0] != len(labels) or images.shape[0] != maps.weights.shape[1]:
            raise ConfigError("images, labels and attention must hold the same number of images")
        payload = {
            "layers": maps.num_layers, "heads": maps.num_heads, "grid": maps.grid_size,
            "tokens": maps.weights.shape[-1], "classes": list(self.class_names),
            "images": [self._image_entry(images[i], labels[i], maps, i) for i in range(len(labels))],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"window.ATTENTION_DATA = {json.dumps(payload)};\n", encoding="utf-8")
        return path

    def _image_entry(self, image: np.ndarray, label: int, maps: AttentionMaps, index: int) -> dict:
        per_head = maps.weights[:, index].numpy()
        with_mean = np.concatenate([per_head, per_head.mean(axis=1, keepdims=True)], axis=1)
        return {
            "png": self._png_data_url(image),
            "label": int(label),
            "pred": int(maps.predictions[index]),
            "attention": base64.b64encode(self.quantizer.quantize(with_mean).tobytes()).decode("ascii"),
        }

    @staticmethod
    def _png_data_url(image: np.ndarray) -> str:
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
