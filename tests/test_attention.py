import base64
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from vitcifar.attention import (
    AttentionExtractor, AttentionMapPlotter, AttentionMaps, AttentionRollout, DemoExporter, RowQuantizer,
)
from vitcifar.errors import ConfigError
from vitcifar.models.vit import VisionTransformer

JS_PREFIX = "window.ATTENTION_DATA = "


@pytest.fixture
def maps(tiny_vit: VisionTransformer) -> AttentionMaps:
    return AttentionExtractor(tiny_vit, torch.device("cpu")).extract(torch.randn(2, 3, 16, 16))


def test_extractor_stacks_layers_first(maps: AttentionMaps) -> None:
    assert maps.weights.shape == (2, 2, 4, 17, 17)
    assert maps.predictions.shape == (2,)


def test_cls_to_patches_is_a_grid_per_layer_and_head(maps: AttentionMaps) -> None:
    assert maps.cls_to_patches(0).shape == (2, 4, 4, 4)


def test_rollout_of_identity_attention_is_identity() -> None:
    eye = torch.eye(5).expand(3, 1, 2, 5, 5)
    assert torch.allclose(AttentionRollout().compute(eye), torch.eye(5).expand(1, 5, 5))


def test_rollout_rows_stay_normalized(maps: AttentionMaps) -> None:
    rollout = AttentionRollout().compute(maps.weights)
    assert torch.allclose(rollout.sum(dim=-1), torch.ones(2, 17), atol=1e-5)


def test_rollout_rejects_residual_outside_unit_range() -> None:
    with pytest.raises(ConfigError):
        AttentionRollout(residual=1.5)


def test_quantizer_maps_each_row_max_to_255() -> None:
    rows = np.array([[0.1, 0.2, 0.4], [0.0, 0.0, 0.0]])
    assert RowQuantizer().quantize(rows).tolist() == [[64, 128, 255], [0, 0, 0]]


def test_layer_plot_writes_png(maps: AttentionMaps, tmp_path: Path) -> None:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    path = AttentionMapPlotter(("a", "b", "c")).plot_layers(image, maps, 0, 1, tmp_path / "p.png")
    assert path.read_bytes()[:4] == b"\x89PNG"


def test_demo_export_holds_heads_plus_mean(maps: AttentionMaps, tmp_path: Path) -> None:
    images = np.zeros((2, 16, 16, 3), dtype=np.uint8)
    path = DemoExporter(("a", "b", "c")).export(images, [0, 2], maps, tmp_path / "data.js")
    payload = json.loads(path.read_text().removeprefix(JS_PREFIX).rstrip(";\n"))
    attention = base64.b64decode(payload["images"][0]["attention"])
    assert len(attention) == 2 * (4 + 1) * 17 * 17
    assert payload["images"][1]["label"] == 2
    assert payload["images"][0]["png"].startswith("data:image/png;base64,")


def test_demo_export_rejects_mismatched_counts(maps: AttentionMaps, tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        DemoExporter().export(np.zeros((3, 16, 16, 3), np.uint8), [0, 1, 2], maps, tmp_path / "d.js")
