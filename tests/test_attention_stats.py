import math
from pathlib import Path

import pytest
import torch

from vitcifar.attention_stats import AttentionStatistics, AttentionStatsPlotter, PatchDistances
from vitcifar.errors import ConfigError

GRID, PATCH = 2, 4  # 4 patches + CLS = 5 tokens
TOKENS = GRID * GRID + 1


def _uniform(layers: int = 2, batch: int = 3, heads: int = 2) -> torch.Tensor:
    return torch.full((layers, batch, heads, TOKENS, TOKENS), 1.0 / TOKENS)


def test_patch_distances_are_pixel_distances_between_centers() -> None:
    matrix = PatchDistances(GRID, PATCH).matrix
    assert matrix[0, 1] == 4.0 and matrix[0, 3] == pytest.approx(4.0 * math.sqrt(2)) and matrix[2, 2] == 0.0


def test_patch_distances_reject_bad_sizes() -> None:
    with pytest.raises(ConfigError):
        PatchDistances(0, 4)


def test_uniform_attention_gives_maximum_entropy_and_the_uniform_distance() -> None:
    stats = AttentionStatistics(GRID, PATCH)
    stats.add(_uniform())
    result = stats.result()
    assert result.entropy == pytest.approx(math.log(TOKENS))
    assert result.distance == pytest.approx(result.uniform_distance)
    assert result.images == 3 and result.distance.shape == (2, 2)


def test_self_attention_gives_zero_distance_and_zero_entropy() -> None:
    weights = torch.eye(TOKENS).expand(1, 2, 1, TOKENS, TOKENS)
    stats = AttentionStatistics(GRID, PATCH)
    stats.add(weights)
    result = stats.result()
    assert result.distance.item() == 0.0 and result.entropy.item() == pytest.approx(0.0, abs=1e-6)


def test_cls_weight_is_ignored_for_distance() -> None:
    """A query that splits its attention between CLS and its own patch is still local."""
    weights = torch.zeros(1, 1, 1, TOKENS, TOKENS)
    for q in range(1, TOKENS):
        weights[..., q, 0] = 0.5
        weights[..., q, q] = 0.5
    stats = AttentionStatistics(GRID, PATCH)
    stats.add(weights)
    assert stats.result().distance.item() == 0.0


def test_batches_are_averaged_by_image_count() -> None:
    stats = AttentionStatistics(GRID, PATCH)
    stats.add(_uniform(batch=1))
    stats.add(torch.eye(TOKENS).expand(2, 3, 2, TOKENS, TOKENS))
    assert stats.result().distance[0, 0] == pytest.approx(stats.result().uniform_distance / 4)


def test_statistics_reject_wrong_token_count_and_empty_result() -> None:
    stats = AttentionStatistics(GRID, PATCH)
    with pytest.raises(ConfigError):
        stats.add(torch.zeros(1, 1, 1, 3, 3))
    with pytest.raises(ConfigError):
        stats.result()


def test_plots_are_written(tmp_path: Path) -> None:
    stats = AttentionStatistics(GRID, PATCH)
    stats.add(_uniform())
    result, plotter = stats.result(), AttentionStatsPlotter()
    paths = [plotter.plot_heads(result, "t", tmp_path / "a.png"),
             plotter.plot_models({"a": result, "b": result}, tmp_path / "b.png")]
    assert all(p.read_bytes()[:4] == b"\x89PNG" for p in paths)
