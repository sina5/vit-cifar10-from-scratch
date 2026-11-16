import pytest
import torch

from vitcifar.errors import ConfigError
from vitcifar.models.vit import DropPath, ViTConfig, VisionTransformer

TINY = dict(image_size=16, patch_size=4, num_classes=3, dim=32, depth=2, num_heads=4)


def test_forward_returns_one_logit_row_per_image(tiny_vit: VisionTransformer) -> None:
    logits = tiny_vit(torch.randn(5, 3, 16, 16))
    assert logits.shape == (5, 3)


def test_attention_has_one_map_per_layer_with_cls_token(tiny_vit: VisionTransformer) -> None:
    _, maps = tiny_vit.forward_with_attention(torch.randn(2, 3, 16, 16))
    assert len(maps) == 2
    assert maps[0].shape == (2, 4, 17, 17)


def test_attention_rows_sum_to_one(tiny_vit: VisionTransformer) -> None:
    _, maps = tiny_vit.forward_with_attention(torch.randn(2, 3, 16, 16))
    assert torch.allclose(maps[1].sum(dim=-1), torch.ones(2, 4, 17), atol=1e-5)


def test_explicit_attention_matches_fused_kernel(tiny_vit: VisionTransformer) -> None:
    images = torch.randn(3, 3, 16, 16)
    fused = tiny_vit(images)
    explicit, _ = tiny_vit.forward_with_attention(images)
    assert torch.allclose(fused, explicit, atol=1e-5)


def test_config_counts_patches(tiny_config: ViTConfig) -> None:
    assert (tiny_config.grid_size, tiny_config.num_patches) == (4, 16)


@pytest.mark.parametrize("overrides", [{"patch_size": 5}, {"patch_size": 0}, {"num_heads": 5}, {"depth": 0}])
def test_config_rejects_inconsistent_shapes(overrides: dict) -> None:
    with pytest.raises(ConfigError):
        ViTConfig(**(TINY | overrides))


def test_drop_path_rate_rises_linearly_over_depth() -> None:
    config = ViTConfig(**(TINY | {"depth": 3, "drop_path_rate": 0.2}))
    assert [config.drop_path_for(i) for i in range(3)] == pytest.approx([0.0, 0.1, 0.2])


def test_drop_path_is_identity_in_eval_mode() -> None:
    layer = DropPath(0.5).eval()
    x = torch.randn(8, 4)
    assert torch.equal(layer(x), x)


def test_drop_path_zeroes_whole_samples_in_training() -> None:
    out = DropPath(0.5).train()(torch.ones(256, 3))
    rows = set(out.sum(dim=1).tolist())
    assert rows == {0.0, 6.0}


def test_config_rejects_drop_path_rate_of_one() -> None:
    with pytest.raises(ConfigError):
        ViTConfig(**(TINY | {"drop_path_rate": 1.0}))
