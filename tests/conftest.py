"""Shared fixtures. Tests run on CPU with tiny models and random tensors."""

import pytest
import torch

from vitcifar.models.vit import ViTConfig, VisionTransformer


@pytest.fixture(autouse=True)
def seed() -> None:
    torch.manual_seed(0)


@pytest.fixture
def tiny_config() -> ViTConfig:
    return ViTConfig(image_size=16, patch_size=4, num_classes=3, dim=32, depth=2, num_heads=4)


@pytest.fixture
def tiny_vit(tiny_config: ViTConfig) -> VisionTransformer:
    return VisionTransformer(tiny_config).eval()
