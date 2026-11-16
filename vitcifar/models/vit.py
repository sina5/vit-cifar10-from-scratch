"""A small Vision Transformer written from scratch.

The attention layer can return its attention weights, which Project 2 uses to draw
attention maps. In normal training it uses the fused PyTorch kernel instead.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from vitcifar.errors import ConfigError

INIT_STD = 0.02


@dataclass(frozen=True)
class ViTConfig:
    """Shape of a Vision Transformer.

    Raises:
        ConfigError: if the image does not split into whole patches, or the width
            does not split into whole heads.
    """

    image_size: int
    patch_size: int
    num_classes: int
    dim: int
    depth: int
    num_heads: int
    mlp_ratio: float = 4.0
    in_channels: int = 3
    dropout: float = 0.0
    drop_path_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.patch_size <= 0 or self.image_size % self.patch_size != 0:
            raise ConfigError(
                f"image_size {self.image_size} must be a multiple of patch_size {self.patch_size}"
            )
        if self.num_heads <= 0 or self.dim % self.num_heads != 0:
            raise ConfigError(f"dim {self.dim} must be a multiple of num_heads {self.num_heads}")
        if min(self.depth, self.num_classes, self.in_channels) <= 0:
            raise ConfigError("depth, num_classes and in_channels must be positive")
        if not 0.0 <= self.drop_path_rate < 1.0:
            raise ConfigError("drop_path_rate must be in [0, 1)")

    def drop_path_for(self, layer: int) -> float:
        """Stochastic-depth rate of one block: rises linearly from 0 to drop_path_rate."""
        return self.drop_path_rate * layer / max(self.depth - 1, 1)

    @property
    def grid_size(self) -> int:
        """Number of patches along one side of the image."""
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        """Number of patch tokens (the CLS token is not included)."""
        return self.grid_size**2


class PatchEmbedding(nn.Module):
    """Cuts the image into patches and projects each patch to a token vector."""

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            config.in_channels, config.dim, kernel_size=config.patch_size, stride=config.patch_size
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Map (B, C, H, W) images to (B, num_patches, dim) tokens."""
        return self.proj(images).flatten(2).transpose(1, 2)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention that can also return its attention weights."""

    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply attention.

        Args:
            x: tokens of shape (B, T, dim).
            return_attention: if True, compute the weights explicitly and return them.

        Returns:
            The output tokens (B, T, dim) and the weights (B, heads, T, T) or None.
        """
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        drop = self.dropout if self.training else 0.0
        if return_attention:
            weights = (q @ k.transpose(-2, -1) * self.head_dim**-0.5).softmax(dim=-1)
            out = F.dropout(weights, drop, self.training) @ v
        else:
            weights = None
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop)
        out = out.transpose(1, 2).reshape(batch, tokens, dim)
        return self.proj(out), weights


class DropPath(nn.Module):
    """Stochastic depth: in training, skips the whole residual branch for some samples."""

    def __init__(self, rate: float) -> None:
        super().__init__()
        self.rate = rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Zero the branch for a random `rate` share of samples and rescale the rest."""
        if not self.training or self.rate == 0.0:
            return x
        keep = 1.0 - self.rate
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block: attention, then an MLP, each with a residual."""

    def __init__(self, config: ViTConfig, drop_path: float = 0.0) -> None:
        super().__init__()
        self.drop_path = DropPath(drop_path)
        hidden = int(config.dim * config.mlp_ratio)
        self.norm1 = nn.LayerNorm(config.dim)
        self.attn = MultiHeadSelfAttention(config.dim, config.num_heads, config.dropout)
        self.norm2 = nn.LayerNorm(config.dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.dim, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, config.dim),
            nn.Dropout(config.dropout),
        )

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the block. Returns the new tokens and, on request, the attention weights."""
        attended, weights = self.attn(self.norm1(x), return_attention)
        x = x + self.drop_path(attended)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, weights


class VisionTransformer(nn.Module):
    """ViT classifier with a learned CLS token and learned position embeddings."""

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbedding(config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, config.num_patches + 1, config.dim))
        self.blocks = nn.ModuleList(
            TransformerBlock(config, config.drop_path_for(i)) for i in range(config.depth)
        )
        self.norm = nn.LayerNorm(config.dim)
        self.head = nn.Linear(config.dim, config.num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=INIT_STD)
        nn.init.trunc_normal_(self.cls_token, std=INIT_STD)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=INIT_STD)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _embed(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(images)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        return torch.cat([cls, tokens], dim=1) + self.pos_embed

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return class logits of shape (B, num_classes)."""
        x = self._embed(images)
        for block in self.blocks:
            x, _ = block(x)
        return self.head(self.norm(x[:, 0]))

    def forward_with_attention(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return logits and one (B, heads, T, T) attention tensor per layer.

        Token 0 is the CLS token; tokens 1..T-1 are patches in row-major order.
        """
        x = self._embed(images)
        maps: list[torch.Tensor] = []
        for block in self.blocks:
            x, weights = block(x, return_attention=True)
            maps.append(weights)
        return self.head(self.norm(x[:, 0])), maps
