"""
EECS 445 - Introduction to Machine Learning
Winter 2025 - Project 2

Vision Transformer (ViT) implementation – GPU-friendly / vectorized version.
The architecture is the same as in the starter code:
    - patchify images
    - linear projection to tokens
    - prepend CLS token
    - add sinusoidal positional embeddings
    - several Transformer encoder blocks (MHA + MLP)
    - classify using the CLS token
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import numpy as np


__all__ = ["ViT"]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def patchify(images: torch.Tensor, n_patches: int) -> torch.Tensor:
    """
    Vectorized patchify: split square images into n_patches x n_patches patches.

    Args:
        images: (B, C, H, W) tensor, H == W is assumed.
        n_patches: number of patches along each spatial dimension.

    Returns:
        patches: (B, n_patches**2, patch_dim) tensor, where patch_dim =
                 C * (H/n_patches) * (W/n_patches).
    """
    B, C, H, W = images.shape
    assert H == W, "Patchify implemented for square images only"
    assert H % n_patches == 0, "Image size must be divisible by n_patches"

    patch_size = H // n_patches  # side length of each patch

    # unfold: (B, C, H, W) -> (B, C, n_patches, patch_size, n_patches, patch_size)
    patches = (
        images.unfold(2, patch_size, patch_size)
        .unfold(3, patch_size, patch_size)
    )  # (B, C, n_patches, n_patches, patch_size, patch_size)

    # move patch grid to front and flatten
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    patches = patches.view(B, n_patches * n_patches, -1)  # (B, N, C*ps*ps)
    return patches


def get_positional_embeddings(
    sequence_length: int,
    d: int,
    device: str,
    datatype: torch.dtype | None,
) -> torch.Tensor:
    """
    Standard sinusoidal positional embeddings (vectorized).

    Returns:
        (seq_len, d) tensor.
    """
    if datatype is None:
        datatype = torch.float32

    pe = torch.zeros(sequence_length, d, device=device, dtype=datatype)
    position = torch.arange(sequence_length, device=device, dtype=datatype).unsqueeze(1)  # (L, 1)
    div_term = torch.exp(
        torch.arange(0, d, 2, device=device, dtype=datatype)
        * (-math.log(10000.0) / d)
    )  # (d/2,)

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (L, d)


# --------------------------------------------------------------------------
# Transformer blocks
# --------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    """
    Standard vectorized multi-head self-attention.

    Input:  (B, N, C)
    Output: (B, N, C)
    """

    def __init__(self, num_features: int, num_heads: int) -> None:
        super().__init__()
        assert (
            num_features % num_heads == 0
        ), "num_features must be divisible by num_heads"

        self.num_features = num_features
        self.num_heads = num_heads
        self.head_dim = num_features // num_heads
        self.scale = self.head_dim ** -0.5

        # one big qkv projection, then we reshape/split into heads
        self.qkv = nn.Linear(num_features, num_features * 3)
        self.proj = nn.Linear(num_features, num_features)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape  # batch, tokens, features

        # (B, N, 3*C)
        qkv = self.qkv(x)

        # (B, N, 3, num_heads, head_dim) -> (3, B, num_heads, N, head_dim)
        qkv = qkv.view(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, num_heads, N, head_dim)

        # scaled dot-product attention: (B, num_heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.softmax(attn)

        # (B, num_heads, N, head_dim)
        out = attn @ v

        # merge heads: (B, N, num_heads * head_dim = C)
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        out = self.proj(out)
        return out


class TransformerEncoder(nn.Module):
    """
    One Transformer encoder block: LN -> MHA -> +res -> LN -> MLP -> +res
    """

    def __init__(self, hidden_d: int, n_heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()

        self.hidden_d = hidden_d

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mha = MultiHeadAttention(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        # MHA + residual
        x = x + self.mha(self.norm1(x))
        # MLP + residual
        x = x + self.mlp(self.norm2(x))
        return x


# --------------------------------------------------------------------------
# ViT
# --------------------------------------------------------------------------


class ViT(nn.Module):
    def __init__(
        self,
        num_patches: int,
        num_blocks: int,
        num_hidden: int,
        num_heads: int,
        num_classes: int = 8,
        chw_shape: Tuple[int, int, int] = (3, 64, 64),
        device: str = "cuda",
        datatype: torch.dtype | None = None,
    ) -> None:
        """
        Vision Transformer (ViT) model.

        Args:
            num_patches: number of patches per spatial dimension (P)
            num_blocks: number of Transformer encoder blocks
            num_hidden: embedding dimension (D)
            num_heads: number of attention heads
            num_classes: number of output classes
            chw_shape: (C, H, W) input image shape
        """
        super().__init__()

        self.chw = chw_shape
        self.num_patches = num_patches
        self.embedding_d = num_hidden
        self.num_heads = num_heads
        self.num_blocks = num_blocks

        C, H, W = chw_shape
        assert H == W, "ViT assumes square images"
        assert (
            H % num_patches == 0
        ), "Image size must be divisible by num_patches"

        self.patch_size = H // num_patches  # scalar
        self.flattened_patch_d = C * self.patch_size * self.patch_size  # patch_dim

        # 1) linear projection: patch -> token
        self.patch_to_token = nn.Linear(self.flattened_patch_d, self.embedding_d)

        # 2) learnable CLS token, shape (1, 1, D)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embedding_d))

        # 3) fixed sinusoidal positional encoding, shape (1, N+1, D)
        seq_len = num_patches**2 + 1  # +1 for CLS
        pos_embed = get_positional_embeddings(
            seq_len, self.embedding_d, device=device, datatype=datatype
        )
        self.pos_embed = nn.Parameter(pos_embed.unsqueeze(0), requires_grad=False)

        # 4) transformer encoder blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerEncoder(num_hidden, num_heads) for _ in range(num_blocks)]
        )

        # 5) classification head
        self.mlp1 = nn.Linear(self.embedding_d, num_classes)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: (B, C, H, W) batch of images.

        Returns:
            logits: (B, num_classes)
        """
        B, C, H, W = X.shape

        # ---- 1) patchify & linear projection ----
        patches = patchify(X, self.num_patches)  # (B, N, patch_dim)
        tokens = self.patch_to_token(patches)    # (B, N, D)

        # ---- 2) prepend CLS token ----
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls_tokens, tokens], dim=1)     # (B, N+1, D)

        # ---- 3) add positional embeddings ----
        x = x + self.pos_embed  # broadcast: (1, N+1, D) -> (B, N+1, D)

        # ---- 4) pass through Transformer encoders ----
        for block in self.transformer_blocks:
            x = block(x)  # (B, N+1, D)

        # ---- 5) classification head on CLS token ----
        cls_rep = x[:, 0, :]          # (B, D)
        logits = self.mlp1(cls_rep)   # (B, num_classes)

        return logits