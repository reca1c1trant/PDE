"""
V2 Patchify Encoder for PDE Foundation Model.

Architecture:
    Input [B, T, H, W, C]
    -> Patchify (16x16 patches)
    -> CNN Stem + ResBlock (16x16 -> 4x4)
    -> Intra-patch Temporal Attention
    -> Attention Pool (4x4 -> 1x1)
    -> 2D Position Encoding
    -> Output [B, T*n_h*n_w, D]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math

from pretrain.attention_v2 import (
    IntraPatchTemporalAttention,
    IntraPatchTemporalAttention1D,
    IntraPatchTemporalAttention3D,
    AttentionPool,
    AttentionPool1D,
    AttentionPool3D,
    RMSNorm,
)


class ResBlock2D(nn.Module):
    """Residual block with GroupNorm."""

    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode='replicate'),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode='replicate'),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)


class CNNStem(nn.Module):
    """
    CNN Stem: 16x16 -> 4x4 with ResBlocks.

    Architecture:
        16x16 -> Conv s2 -> 8x8 -> ResBlock
              -> Conv s2 -> 4x4 -> ResBlock
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 256,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.stem = nn.Sequential(
            # 16x16 -> 8x8
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            ResBlock2D(hidden_channels),
            # 8x8 -> 4x4
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            ResBlock2D(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, C, 16, 16] where N = B*T*n_h*n_w
        Returns:
            out: [N, out_channels, 4, 4]
        """
        return self.stem(x)


class CNNStemWithPool(nn.Module):
    """
    Alternative CNN Stem with learnable downsampling to 1x1.

    Architecture:
        16x16 -> Conv s2 + ResBlock -> 8x8
              -> Conv s2 + ResBlock -> 4x4
              -> Conv s2 + ResBlock -> 2x2
              -> Conv s2            -> 1x1

    Use this if AttentionPool is not desired.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 256,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            # 16x16 -> 8x8
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            ResBlock2D(hidden_channels),
            # 8x8 -> 4x4
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            ResBlock2D(hidden_channels),
            # 4x4 -> 2x2
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            ResBlock2D(out_channels),
            # 2x2 -> 1x1
            nn.Conv2d(out_channels, out_channels, kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, C, 16, 16]
        Returns:
            out: [N, out_channels, 1, 1]
        """
        return self.stem(x)


class PositionalEncoding2D(nn.Module):
    """
    Learnable 2D positional encoding for (t, h, w) positions.

    pos(t, h, w) = Embed_t(t) + Embed_h(h) + Embed_w(w)
    """

    def __init__(
        self,
        hidden_dim: int,
        max_t: int = 64,
        max_h: int = 64,
        max_w: int = 64,
    ):
        super().__init__()
        self.t_embed = nn.Embedding(max_t, hidden_dim)
        self.h_embed = nn.Embedding(max_h, hidden_dim)
        self.w_embed = nn.Embedding(max_w, hidden_dim)

        # Initialize with small values
        nn.init.normal_(self.t_embed.weight, std=0.02)
        nn.init.normal_(self.h_embed.weight, std=0.02)
        nn.init.normal_(self.w_embed.weight, std=0.02)

    def forward(self, T: int, n_h: int, n_w: int, device: torch.device) -> torch.Tensor:
        """
        Generate positional encoding.

        Args:
            T: Number of time steps
            n_h: Number of patches in height
            n_w: Number of patches in width
            device: Device to create tensors on

        Returns:
            pos: [T, n_h, n_w, D] positional encoding
        """
        t_ids = torch.arange(T, device=device)
        h_ids = torch.arange(n_h, device=device)
        w_ids = torch.arange(n_w, device=device)

        t_enc = self.t_embed(t_ids)[:, None, None, :]  # [T, 1, 1, D]
        h_enc = self.h_embed(h_ids)[None, :, None, :]  # [1, n_h, 1, D]
        w_enc = self.w_embed(w_ids)[None, None, :, :]  # [1, 1, n_w, D]

        pos = t_enc + h_enc + w_enc  # [T, n_h, n_w, D]
        return pos


class PatchifyEncoder(nn.Module):
    """
    V2 Patchify Encoder.

    Converts variable-size input to fixed-dimension token sequence.

    Args:
        in_channels: Input channels (e.g., 18 = 3 vector + 15 scalar)
        hidden_dim: Output hidden dimension for transformer
        patch_size: Size of each patch (default 16)
        stem_hidden: Hidden channels in CNN stem
        stem_out: Output channels of CNN stem (before projection)
        intra_patch_layers: Number of intra-patch attention layers
        intra_patch_window: Temporal window for intra-patch attention
        intra_patch_heads: Number of attention heads for intra-patch attention
        use_cnn_pool: If True, use CNN for pooling instead of AttentionPool
    """

    def __init__(
        self,
        in_channels: int = 18,
        hidden_dim: int = 768,
        patch_size: int = 16,
        stem_hidden: int = 128,
        stem_out: int = 256,
        intra_patch_layers: int = 2,
        intra_patch_window: int = 3,
        intra_patch_heads: int = 8,
        use_cnn_pool: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.use_cnn_pool = use_cnn_pool

        # CNN Stem: 16x16 -> 4x4
        if use_cnn_pool:
            # Use CNN all the way to 1x1
            self.cnn_stem = CNNStemWithPool(in_channels, stem_hidden, stem_out)
            self.intra_patch_attn = None
            self.attn_pool = None
        else:
            # Use CNN to 4x4, then attention
            self.cnn_stem = CNNStem(in_channels, stem_hidden, stem_out)

            # Intra-patch temporal attention
            self.intra_patch_attn = IntraPatchTemporalAttention(
                hidden_dim=stem_out,
                num_heads=intra_patch_heads,
                num_layers=intra_patch_layers,
                temporal_window=intra_patch_window,
            )

            # Attention pool: 4x4 -> 1x1
            self.attn_pool = AttentionPool(hidden_dim=stem_out)

        # Project to transformer hidden dim
        self.proj = nn.Linear(stem_out, hidden_dim)
        self.norm = RMSNorm(hidden_dim)

        # Positional encoding
        self.pos_enc = PositionalEncoding2D(hidden_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: [B, T, H, W, C] input tensor

        Returns:
            tokens: [B, T*n_h*n_w, D] token sequence
            shape_info: dict with T, n_h, n_w for decoder
        """
        B, T, H, W, C = x.shape
        P = self.patch_size

        assert H % P == 0 and W % P == 0, \
            f"H={H}, W={W} must be divisible by patch_size={P}"

        n_h = H // P
        n_w = W // P

        # Step 1: Patchify
        # [B, T, H, W, C] -> [B, T, n_h, P, n_w, P, C]
        x = x.reshape(B, T, n_h, P, n_w, P, C)
        # -> [B, T, n_h, n_w, P, P, C]
        x = x.permute(0, 1, 2, 4, 3, 5, 6)
        # -> [B*T*n_h*n_w, C, P, P]
        x = x.reshape(B * T * n_h * n_w, C, P, P)

        # Step 2: CNN Stem
        x = self.cnn_stem(x)  # [B*T*n_h*n_w, stem_out, 4, 4] or [.., 1, 1]

        if self.use_cnn_pool:
            # CNN already pooled to 1x1
            x = x.squeeze(-1).squeeze(-1)  # [B*T*n_h*n_w, stem_out]
            x = x.reshape(B, T, n_h, n_w, -1)  # [B, T, n_h, n_w, stem_out]
        else:
            # Reshape for intra-patch attention
            _, stem_out, h, w = x.shape  # h=w=4
            x = x.permute(0, 2, 3, 1)  # [B*T*n_h*n_w, 4, 4, stem_out]
            x = x.reshape(B, T, n_h, n_w, h, w, stem_out)

            # Step 3: Intra-patch temporal attention
            x = self.intra_patch_attn(x)  # [B, T, n_h, n_w, 4, 4, stem_out]

            # Step 4: Attention pool
            x = self.attn_pool(x)  # [B, T, n_h, n_w, stem_out]

        # Step 5: Project to hidden dim
        x = self.proj(x)  # [B, T, n_h, n_w, hidden_dim]
        x = self.norm(x)

        # Step 6: Add positional encoding
        pos = self.pos_enc(T, n_h, n_w, x.device)  # [T, n_h, n_w, D]
        x = x + pos.unsqueeze(0)  # [B, T, n_h, n_w, D]

        # Step 7: Flatten to sequence
        x = x.reshape(B, T * n_h * n_w, self.hidden_dim)

        shape_info = {'T': T, 'n_h': n_h, 'n_w': n_w, 'H': H, 'W': W}
        return x, shape_info


# =============================================================================
# 1D Encoder Components
# =============================================================================

class ResBlock1D(nn.Module):
    """Residual block with GroupNorm for 1D."""

    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 3, padding=1, padding_mode='replicate'),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 3, padding=1, padding_mode='replicate'),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)


class CNNStem1D(nn.Module):
    """
    CNN Stem for 1D: 16 -> 8 -> 4 with ResBlocks.

    Architecture:
        16 -> Conv s2 -> 8 -> ResBlock
           -> Conv s2 -> 4 -> ResBlock
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 256,
    ):
        super().__init__()
        self.out_channels = out_channels

        self.stem = nn.Sequential(
            # 16 -> 8
            nn.Conv1d(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            ResBlock1D(hidden_channels),
            # 8 -> 4
            nn.Conv1d(hidden_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            ResBlock1D(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, C, 16] where N = B*T*n_x
        Returns:
            out: [N, out_channels, 4]
        """
        return self.stem(x)


class PositionalEncoding1D(nn.Module):
    """
    Learnable 1D positional encoding for (t, x) positions.

    pos(t, x) = Embed_t(t) + Embed_x(x)
    """

    def __init__(
        self,
        hidden_dim: int,
        max_t: int = 64,
        max_x: int = 256,
    ):
        super().__init__()
        self.t_embed = nn.Embedding(max_t, hidden_dim)
        self.x_embed = nn.Embedding(max_x, hidden_dim)

        nn.init.normal_(self.t_embed.weight, std=0.02)
        nn.init.normal_(self.x_embed.weight, std=0.02)

    def forward(self, T: int, n_x: int, device: torch.device) -> torch.Tensor:
        """
        Returns:
            pos: [T, n_x, D] positional encoding
        """
        t_ids = torch.arange(T, device=device)
        x_ids = torch.arange(n_x, device=device)

        t_enc = self.t_embed(t_ids)[:, None, :]   # [T, 1, D]
        x_enc = self.x_embed(x_ids)[None, :, :]   # [1, n_x, D]

        pos = t_enc + x_enc  # [T, n_x, D]
        return pos


class PatchifyEncoder1D(nn.Module):
    """
    1D Patchify Encoder.

    Converts 1D PDE input to fixed-dimension token sequence.

    Input [B, T, X, C]
    -> Patchify (patch_size=16)
    -> CNN Stem (16 -> 8 -> 4)
    -> IntraPatchTemporalAttention (4 positions)
    -> AttentionPool (4 -> 1)
    -> Project + PosEnc
    -> Output [B, T*n_x, D]
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 768,
        patch_size: int = 16,
        stem_hidden: int = 128,
        stem_out: int = 256,
        intra_patch_layers: int = 2,
        intra_patch_window: int = 3,
        intra_patch_heads: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size

        # CNN Stem: 16 -> 8 -> 4
        self.cnn_stem = CNNStem1D(in_channels, stem_hidden, stem_out)

        # Intra-patch temporal attention
        self.intra_patch_attn = IntraPatchTemporalAttention1D(
            hidden_dim=stem_out,
            num_heads=intra_patch_heads,
            num_layers=intra_patch_layers,
            temporal_window=intra_patch_window,
        )

        # Attention pool: 4 -> 1
        self.attn_pool = AttentionPool1D(hidden_dim=stem_out)

        # Project to transformer hidden dim
        self.proj = nn.Linear(stem_out, hidden_dim)
        self.norm = RMSNorm(hidden_dim)

        # Positional encoding
        self.pos_enc = PositionalEncoding1D(hidden_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: [B, T, X, C] input tensor

        Returns:
            tokens: [B, T*n_x, D] token sequence
            shape_info: dict with T, n_x for decoder
        """
        B, T, X, C = x.shape
        P = self.patch_size

        assert X % P == 0, f"X={X} must be divisible by patch_size={P}"
        n_x = X // P

        # Step 1: Patchify
        # [B, T, X, C] -> [B, T, n_x, P, C]
        x = x.reshape(B, T, n_x, P, C)
        # -> [B*T*n_x, C, P]
        x = x.reshape(B * T * n_x, P, C).permute(0, 2, 1)

        # Step 2: CNN Stem
        x = self.cnn_stem(x)  # [B*T*n_x, stem_out, 4]

        # Reshape for intra-patch attention
        _, stem_out, sub_x = x.shape  # sub_x=4
        x = x.permute(0, 2, 1)  # [B*T*n_x, 4, stem_out]
        x = x.reshape(B, T, n_x, sub_x, stem_out)

        # Step 3: Intra-patch temporal attention
        x = self.intra_patch_attn(x)  # [B, T, n_x, 4, stem_out]

        # Step 4: Attention pool
        x = self.attn_pool(x)  # [B, T, n_x, stem_out]

        # Step 5: Project to hidden dim
        x = self.proj(x)  # [B, T, n_x, hidden_dim]
        x = self.norm(x)

        # Step 6: Add positional encoding
        pos = self.pos_enc(T, n_x, x.device)  # [T, n_x, D]
        x = x + pos.unsqueeze(0)  # [B, T, n_x, D]

        # Step 7: Flatten to sequence
        x = x.reshape(B, T * n_x, self.hidden_dim)

        shape_info = {'T': T, 'n_x': n_x, 'X': X}
        return x, shape_info


# =============================================================================
# 3D Encoder Components
# =============================================================================

class ResBlock3D(nn.Module):
    """Residual block with GroupNorm for 3D."""

    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, 3, padding=1, padding_mode='replicate'),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, 3, padding=1, padding_mode='replicate'),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)


class CNNStem3D(nn.Module):
    """
    CNN Stem for 3D: P^3 -> 2^3 with stride-2 Conv3d + ResBlocks.

    Dynamically builds log2(P)-1 stages:
        P=8:  8->4->2  (2 stages)
        P=16: 16->8->4->2  (3 stages)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 256,
        patch_size: int = 8,
    ):
        super().__init__()
        self.out_channels = out_channels

        import math
        num_stages = int(math.log2(patch_size)) - 1  # P=8: 2, P=16: 3

        layers = []
        ch = in_channels
        for i in range(num_stages):
            ch_out = out_channels if i == num_stages - 1 else hidden_channels
            layers.extend([
                nn.Conv3d(ch, ch_out, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                ResBlock3D(ch_out),
            ])
            ch = ch_out

        self.stem = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, C, P, P, P] where N = B*T*n_d*n_h*n_w
        Returns:
            out: [N, out_channels, 2, 2, 2]
        """
        return self.stem(x)


class PositionalEncoding3D(nn.Module):
    """
    Learnable 3D positional encoding for (t, d, h, w) positions.

    pos(t, d, h, w) = Embed_t(t) + Embed_d(d) + Embed_h(h) + Embed_w(w)
    """

    def __init__(
        self,
        hidden_dim: int,
        max_t: int = 64,
        max_d: int = 64,
        max_h: int = 64,
        max_w: int = 64,
    ):
        super().__init__()
        self.t_embed = nn.Embedding(max_t, hidden_dim)
        self.d_embed = nn.Embedding(max_d, hidden_dim)
        self.h_embed = nn.Embedding(max_h, hidden_dim)
        self.w_embed = nn.Embedding(max_w, hidden_dim)

        nn.init.normal_(self.t_embed.weight, std=0.02)
        nn.init.normal_(self.d_embed.weight, std=0.02)
        nn.init.normal_(self.h_embed.weight, std=0.02)
        nn.init.normal_(self.w_embed.weight, std=0.02)

    def forward(
        self, T: int, n_d: int, n_h: int, n_w: int, device: torch.device
    ) -> torch.Tensor:
        """
        Returns:
            pos: [T, n_d, n_h, n_w, D] positional encoding
        """
        t_ids = torch.arange(T, device=device)
        d_ids = torch.arange(n_d, device=device)
        h_ids = torch.arange(n_h, device=device)
        w_ids = torch.arange(n_w, device=device)

        t_enc = self.t_embed(t_ids)[:, None, None, None, :]  # [T, 1, 1, 1, D]
        d_enc = self.d_embed(d_ids)[None, :, None, None, :]  # [1, n_d, 1, 1, D]
        h_enc = self.h_embed(h_ids)[None, None, :, None, :]  # [1, 1, n_h, 1, D]
        w_enc = self.w_embed(w_ids)[None, None, None, :, :]  # [1, 1, 1, n_w, D]

        pos = t_enc + d_enc + h_enc + w_enc  # [T, n_d, n_h, n_w, D]
        return pos


class PatchifyEncoder3D(nn.Module):
    """
    3D Patchify Encoder.

    Converts 3D PDE input to fixed-dimension token sequence.

    Input [B, T, D, H, W, C]
    -> Patchify (patch_size=8)
    -> CNN Stem (8^3 -> 4^3 -> 2^3)
    -> IntraPatchTemporalAttention (8 positions)
    -> AttentionPool (8 -> 1)
    -> Project + PosEnc
    -> Output [B, T*n_d*n_h*n_w, D]
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 768,
        patch_size: int = 8,
        stem_hidden: int = 128,
        stem_out: int = 256,
        intra_patch_layers: int = 2,
        intra_patch_window: int = 3,
        intra_patch_heads: int = 8,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.gradient_checkpointing = gradient_checkpointing

        # CNN Stem: P^3 -> 2^3
        self.cnn_stem = CNNStem3D(in_channels, stem_hidden, stem_out, patch_size=patch_size)

        # Intra-patch temporal attention
        self.intra_patch_attn = IntraPatchTemporalAttention3D(
            hidden_dim=stem_out,
            num_heads=intra_patch_heads,
            num_layers=intra_patch_layers,
            temporal_window=intra_patch_window,
        )

        # Attention pool: 8 -> 1
        self.attn_pool = AttentionPool3D(hidden_dim=stem_out)

        # Project to transformer hidden dim
        self.proj = nn.Linear(stem_out, hidden_dim)
        self.norm = RMSNorm(hidden_dim)

        # Positional encoding
        self.pos_enc = PositionalEncoding3D(hidden_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: [B, T, D, H, W, C] input tensor

        Returns:
            tokens: [B, T*n_d*n_h*n_w, D] token sequence
            shape_info: dict with T, n_d, n_h, n_w for decoder
        """
        B, T, D_in, H, W, C = x.shape
        P = self.patch_size

        assert D_in % P == 0 and H % P == 0 and W % P == 0, \
            f"D={D_in}, H={H}, W={W} must be divisible by patch_size={P}"

        n_d = D_in // P
        n_h = H // P
        n_w = W // P

        # Step 1: Patchify
        # [B, T, D, H, W, C] -> [B, T, n_d, P, n_h, P, n_w, P, C]
        x = x.reshape(B, T, n_d, P, n_h, P, n_w, P, C)
        # -> [B, T, n_d, n_h, n_w, P, P, P, C]
        x = x.permute(0, 1, 2, 4, 6, 3, 5, 7, 8)
        # -> [B*T*n_d*n_h*n_w, C, P, P, P]
        x = x.reshape(B * T * n_d * n_h * n_w, P, P, P, C)
        x = x.permute(0, 4, 1, 2, 3)  # [N, C, P, P, P]

        # Step 2: CNN Stem (checkpoint to save ~8GB activation memory for 3D)
        if self.training and self.gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(self.cnn_stem, x, use_reentrant=False)
        else:
            x = self.cnn_stem(x)  # [N, stem_out, 2, 2, 2]

        # Reshape for intra-patch attention
        _, stem_out, sd, sh, sw = x.shape  # sd=sh=sw=2
        x = x.permute(0, 2, 3, 4, 1)  # [N, 2, 2, 2, stem_out]
        x = x.reshape(B, T, n_d, n_h, n_w, sd, sh, sw, stem_out)

        # Step 3: Intra-patch temporal attention
        x = self.intra_patch_attn(x)  # [B, T, n_d, n_h, n_w, 2, 2, 2, stem_out]

        # Step 4: Attention pool
        x = self.attn_pool(x)  # [B, T, n_d, n_h, n_w, stem_out]

        # Step 5: Project to hidden dim
        x = self.proj(x)  # [B, T, n_d, n_h, n_w, hidden_dim]
        x = self.norm(x)

        # Step 6: Add positional encoding
        pos = self.pos_enc(T, n_d, n_h, n_w, x.device)  # [T, n_d, n_h, n_w, D]
        x = x + pos.unsqueeze(0)  # [B, T, n_d, n_h, n_w, D]

        # Step 7: Flatten to sequence
        x = x.reshape(B, T * n_d * n_h * n_w, self.hidden_dim)

        shape_info = {
            'T': T, 'n_d': n_d, 'n_h': n_h, 'n_w': n_w,
            'D_in': D_in, 'H': H, 'W': W,
        }
        return x, shape_info


def create_encoder_v2(config: Dict) -> PatchifyEncoder:
    """Create encoder from config dict."""
    model_cfg = config.get('model', {})
    encoder_cfg = model_cfg.get('encoder', {})
    intra_cfg = model_cfg.get('intra_patch', {})

    return PatchifyEncoder(
        in_channels=model_cfg.get('in_channels', 18),
        hidden_dim=model_cfg.get('hidden_dim', 768),
        patch_size=model_cfg.get('patch_size', 16),
        stem_hidden=encoder_cfg.get('stem_hidden', 128),
        stem_out=encoder_cfg.get('stem_out', 256),
        intra_patch_layers=intra_cfg.get('num_layers', 2),
        intra_patch_window=intra_cfg.get('temporal_window', 3),
        intra_patch_heads=intra_cfg.get('num_heads', 8),
        use_cnn_pool=encoder_cfg.get('use_cnn_pool', False),
    )


if __name__ == "__main__":
    """Test encoder."""
    print("=" * 60)
    print("Testing PatchifyEncoder")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test different input sizes
    test_cases = [
        (2, 8, 128, 128, 18),   # Standard 128x128
        (2, 8, 256, 256, 18),   # 256x256
        (2, 8, 128, 512, 18),   # Non-square 128x512
        (1, 8, 512, 512, 18),   # Large 512x512
    ]

    encoder = PatchifyEncoder(
        in_channels=18,
        hidden_dim=768,
        patch_size=16,
        intra_patch_layers=2,
        intra_patch_window=3,
    ).to(device)

    print(f"\nEncoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    print()

    for B, T, H, W, C in test_cases:
        x = torch.randn(B, T, H, W, C, device=device)
        tokens, shape_info = encoder(x)

        n_h, n_w = H // 16, W // 16
        expected_tokens = T * n_h * n_w

        print(f"Input: [{B}, {T}, {H}, {W}, {C}]")
        print(f"  -> Tokens: {tokens.shape}")
        print(f"  -> shape_info: {shape_info}")
        assert tokens.shape == (B, expected_tokens, 768), f"Shape mismatch!"
        print(f"  ✓ Passed")
        print()

    # Test with CNN pool
    print("Testing with use_cnn_pool=True:")
    encoder_cnn = PatchifyEncoder(
        in_channels=18,
        hidden_dim=768,
        patch_size=16,
        use_cnn_pool=True,
    ).to(device)

    x = torch.randn(2, 8, 128, 128, 18, device=device)
    tokens, shape_info = encoder_cnn(x)
    print(f"Input: {x.shape}")
    print(f"  -> Tokens: {tokens.shape}")
    print(f"  ✓ CNN pool passed")

    print("\n" + "=" * 60)
    print("All encoder tests passed!")
    print("=" * 60)
