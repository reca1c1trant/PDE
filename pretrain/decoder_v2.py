"""
V2 Patchify Decoder for PDE Foundation Model.

Architecture:
    Input [B, T*n_h*n_w, D]
    -> Reshape to [B, T, n_h, n_w, D]
    -> Project to stem channels
    -> CNN Decoder (1x1 -> 16x16)
    -> Reassemble patches
    -> Output [B, T, H, W, C]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from pretrain.attention_v2 import RMSNorm


class ResBlock2DTranspose(nn.Module):
    """Residual block for decoder (same as encoder ResBlock)."""

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


class CNNDecoder(nn.Module):
    """
    CNN Decoder: 1x1 -> 16x16 with ResBlocks.

    Architecture:
        1x1 -> ConvT -> 2x2 -> ResBlock
            -> ConvT -> 4x4 -> ResBlock
            -> ConvT -> 8x8 -> ResBlock
            -> ConvT -> 16x16
    """

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 128,
        out_channels: int = 18,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.decoder = nn.Sequential(
            # 1x1 -> 2x2
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2),
            nn.GELU(),
            ResBlock2DTranspose(in_channels),
            # 2x2 -> 4x4
            nn.ConvTranspose2d(in_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            ResBlock2DTranspose(hidden_channels),
            # 4x4 -> 8x8
            nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            ResBlock2DTranspose(hidden_channels),
            # 8x8 -> 16x16
            nn.ConvTranspose2d(hidden_channels, out_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, in_channels, 1, 1] where N = B*T*n_h*n_w
        Returns:
            out: [N, out_channels, 16, 16]
        """
        return self.decoder(x)


class CNNDecoderFrom4x4(nn.Module):
    """
    Alternative CNN Decoder starting from 4x4.

    Use this if encoder outputs 4x4 instead of 1x1.

    Architecture:
        4x4 -> ResBlock -> ConvT -> 8x8 -> ResBlock
            -> ConvT -> 16x16
    """

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 128,
        out_channels: int = 18,
    ):
        super().__init__()
        self.decoder = nn.Sequential(
            # 4x4
            ResBlock2DTranspose(in_channels),
            # 4x4 -> 8x8
            nn.ConvTranspose2d(in_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            ResBlock2DTranspose(hidden_channels),
            # 8x8 -> 16x16
            nn.ConvTranspose2d(hidden_channels, out_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class PatchifyDecoder(nn.Module):
    """
    V2 Patchify Decoder.

    Converts token sequence back to spatial output.

    Args:
        out_channels: Output channels (e.g., 18 = 3 vector + 15 scalar)
        hidden_dim: Input hidden dimension from transformer
        patch_size: Size of each patch (default 16)
        stem_channels: Channels after projection (before CNN decoder)
        decoder_hidden: Hidden channels in CNN decoder
    """

    def __init__(
        self,
        out_channels: int = 18,
        hidden_dim: int = 768,
        patch_size: int = 16,
        stem_channels: int = 256,
        decoder_hidden: int = 128,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.stem_channels = stem_channels

        # Project from transformer hidden dim
        self.norm = RMSNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, stem_channels)

        # CNN Decoder: 1x1 -> 16x16
        self.cnn_decoder = CNNDecoder(
            in_channels=stem_channels,
            hidden_channels=decoder_hidden,
            out_channels=out_channels,
        )

    def forward(self, x: torch.Tensor, shape_info: Dict) -> torch.Tensor:
        """
        Args:
            x: [B, T*n_h*n_w, D] token sequence
            shape_info: dict with T, n_h, n_w, H, W

        Returns:
            output: [B, T, H, W, C]
        """
        B = x.shape[0]
        T = shape_info['T']
        n_h = shape_info['n_h']
        n_w = shape_info['n_w']
        H = shape_info.get('H', n_h * self.patch_size)
        W = shape_info.get('W', n_w * self.patch_size)
        P = self.patch_size

        # Step 1: Norm and project
        x = self.norm(x)
        x = self.proj(x)  # [B, T*n_h*n_w, stem_channels]

        # Step 2: Reshape to patches
        x = x.reshape(B, T, n_h, n_w, self.stem_channels)
        x = x.reshape(B * T * n_h * n_w, self.stem_channels, 1, 1)

        # Step 3: CNN decoder
        x = self.cnn_decoder(x)  # [B*T*n_h*n_w, out_channels, P, P]

        # Step 4: Reassemble patches
        # [B*T*n_h*n_w, C, P, P] -> [B, T, n_h, n_w, C, P, P]
        x = x.reshape(B, T, n_h, n_w, self.out_channels, P, P)
        # -> [B, T, n_h, P, n_w, P, C]
        x = x.permute(0, 1, 2, 5, 3, 6, 4)
        # -> [B, T, H, W, C]
        x = x.reshape(B, T, H, W, self.out_channels)

        return x


class PatchifyDecoderWithExpand(nn.Module):
    """
    Alternative decoder that expands 1x1 to 4x4 before CNN.

    More symmetric with encoder that uses AttentionPool.
    """

    def __init__(
        self,
        out_channels: int = 18,
        hidden_dim: int = 768,
        patch_size: int = 16,
        stem_channels: int = 256,
        decoder_hidden: int = 128,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.stem_channels = stem_channels

        # Project and expand
        self.norm = RMSNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, stem_channels * 16)  # Expand to 4x4

        # CNN Decoder from 4x4
        self.cnn_decoder = CNNDecoderFrom4x4(
            in_channels=stem_channels,
            hidden_channels=decoder_hidden,
            out_channels=out_channels,
        )

    def forward(self, x: torch.Tensor, shape_info: Dict) -> torch.Tensor:
        """Same interface as PatchifyDecoder."""
        B = x.shape[0]
        T = shape_info['T']
        n_h = shape_info['n_h']
        n_w = shape_info['n_w']
        H = shape_info.get('H', n_h * self.patch_size)
        W = shape_info.get('W', n_w * self.patch_size)
        P = self.patch_size

        # Norm and project with expansion
        x = self.norm(x)
        x = self.proj(x)  # [B, T*n_h*n_w, stem_channels*16]

        # Reshape to 4x4
        x = x.reshape(B * T * n_h * n_w, self.stem_channels, 4, 4)

        # CNN decoder from 4x4
        x = self.cnn_decoder(x)  # [B*T*n_h*n_w, out_channels, P, P]

        # Reassemble patches
        x = x.reshape(B, T, n_h, n_w, self.out_channels, P, P)
        x = x.permute(0, 1, 2, 5, 3, 6, 4)
        x = x.reshape(B, T, H, W, self.out_channels)

        return x


# =============================================================================
# 1D Decoder Components
# =============================================================================

class ResBlock1DTranspose(nn.Module):
    """Residual block for 1D decoder."""

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


class CNNDecoder1D(nn.Module):
    """
    CNN Decoder for 1D: 1 -> 16 with ResBlocks.

    Architecture:
        1 -> ConvT -> 2 -> ResBlock
          -> ConvT -> 4 -> ResBlock
          -> ConvT -> 8 -> ResBlock
          -> ConvT -> 16
    """

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 128,
        out_channels: int = 1,
    ):
        super().__init__()
        self.decoder = nn.Sequential(
            # 1 -> 2
            nn.ConvTranspose1d(in_channels, in_channels, kernel_size=2),
            nn.GELU(),
            ResBlock1DTranspose(in_channels),
            # 2 -> 4
            nn.ConvTranspose1d(in_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            ResBlock1DTranspose(hidden_channels),
            # 4 -> 8
            nn.ConvTranspose1d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            ResBlock1DTranspose(hidden_channels),
            # 8 -> 16
            nn.ConvTranspose1d(hidden_channels, out_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, in_channels, 1]
        Returns:
            out: [N, out_channels, 16]
        """
        return self.decoder(x)


class PatchifyDecoder1D(nn.Module):
    """
    1D Patchify Decoder.

    Converts token sequence back to 1D spatial output.

    Input [B, T*n_x, D]
    -> Project to stem channels
    -> CNN Decoder (1 -> 16)
    -> Reassemble patches
    -> Output [B, T, X, C]
    """

    def __init__(
        self,
        out_channels: int = 1,
        hidden_dim: int = 768,
        patch_size: int = 16,
        stem_channels: int = 256,
        decoder_hidden: int = 128,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.stem_channels = stem_channels

        # Project from transformer hidden dim
        self.norm = RMSNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, stem_channels)

        # CNN Decoder: 1 -> 16
        self.cnn_decoder = CNNDecoder1D(
            in_channels=stem_channels,
            hidden_channels=decoder_hidden,
            out_channels=out_channels,
        )

    def forward(self, x: torch.Tensor, shape_info: Dict) -> torch.Tensor:
        """
        Args:
            x: [B, T*n_x, D] token sequence
            shape_info: dict with T, n_x, X

        Returns:
            output: [B, T, X, C]
        """
        B = x.shape[0]
        T = shape_info['T']
        n_x = shape_info['n_x']
        X = shape_info.get('X', n_x * self.patch_size)
        P = self.patch_size

        # Step 1: Norm and project
        x = self.norm(x)
        x = self.proj(x)  # [B, T*n_x, stem_channels]

        # Step 2: Reshape to patches
        x = x.reshape(B * T * n_x, self.stem_channels, 1)

        # Step 3: CNN decoder
        x = self.cnn_decoder(x)  # [B*T*n_x, out_channels, P]

        # Step 4: Reassemble patches
        # [B*T*n_x, C, P] -> [B, T, n_x, C, P]
        x = x.reshape(B, T, n_x, self.out_channels, P)
        # -> [B, T, n_x, P, C]
        x = x.permute(0, 1, 2, 4, 3)
        # -> [B, T, X, C]
        x = x.reshape(B, T, X, self.out_channels)

        return x


# =============================================================================
# 3D Decoder Components
# =============================================================================

class ResBlock3DTranspose(nn.Module):
    """Residual block for 3D decoder."""

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


class CNNDecoder3D(nn.Module):
    """
    CNN Decoder for 3D: 1x1x1 -> 8x8x8 with ResBlocks.

    Architecture:
        1x1x1 -> ConvT -> 2x2x2 -> ResBlock
              -> ConvT -> 4x4x4 -> ResBlock
              -> ConvT -> 8x8x8
    """

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 128,
        out_channels: int = 1,
    ):
        super().__init__()
        self.decoder = nn.Sequential(
            # 1x1x1 -> 2x2x2
            nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2),
            nn.GELU(),
            ResBlock3DTranspose(in_channels),
            # 2x2x2 -> 4x4x4
            nn.ConvTranspose3d(in_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            ResBlock3DTranspose(hidden_channels),
            # 4x4x4 -> 8x8x8
            nn.ConvTranspose3d(hidden_channels, out_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, in_channels, 1, 1, 1]
        Returns:
            out: [N, out_channels, 8, 8, 8]
        """
        return self.decoder(x)


class PatchifyDecoder3D(nn.Module):
    """
    3D Patchify Decoder.

    Converts token sequence back to 3D spatial output.

    Input [B, T*n_d*n_h*n_w, D]
    -> Project to stem channels
    -> CNN Decoder (1x1x1 -> 8x8x8)
    -> Reassemble patches
    -> Output [B, T, D, H, W, C]
    """

    def __init__(
        self,
        out_channels: int = 1,
        hidden_dim: int = 768,
        patch_size: int = 8,
        stem_channels: int = 256,
        decoder_hidden: int = 128,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.stem_channels = stem_channels

        # Project from transformer hidden dim
        self.norm = RMSNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, stem_channels)

        # CNN Decoder: 1x1x1 -> 8x8x8
        self.cnn_decoder = CNNDecoder3D(
            in_channels=stem_channels,
            hidden_channels=decoder_hidden,
            out_channels=out_channels,
        )

    def forward(self, x: torch.Tensor, shape_info: Dict) -> torch.Tensor:
        """
        Args:
            x: [B, T*n_d*n_h*n_w, D] token sequence
            shape_info: dict with T, n_d, n_h, n_w, D_in, H, W

        Returns:
            output: [B, T, D_in, H, W, C]
        """
        B = x.shape[0]
        T = shape_info['T']
        n_d = shape_info['n_d']
        n_h = shape_info['n_h']
        n_w = shape_info['n_w']
        P = self.patch_size
        D_in = shape_info.get('D_in', n_d * P)
        H = shape_info.get('H', n_h * P)
        W = shape_info.get('W', n_w * P)

        # Step 1: Norm and project
        x = self.norm(x)
        x = self.proj(x)  # [B, T*n_d*n_h*n_w, stem_channels]

        # Step 2: Reshape to patches
        x = x.reshape(B * T * n_d * n_h * n_w, self.stem_channels, 1, 1, 1)

        # Step 3: CNN decoder
        x = self.cnn_decoder(x)  # [B*T*n_d*n_h*n_w, out_channels, P, P, P]

        # Step 4: Reassemble patches
        # [N, C, P, P, P] -> [B, T, n_d, n_h, n_w, C, P, P, P]
        x = x.reshape(B, T, n_d, n_h, n_w, self.out_channels, P, P, P)
        # -> [B, T, n_d, P, n_h, P, n_w, P, C]
        x = x.permute(0, 1, 2, 6, 3, 7, 4, 8, 5)
        # -> [B, T, D_in, H, W, C]
        x = x.reshape(B, T, D_in, H, W, self.out_channels)

        return x


def create_decoder_v2(config: Dict) -> PatchifyDecoder:
    """Create decoder from config dict."""
    model_cfg = config.get('model', {})
    decoder_cfg = model_cfg.get('decoder', {})

    return PatchifyDecoder(
        out_channels=model_cfg.get('in_channels', 18),
        hidden_dim=model_cfg.get('hidden_dim', 768),
        patch_size=model_cfg.get('patch_size', 16),
        stem_channels=decoder_cfg.get('stem_channels', 256),
        decoder_hidden=decoder_cfg.get('hidden_channels', 128),
    )


if __name__ == "__main__":
    """Test decoder."""
    print("=" * 60)
    print("Testing PatchifyDecoder")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test different configurations
    test_cases = [
        # (B, T, n_h, n_w, H, W)
        (2, 8, 8, 8, 128, 128),
        (2, 8, 16, 16, 256, 256),
        (2, 8, 8, 32, 128, 512),
        (1, 8, 32, 32, 512, 512),
    ]

    decoder = PatchifyDecoder(
        out_channels=18,
        hidden_dim=768,
        patch_size=16,
    ).to(device)

    print(f"\nDecoder parameters: {sum(p.numel() for p in decoder.parameters()):,}")
    print()

    for B, T, n_h, n_w, H, W in test_cases:
        seq_len = T * n_h * n_w
        x = torch.randn(B, seq_len, 768, device=device)
        shape_info = {'T': T, 'n_h': n_h, 'n_w': n_w, 'H': H, 'W': W}

        output = decoder(x, shape_info)

        print(f"Tokens: [{B}, {seq_len}, 768]")
        print(f"  -> Output: {output.shape}")
        assert output.shape == (B, T, H, W, 18), f"Shape mismatch!"
        print(f"  ✓ Passed")
        print()

    # Test alternative decoder
    print("Testing PatchifyDecoderWithExpand:")
    decoder_expand = PatchifyDecoderWithExpand(
        out_channels=18,
        hidden_dim=768,
        patch_size=16,
    ).to(device)

    x = torch.randn(2, 8 * 8 * 8, 768, device=device)
    shape_info = {'T': 8, 'n_h': 8, 'n_w': 8, 'H': 128, 'W': 128}
    output = decoder_expand(x, shape_info)
    print(f"Tokens: {x.shape}")
    print(f"  -> Output: {output.shape}")
    print(f"  ✓ Expand decoder passed")

    print("\n" + "=" * 60)
    print("All decoder tests passed!")
    print("=" * 60)
