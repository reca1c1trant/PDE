"""
Spectral Decoder for SUT-FNO.

Features:
1. Multi-scale upsampling with ResBlocks
2. Spectral Fusion: IFFT of skip + Concat + Conv
3. FNO layer at each scale
4. FNO Refinement Head for final polish
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from encoder_spectral import SpectralConv2d, FNOLayer, ResBlock


class SpectralFusion(nn.Module):
    """
    Fuse decoder features with spectral skip connections.

    Process:
    1. IFFT spectral skip to spatial domain
    2. Concatenate with decoder features
    3. Conv to fuse
    """

    def __init__(self, decoder_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(decoder_channels + skip_channels, out_channels, 3, padding=1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.GELU(),
        )

    def forward(
        self,
        decoder_feat: torch.Tensor,
        spectral_skip: torch.Tensor,
        output_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Args:
            decoder_feat: [B, C_dec, H, W]
            spectral_skip: [B, C_skip, modes, modes//2+1] complex FFT coefficients
            output_size: (H, W) target spatial size

        Returns:
            fused: [B, C_out, H, W]
        """
        # IFFT to spatial domain
        skip_spatial = torch.fft.irfft2(spectral_skip, s=output_size)

        # Concat and fuse
        concat = torch.cat([decoder_feat, skip_spatial], dim=1)
        fused = self.fusion_conv(concat)

        return fused


class SpectralDecoderBlock(nn.Module):
    """
    Single decoder block with:
    1. Upsampling
    2. Spectral Fusion with skip connection
    3. ResBlock
    4. FNO layer
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        fno_modes: int,
        upsample: bool = True,
    ):
        super().__init__()
        self.upsample_flag = upsample

        # Upsampling
        if upsample:
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1),
                nn.GroupNorm(min(8, in_channels), in_channels),
                nn.GELU(),
            )
        else:
            self.upsample = nn.Identity()

        # Spectral fusion
        self.fusion = SpectralFusion(in_channels, skip_channels, out_channels)

        # ResBlock
        self.resblock = ResBlock(out_channels)

        # FNO layer
        self.fno = FNOLayer(out_channels, fno_modes)

    def forward(
        self,
        x: torch.Tensor,
        spectral_skip: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, H, W]
            spectral_skip: [B, C_skip, modes, modes//2+1]

        Returns:
            out: [B, C_out, H*2, W*2]
        """
        # Upsample
        x = self.upsample(x)
        H, W = x.shape[-2:]

        # Spectral fusion
        x = self.fusion(x, spectral_skip, (H, W))

        # ResBlock
        x = self.resblock(x)

        # FNO layer
        x = self.fno(x)

        return x


class FNORefinementHead(nn.Module):
    """
    Final refinement using FNO layers at full resolution.

    Uses residual connection for training stability:
    output = input + FNO(input)
    """

    def __init__(
        self,
        in_channels: int,
        modes: int = 32,
        width: int = 64,
        n_layers: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels

        # Lifting
        self.lift = nn.Conv2d(in_channels, width, 1)

        # Fourier layers
        self.fno_layers = nn.ModuleList([
            FNOLayer(width, modes) for _ in range(n_layers)
        ])

        # Projection
        self.proj = nn.Sequential(
            nn.Conv2d(width, width, 1),
            nn.GELU(),
            nn.Conv2d(width, in_channels, 1),
        )

        # Initialize projection to near-zero for stable training
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]

        Returns:
            refined: [B, C, H, W]
        """
        # Lift
        h = self.lift(x)

        # Fourier layers
        for layer in self.fno_layers:
            h = layer(h)

        # Project
        residual = self.proj(h)

        # Residual connection
        return x + residual


class SpectralDecoder(nn.Module):
    """
    Complete Spectral Decoder.

    Input:
        - latent: [B, T*16*16, hidden_dim] from transformer
        - spectral_skips: list of FFT coefficients from encoder

    Output: [B, T, H, W, C] where H=W=128
    """

    def __init__(
        self,
        out_channels: int = 6,
        channels: List[int] = [256, 128, 64],
        skip_channels: List[int] = [256, 128, 64],
        fno_modes: List[int] = [16, 24, 32],
        hidden_dim: int = 768,
        fno_head_modes: int = 32,
        fno_head_width: int = 64,
        fno_head_layers: int = 3,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.channels = channels
        self.hidden_dim = hidden_dim

        # Project from transformer hidden dim
        self.proj = nn.Linear(hidden_dim, channels[0])

        # Decoder blocks (reverse order of encoder)
        self.blocks = nn.ModuleList()

        # Block 1: 16 -> 32
        self.blocks.append(SpectralDecoderBlock(
            channels[0], skip_channels[0], channels[1], fno_modes[0], upsample=True
        ))

        # Block 2: 32 -> 64
        self.blocks.append(SpectralDecoderBlock(
            channels[1], skip_channels[1], channels[2], fno_modes[1], upsample=True
        ))

        # Block 3: 64 -> 128
        self.blocks.append(SpectralDecoderBlock(
            channels[2], skip_channels[2], channels[2], fno_modes[2], upsample=True
        ))

        # Final conv to output channels
        self.final_conv = nn.Sequential(
            nn.Conv2d(channels[2], out_channels, 3, padding=1),
        )

        # FNO Refinement Head
        self.fno_head = FNORefinementHead(
            in_channels=out_channels,
            modes=fno_head_modes,
            width=fno_head_width,
            n_layers=fno_head_layers,
        )

    def forward(
        self,
        latent: torch.Tensor,
        spectral_skips: List[torch.Tensor],
        temporal_length: int = 16,
    ) -> torch.Tensor:
        """
        Args:
            latent: [B, T*16*16, hidden_dim] from transformer
            spectral_skips: list of [B*T, C, modes, modes//2+1] from encoder
            temporal_length: T, number of timesteps

        Returns:
            output: [B, T, H, W, C] reconstructed tensor
        """
        B = latent.shape[0]
        T = temporal_length

        # Project from hidden dim
        x = self.proj(latent)  # [B, T*256, channels[0]]

        # Reshape to spatial: [B*T, C, 16, 16]
        x = x.reshape(B, T, 16, 16, -1)
        x = x.permute(0, 1, 4, 2, 3)  # [B, T, C, 16, 16]
        x = x.reshape(B * T, -1, 16, 16)

        # Decoder blocks with spectral skips (reverse order)
        for i, block in enumerate(self.blocks):
            skip_idx = len(spectral_skips) - 1 - i
            x = block(x, spectral_skips[skip_idx])

        # Final conv
        x = self.final_conv(x)  # [B*T, out_channels, 128, 128]

        # FNO refinement
        x = self.fno_head(x)

        # Reshape to output format: [B, T, H, W, C]
        _, C, H, W = x.shape
        x = x.reshape(B, T, C, H, W)
        x = x.permute(0, 1, 3, 4, 2)  # [B, T, H, W, C]

        return x


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Spectral Decoder")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create decoder
    decoder = SpectralDecoder(
        out_channels=6,
        channels=[256, 128, 64],
        skip_channels=[256, 128, 64],
        fno_modes=[16, 24, 32],
        hidden_dim=768,
        fno_head_modes=32,
        fno_head_width=64,
        fno_head_layers=3,
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"Total parameters: {total_params:,}")

    # Create mock inputs
    B, T = 2, 16
    latent = torch.randn(B, T * 16 * 16, 768).to(device)

    # Mock spectral skips (from encoder)
    spectral_skips = [
        torch.randn(B * T, 64, 32, 17, dtype=torch.cfloat).to(device),   # 64x64, modes=32
        torch.randn(B * T, 128, 16, 9, dtype=torch.cfloat).to(device),  # 32x32, modes=16
        torch.randn(B * T, 256, 8, 5, dtype=torch.cfloat).to(device),   # 16x16, modes=8
    ]

    # Forward pass
    output = decoder(latent, spectral_skips, temporal_length=T)

    print(f"\nLatent shape: {latent.shape}")
    print(f"Spectral skips: {[s.shape for s in spectral_skips]}")
    print(f"Output shape: {output.shape}")

    # Test gradient flow
    loss = output.sum()
    loss.backward()
    print("\nGradient test passed!")

    print("\nSpectral Decoder test passed!")
