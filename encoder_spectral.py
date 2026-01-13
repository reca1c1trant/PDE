"""
Spectral Encoder for SUT-FNO.

Features:
1. Multi-scale downsampling with ResBlocks
2. FNO layer at each scale for frequency domain processing
3. Spectral Skip connections (FFT coefficients) for decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math


class SpectralConv2d(nn.Module):
    """
    2D Spectral Convolution Layer.

    Performs convolution in Fourier space:
    1. FFT to frequency domain
    2. Multiply with learnable complex weights
    3. IFFT back to spatial domain
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)

        # Complex weights for spectral convolution
        self.weights1 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Complex multiplication in Fourier space."""
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] tensor
        Returns:
            [B, out_channels, H, W] tensor
        """
        B, C, H, W = x.shape

        # Compute 2D FFT
        x_ft = torch.fft.rfft2(x)

        # Output tensor in Fourier space
        out_ft = torch.zeros(
            B, self.out_channels, H, W // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        # Multiply relevant Fourier modes
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2],
            self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2],
            self.weights2
        )

        # Inverse FFT
        x = torch.fft.irfft2(out_ft, s=(H, W))

        return x


class FNOLayer(nn.Module):
    """
    Single Fourier Layer = Spectral Conv + Local Conv + Activation.
    """

    def __init__(self, channels: int, modes: int):
        super().__init__()
        self.spectral_conv = SpectralConv2d(channels, channels, modes, modes)
        self.local_conv = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.spectral_conv(x)
        x2 = self.local_conv(x)
        return self.activation(self.norm(x1 + x2))


class ResBlock(nn.Module):
    """Residual block with GroupNorm."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.activation(x + residual)


class SpectralEncoderBlock(nn.Module):
    """
    Single encoder block with:
    1. Downsampling convolution
    2. ResBlock
    3. FNO layer
    4. Spectral skip extraction (FFT)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        fno_modes: int,
        spectral_skip_modes: int,
        downsample: bool = True,
    ):
        super().__init__()
        self.spectral_skip_modes = spectral_skip_modes

        # Downsampling
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.GELU(),
            )
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.GELU(),
            )

        # ResBlock
        self.resblock = ResBlock(out_channels)

        # FNO layer
        self.fno = FNOLayer(out_channels, fno_modes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, C_in, H, W]

        Returns:
            out: [B, C_out, H/2, W/2] downsampled features
            spectral_skip: [B, C_out, modes, modes//2+1] FFT coefficients
        """
        # Downsample
        x = self.downsample(x)

        # ResBlock
        x = self.resblock(x)

        # FNO layer
        x = self.fno(x)

        # Extract spectral skip (FFT coefficients)
        x_fft = torch.fft.rfft2(x)
        modes = self.spectral_skip_modes
        spectral_skip = x_fft[:, :, :modes, :modes//2+1].clone()

        return x, spectral_skip


class SpectralEncoder(nn.Module):
    """
    Complete Spectral Encoder.

    Input: [B, T, H, W, C] where H=W=128
    Output:
        - latent: [B, T*16*16, hidden_dim] for transformer
        - spectral_skips: list of FFT coefficients for decoder
    """

    def __init__(
        self,
        in_channels: int = 6,
        channels: List[int] = [64, 128, 256],
        fno_modes: List[int] = [32, 16, 8],
        spectral_skip_modes: List[int] = [32, 16, 8],
        hidden_dim: int = 768,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.hidden_dim = hidden_dim

        # Initial projection
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 3, padding=1),
            nn.GroupNorm(8, channels[0]),
            nn.GELU(),
        )

        # Encoder blocks
        self.blocks = nn.ModuleList()

        # Block 1: 128 -> 64
        self.blocks.append(SpectralEncoderBlock(
            channels[0], channels[0], fno_modes[0], spectral_skip_modes[0], downsample=True
        ))

        # Block 2: 64 -> 32
        self.blocks.append(SpectralEncoderBlock(
            channels[0], channels[1], fno_modes[1], spectral_skip_modes[1], downsample=True
        ))

        # Block 3: 32 -> 16
        self.blocks.append(SpectralEncoderBlock(
            channels[1], channels[2], fno_modes[2], spectral_skip_modes[2], downsample=True
        ))

        # Project to transformer hidden dim
        self.proj = nn.Linear(channels[2], hidden_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: [B, T, H, W, C] input tensor

        Returns:
            latent: [B, T*16*16, hidden_dim] for transformer
            spectral_skips: list of [B*T, C, modes, modes//2+1] FFT coefficients
        """
        B, T, H, W, C = x.shape

        # Reshape for 2D conv: [B*T, C, H, W]
        x = x.permute(0, 1, 4, 2, 3).reshape(B * T, C, H, W)

        # Initial conv
        x = self.init_conv(x)

        # Encoder blocks with spectral skips
        spectral_skips = []
        for block in self.blocks:
            x, skip = block(x)
            spectral_skips.append(skip)

        # x is now [B*T, channels[-1], 16, 16]
        BT, C_out, H_out, W_out = x.shape

        # Reshape for transformer: [B, T*H*W, hidden_dim]
        x = x.reshape(B, T, C_out, H_out, W_out)
        x = x.permute(0, 1, 3, 4, 2)  # [B, T, H, W, C]
        x = x.reshape(B, T * H_out * W_out, C_out)  # [B, T*256, C]

        # Project to hidden dim
        latent = self.proj(x)  # [B, 4096, hidden_dim]

        return latent, spectral_skips


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Spectral Encoder")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = SpectralEncoder(
        in_channels=6,
        channels=[64, 128, 256],
        fno_modes=[32, 16, 8],
        spectral_skip_modes=[32, 16, 8],
        hidden_dim=768,
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"Total parameters: {total_params:,}")

    # Test forward pass
    x = torch.randn(2, 16, 128, 128, 6).to(device)
    latent, skips = encoder(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Number of spectral skips: {len(skips)}")
    for i, skip in enumerate(skips):
        print(f"  Skip {i+1}: {skip.shape}")

    print("\nSpectral Encoder test passed!")
