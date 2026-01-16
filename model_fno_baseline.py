"""
FNO Baseline Model for PDE Prediction.

Based on: "Fourier Neural Operator for Parametric Partial Differential Equations"
          Li et al., 2020 (https://arxiv.org/abs/2010.08895)

This is a simple but effective baseline for PDE surrogate modeling.
Architecture:
    Input [B, T, H, W, C]
        ↓ Lifting (project to hidden dim)
        ↓ Fourier Layer × N
        ↓ Projection (project to output)
    Output [B, T-1, H, W, C]

Key features:
- Pure frequency domain operations
- Global receptive field via FFT
- Designed specifically for PDE tasks
- Much smaller than SUT-FNO (~2-3M vs ~87M params)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class SpectralConv2d(nn.Module):
    """
    2D Spectral Convolution Layer.

    Performs convolution in Fourier space by:
    1. FFT to frequency domain
    2. Multiply with learnable complex weights (for selected modes)
    3. IFFT back to spatial domain
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to keep in first dimension
        self.modes2 = modes2  # Number of Fourier modes to keep in second dimension

        self.scale = 1 / (in_channels * out_channels)

        # Complex weights for spectral convolution
        # Two sets for positive and negative frequencies
        self.weights1 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Complex multiplication in Fourier space."""
        # input: [B, C_in, H, W], weights: [C_in, C_out, modes1, modes2]
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] input tensor
        Returns:
            [B, C_out, H, W] output tensor
        """
        B, C, H, W = x.shape
        dtype = x.dtype

        # FFT requires float32
        x = x.float()

        # Compute 2D real FFT
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                            dtype=torch.cfloat, device=x.device)

        # Positive frequencies (top-left corner)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        # Negative frequencies (bottom-left corner, for modes1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )

        # Inverse FFT
        x = torch.fft.irfft2(out_ft, s=(H, W))

        return x.to(dtype)


class FNOBlock(nn.Module):
    """
    Single FNO Block = Spectral Conv + Bypass Conv + Activation.

    Following the original FNO paper:
    output = activation(SpectralConv(x) + Conv1x1(x))
    """

    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.spectral_conv = SpectralConv2d(width, width, modes1, modes2)
        self.bypass_conv = nn.Conv2d(width, width, 1)
        self.norm = nn.InstanceNorm2d(width)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, width, H, W]
        Returns:
            [B, width, H, W]
        """
        x1 = self.spectral_conv(x)
        x2 = self.bypass_conv(x)
        out = self.norm(x1 + x2)
        return self.activation(out)


class FNOBaseline(nn.Module):
    """
    FNO Baseline Model for PDE time-series prediction.

    For predicting: given timesteps t[0:T-1], predict t[1:T].
    Each timestep is processed independently through FNO.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        model_cfg = config.get('model', {})
        self.in_channels = model_cfg.get('in_channels', 2)  # u, v only
        self.out_channels = model_cfg.get('out_channels', 2)
        self.width = model_cfg.get('width', 64)
        self.modes = model_cfg.get('modes', 32)
        self.n_layers = model_cfg.get('n_layers', 4)

        # Lifting: project input channels to hidden width
        self.lifting = nn.Sequential(
            nn.Conv2d(self.in_channels, self.width, 1),
            nn.GELU(),
            nn.Conv2d(self.width, self.width, 1),
        )

        # FNO Blocks
        self.fno_blocks = nn.ModuleList([
            FNOBlock(self.width, self.modes, self.modes)
            for _ in range(self.n_layers)
        ])

        # Projection: project back to output channels
        self.projection = nn.Sequential(
            nn.Conv2d(self.width, self.width, 1),
            nn.GELU(),
            nn.Conv2d(self.width, self.out_channels, 1),
        )

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, T, H, W, C] input tensor (T timesteps)

        Returns:
            output: [B, T-1, H, W, C] predicted next timesteps
        """
        B, T, H, W, C = x.shape

        # Use T-1 frames as input to predict T-1 frames
        x_input = x[:, :-1]  # [B, T-1, H, W, C]
        T_in = T - 1

        # Reshape for 2D conv: [B*T_in, C, H, W]
        x_input = x_input.permute(0, 1, 4, 2, 3)  # [B, T-1, C, H, W]
        x_input = x_input.reshape(B * T_in, C, H, W)

        # Lifting
        h = self.lifting(x_input)  # [B*T_in, width, H, W]

        # FNO Blocks
        for block in self.fno_blocks:
            h = block(h)

        # Projection
        out = self.projection(h)  # [B*T_in, C_out, H, W]

        # Reshape back: [B, T-1, H, W, C]
        out = out.reshape(B, T_in, self.out_channels, H, W)
        out = out.permute(0, 1, 3, 4, 2)  # [B, T-1, H, W, C]

        return out

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component."""
        lifting_params = sum(p.numel() for p in self.lifting.parameters())
        fno_params = sum(p.numel() for p in self.fno_blocks.parameters())
        projection_params = sum(p.numel() for p in self.projection.parameters())
        total = lifting_params + fno_params + projection_params

        return {
            'lifting': lifting_params,
            'fno_blocks': fno_params,
            'projection': projection_params,
            'total': total,
        }


def create_fno_baseline(config: dict) -> FNOBaseline:
    """Factory function to create FNO baseline model."""
    return FNOBaseline(config)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing FNO Baseline Model")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Test config
    config = {
        'model': {
            'in_channels': 2,   # u, v only
            'out_channels': 2,
            'width': 64,
            'modes': 32,
            'n_layers': 4,
        }
    }

    # Create model
    model = create_fno_baseline(config).to(device)

    # Count parameters
    params = model.count_parameters()
    print(f"\nParameter counts:")
    print(f"  Lifting:    {params['lifting']:>10,}")
    print(f"  FNO Blocks: {params['fno_blocks']:>10,}")
    print(f"  Projection: {params['projection']:>10,}")
    print(f"  Total:      {params['total']:>10,}")

    # Test forward pass
    print(f"\nTesting forward pass...")
    x = torch.randn(2, 17, 128, 128, 2).to(device)  # Only 2 channels (u, v)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        output = model(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")

    # Test backward
    print(f"\nTesting backward pass...")
    loss = output.float().sum()
    loss.backward()
    print("Backward pass successful!")

    # Memory
    if device.type == 'cuda':
        print(f"\nGPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    print("\n" + "=" * 60)
    print("FNO Baseline test passed!")
    print("=" * 60)
