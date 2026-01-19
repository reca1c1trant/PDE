"""
FNO Baseline Model for PDE Prediction.

Aligned with PINO official implementation:
1. Residual skip connection in FNO blocks
2. Positional Embedding (coordinate grid)
3. ChannelMLP inside FNO blocks

Based on:
- FNO: https://arxiv.org/abs/2010.08895
- PINO: https://arxiv.org/abs/2111.03794
- Official neuraloperator repo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import math


class PositionalEmbedding2D(nn.Module):
    """
    2D Positional Embedding - adds coordinate grid to input.

    Concatenates normalized (x, y) coordinates to input channels.
    """

    def __init__(self, grid_size: tuple[int, int]):
        super().__init__()
        H, W = grid_size

        # Create normalized coordinate grids [0, 1]
        x = torch.linspace(0, 1, W)
        y = torch.linspace(0, 1, H)

        # Create meshgrid: [H, W]
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')

        # Stack to [2, H, W]
        grid = torch.stack([grid_x, grid_y], dim=0)

        # Register as buffer (not a parameter, but moves with model)
        self.register_buffer('grid', grid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, C+2, H, W] with (x, y) coordinates appended
        """
        B = x.shape[0]
        # Expand grid to batch: [2, H, W] -> [B, 2, H, W]
        grid = self.grid.unsqueeze(0).expand(B, -1, -1, -1)
        return torch.cat([x, grid], dim=1)


class ChannelMLP(nn.Module):
    """
    Channel-wise MLP for feature mixing.

    Applied point-wise across spatial dimensions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: Optional[int] = None,
        n_layers: int = 2,
        activation: str = 'gelu',
    ):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = in_channels

        act_fn = nn.GELU() if activation == 'gelu' else nn.ReLU()

        layers = []
        for i in range(n_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            out_ch = out_channels if i == n_layers - 1 else hidden_channels
            layers.append(nn.Conv2d(in_ch, out_ch, 1))
            if i < n_layers - 1:
                layers.append(act_fn)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


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
        B, C, H, W = x.shape
        dtype = x.dtype

        x = x.float()
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                            dtype=torch.cfloat, device=x.device)

        # Positive frequencies
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        # Negative frequencies
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )

        x = torch.fft.irfft2(out_ft, s=(H, W))
        return x.to(dtype)


class FNOBlock(nn.Module):
    """
    FNO Block with:
    1. Residual skip connection (x + f(x))
    2. Optional ChannelMLP

    Structure:
        x_out = x + activation(norm(spectral(x) + bypass(x) + mlp(x)))
    """

    def __init__(
        self,
        width: int,
        modes1: int,
        modes2: int,
        use_channel_mlp: bool = True,
        mlp_expansion: float = 0.5,
    ):
        super().__init__()
        self.use_channel_mlp = use_channel_mlp

        # Spectral convolution (global, frequency domain)
        self.spectral_conv = SpectralConv2d(width, width, modes1, modes2)

        # Bypass convolution (local, spatial domain)
        self.bypass_conv = nn.Conv2d(width, width, 1)

        # Optional ChannelMLP
        if use_channel_mlp:
            mlp_hidden = int(width * mlp_expansion)
            self.channel_mlp = ChannelMLP(
                in_channels=width,
                out_channels=width,
                hidden_channels=mlp_hidden,
                n_layers=2,
            )

        self.norm = nn.InstanceNorm2d(width)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, width, H, W]
        Returns:
            [B, width, H, W]
        """
        # Spectral path (global)
        x1 = self.spectral_conv(x)

        # Bypass path (local)
        x2 = self.bypass_conv(x)

        # Combine
        out = x1 + x2

        # Optional MLP
        if self.use_channel_mlp:
            out = out + self.channel_mlp(out)

        # Norm + Activation
        out = self.norm(out)
        out = self.activation(out)

        # Residual skip connection
        return x + out


class FNOBaseline(nn.Module):
    """
    FNO Baseline Model for PDE time-series prediction.

    Features:
    1. Residual skip in FNO blocks
    2. Positional embedding (coordinate grid)
    3. ChannelMLP in FNO blocks
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        model_cfg = config.get('model', {})
        self.in_channels = model_cfg.get('in_channels', 2)
        self.out_channels = model_cfg.get('out_channels', 2)
        self.width = model_cfg.get('width', 64)
        self.modes = model_cfg.get('modes', 32)
        self.n_layers = model_cfg.get('n_layers', 4)

        # Architecture options
        self.use_positional_embedding = model_cfg.get('use_positional_embedding', True)
        self.use_channel_mlp = model_cfg.get('use_channel_mlp', True)
        self.mlp_expansion = model_cfg.get('mlp_expansion', 0.5)

        # Grid size for positional embedding
        grid_h = model_cfg.get('grid_h', 128)
        grid_w = model_cfg.get('grid_w', 128)

        # Positional embedding
        if self.use_positional_embedding:
            self.pos_embed = PositionalEmbedding2D((grid_h, grid_w))
            lifting_in_channels = self.in_channels + 2  # +2 for (x, y) coords
        else:
            self.pos_embed = None
            lifting_in_channels = self.in_channels

        # Lifting: project to hidden width
        self.lifting = ChannelMLP(
            in_channels=lifting_in_channels,
            out_channels=self.width,
            hidden_channels=self.width,
            n_layers=2,
        )

        # FNO Blocks with residual skip and optional MLP
        self.fno_blocks = nn.ModuleList([
            FNOBlock(
                width=self.width,
                modes1=self.modes,
                modes2=self.modes,
                use_channel_mlp=self.use_channel_mlp,
                mlp_expansion=self.mlp_expansion,
            )
            for _ in range(self.n_layers)
        ])

        # Projection: project back to output channels
        self.projection = ChannelMLP(
            in_channels=self.width,
            out_channels=self.out_channels,
            hidden_channels=self.width,
            n_layers=2,
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, T, H, W, C] input tensor

        Returns:
            output: [B, T-1, H, W, C] predicted next timesteps
        """
        B, T, H, W, C = x.shape

        # Use T-1 frames as input
        x_input = x[:, :-1]  # [B, T-1, H, W, C]
        T_in = T - 1

        # Reshape: [B*T_in, C, H, W]
        x_input = x_input.permute(0, 1, 4, 2, 3)  # [B, T-1, C, H, W]
        x_input = x_input.reshape(B * T_in, C, H, W)

        # Positional embedding
        if self.pos_embed is not None:
            x_input = self.pos_embed(x_input)  # [B*T_in, C+2, H, W]

        # Lifting
        h = self.lifting(x_input)  # [B*T_in, width, H, W]

        # FNO Blocks (with residual)
        for block in self.fno_blocks:
            h = block(h)

        # Projection
        out = self.projection(h)  # [B*T_in, C_out, H, W]

        # Reshape back: [B, T-1, H, W, C]
        out = out.reshape(B, T_in, self.out_channels, H, W)
        out = out.permute(0, 1, 3, 4, 2)

        return out

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component."""
        pos_params = sum(p.numel() for p in self.pos_embed.parameters()) if self.pos_embed else 0
        lifting_params = sum(p.numel() for p in self.lifting.parameters())
        fno_params = sum(p.numel() for p in self.fno_blocks.parameters())
        projection_params = sum(p.numel() for p in self.projection.parameters())
        total = pos_params + lifting_params + fno_params + projection_params

        return {
            'positional_embedding': pos_params,
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
            'in_channels': 2,
            'out_channels': 2,
            'width': 64,
            'modes': 32,
            'n_layers': 4,
            'use_positional_embedding': True,
            'use_channel_mlp': True,
            'mlp_expansion': 0.5,
            'grid_h': 128,
            'grid_w': 128,
        }
    }

    # Create model
    model = create_fno_baseline(config).to(device)

    # Count parameters
    params = model.count_parameters()
    print(f"\nParameter counts:")
    print(f"  Pos Embed:  {params['positional_embedding']:>10,}")
    print(f"  Lifting:    {params['lifting']:>10,}")
    print(f"  FNO Blocks: {params['fno_blocks']:>10,}")
    print(f"  Projection: {params['projection']:>10,}")
    print(f"  Total:      {params['total']:>10,}")

    # Test forward pass
    print(f"\nTesting forward pass...")
    x = torch.randn(2, 17, 128, 128, 2).to(device)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        output = model(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")

    # Verify residual connections work
    print(f"\nTesting backward pass...")
    loss = output.float().sum()
    loss.backward()
    print("Backward pass successful!")

    if device.type == 'cuda':
        print(f"\nGPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    print("\n" + "=" * 60)
    print("FNO Baseline test passed!")
    print("=" * 60)
