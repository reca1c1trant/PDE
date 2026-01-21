"""
MLP Baseline Model for PDE Prediction.

Architecture with Tanh + BatchNorm (same as UNet baseline):
1. Positional Embedding (coordinate grid)
2. Tanh activation with BatchNorm
3. Residual connections

Key features:
- Uses Tanh activation (same as UNet/train_PINN_transient.py)
- Conv2d (1x1) acts as point-wise MLP
- BatchNorm2d for normalization
"""

import torch
import torch.nn as nn
from typing import Dict
from collections import OrderedDict


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


class MLPBlock(nn.Module):
    """
    MLP Block with Tanh activation and BatchNorm.

    Structure: Conv1x1 -> BN -> Tanh -> Conv1x1 -> BN -> Tanh (+ residual)

    Same activation pattern as UNet block in train_PINN_transient.py.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_residual: bool = True,
    ):
        super().__init__()
        self.use_residual = use_residual and (in_channels == out_channels)

        self.block = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)),
                ('norm1', nn.BatchNorm2d(out_channels)),
                ('tanh1', nn.Tanh()),
                ('conv2', nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)),
                ('norm2', nn.BatchNorm2d(out_channels)),
                ('tanh2', nn.Tanh()),
            ])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


class MLPBaseline(nn.Module):
    """
    MLP Baseline Model for PDE time-series prediction.

    Features:
    1. Tanh activation with BatchNorm (same as UNet)
    2. Positional embedding (coordinate grid)
    3. Point-wise MLP via Conv2d(1x1)

    Architecture:
        Input [B, T, H, W, C]
        -> Pos Embed -> [B*T, C+2, H, W]
        -> Lifting (Conv1x1 + BN + Tanh)
        -> Hidden MLP Blocks x N
        -> Projection
        -> Output [B, T-1, H, W, C]
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        model_cfg = config.get('model', {})
        self.in_channels = model_cfg.get('in_channels', 2)
        self.out_channels = model_cfg.get('out_channels', 2)
        self.width = model_cfg.get('width', 64)
        self.n_layers = model_cfg.get('n_layers', 4)

        # Architecture options
        self.use_positional_embedding = model_cfg.get('use_positional_embedding', True)
        self.use_residual = model_cfg.get('use_residual', True)

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

        # Lifting: first layer (Conv1x1 + BN + Tanh)
        self.lifting = nn.Sequential(
            nn.Conv2d(lifting_in_channels, self.width, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.width),
            nn.Tanh(),
        )

        # Hidden MLP Blocks
        self.hidden_blocks = nn.ModuleList([
            MLPBlock(
                in_channels=self.width,
                out_channels=self.width,
                use_residual=self.use_residual,
            )
            for _ in range(self.n_layers)
        ])

        # Projection: linear output (no activation)
        self.projection = nn.Conv2d(self.width, self.out_channels, kernel_size=1)

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

        # Lifting (first layer)
        h = self.lifting(x_input)  # [B*T_in, width, H, W]

        # Hidden MLP Blocks
        for block in self.hidden_blocks:
            h = block(h)

        # Projection (linear, no activation)
        out = self.projection(h)  # [B*T_in, C_out, H, W]

        # Reshape back: [B, T-1, H, W, C]
        out = out.reshape(B, T_in, self.out_channels, H, W)
        out = out.permute(0, 1, 3, 4, 2)

        return out

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component."""
        pos_params = sum(p.numel() for p in self.pos_embed.parameters()) if self.pos_embed else 0
        lifting_params = sum(p.numel() for p in self.lifting.parameters())
        hidden_params = sum(p.numel() for p in self.hidden_blocks.parameters())
        projection_params = sum(p.numel() for p in self.projection.parameters())
        total = pos_params + lifting_params + hidden_params + projection_params

        return {
            'positional_embedding': pos_params,
            'lifting': lifting_params,
            'hidden_blocks': hidden_params,
            'projection': projection_params,
            'total': total,
        }


def create_mlp_baseline(config: dict) -> MLPBaseline:
    """Factory function to create MLP baseline model."""
    return MLPBaseline(config)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing MLP Baseline Model (Tanh + BatchNorm)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Test config
    config = {
        'model': {
            'in_channels': 2,
            'out_channels': 2,
            'width': 64,
            'n_layers': 4,
            'use_positional_embedding': True,
            'use_residual': True,
            'grid_h': 128,
            'grid_w': 128,
        }
    }

    # Create model
    model = create_mlp_baseline(config).to(device)

    # Count parameters
    params = model.count_parameters()
    print(f"\nParameter counts:")
    print(f"  Pos Embed:     {params['positional_embedding']:>10,}")
    print(f"  Lifting:       {params['lifting']:>10,}")
    print(f"  Hidden Blocks: {params['hidden_blocks']:>10,}")
    print(f"  Projection:    {params['projection']:>10,}")
    print(f"  Total:         {params['total']:>10,}")

    # Test forward pass
    print(f"\nTesting forward pass...")
    x = torch.randn(2, 17, 128, 128, 2).to(device)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        output = model(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")

    # Verify backward pass
    print(f"\nTesting backward pass...")
    loss = output.float().sum()
    loss.backward()
    print("Backward pass successful!")

    if device.type == 'cuda':
        print(f"\nGPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    print("\n" + "=" * 60)
    print("MLP Baseline test passed!")
    print("=" * 60)
