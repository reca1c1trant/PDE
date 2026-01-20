"""
MLP Baseline Model for PDE Prediction.

SIREN-style architecture inspired by n-PINN:
1. Positional Embedding (coordinate grid)
2. SIREN activation (sin) for smooth function approximation
3. Residual connections

Key features:
- Uses sin activation like SIREN/n-PINN
- Conv2d (1x1) acts as point-wise MLP
- Much simpler than FNO (no spectral convolution)
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


class SIRENLayer(nn.Module):
    """
    SIREN Layer: Linear + sin activation.

    Uses special initialization for SIREN networks.
    First layer uses omega_0 = 30 (default), hidden layers use omega_0 = 1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        is_first: bool = False,
        omega_0: float = 30.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_first = is_first
        self.omega_0 = omega_0

        # Conv2d with kernel_size=1 acts as point-wise linear
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

        self._init_weights()

    def _init_weights(self):
        """SIREN initialization."""
        with torch.no_grad():
            if self.is_first:
                # First layer: uniform in [-1/in_features, 1/in_features]
                bound = 1.0 / self.in_channels
            else:
                # Hidden layers: uniform in [-sqrt(6/in_features)/omega_0, sqrt(6/in_features)/omega_0]
                bound = math.sqrt(6.0 / self.in_channels) / self.omega_0

            self.conv.weight.uniform_(-bound, bound)
            if self.conv.bias is not None:
                self.conv.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, H, W]
        Returns:
            [B, C_out, H, W]
        """
        return torch.sin(self.omega_0 * self.conv(x))


class SIRENBlock(nn.Module):
    """
    SIREN Block with optional residual connection.

    Structure:
        out = x + SIREN(x)  (if residual and in_ch == out_ch)
        out = SIREN(x)      (otherwise)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        is_first: bool = False,
        omega_0: float = 30.0,
        use_residual: bool = True,
    ):
        super().__init__()
        self.use_residual = use_residual and (in_channels == out_channels)

        self.siren = SIRENLayer(in_channels, out_channels, is_first=is_first, omega_0=omega_0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.siren(x)
        if self.use_residual:
            out = out + x
        return out


class MLPBaseline(nn.Module):
    """
    MLP Baseline Model for PDE time-series prediction.

    Features:
    1. SIREN-style activation (sin)
    2. Positional embedding (coordinate grid)
    3. Point-wise MLP via Conv2d(1x1)

    Architecture:
        Input [B, T, H, W, C]
        -> Pos Embed -> [B*T, C+2, H, W]
        -> Lifting (SIREN)
        -> Hidden SIREN Blocks x N
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
        self.omega_0 = model_cfg.get('omega_0', 30.0)

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

        # Lifting: first SIREN layer
        self.lifting = SIRENLayer(
            in_channels=lifting_in_channels,
            out_channels=self.width,
            is_first=True,
            omega_0=self.omega_0,
        )

        # Hidden SIREN Blocks
        self.hidden_blocks = nn.ModuleList([
            SIRENBlock(
                in_channels=self.width,
                out_channels=self.width,
                is_first=False,
                omega_0=1.0,  # Use omega_0=1 for hidden layers
                use_residual=self.use_residual,
            )
            for _ in range(self.n_layers)
        ])

        # Projection: linear output (no activation)
        self.projection = nn.Conv2d(self.width, self.out_channels, 1)
        self._init_projection()

    def _init_projection(self):
        """Initialize projection layer with small weights."""
        with torch.no_grad():
            bound = math.sqrt(6.0 / self.width)
            self.projection.weight.uniform_(-bound, bound)
            if self.projection.bias is not None:
                self.projection.bias.zero_()

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

        # Lifting (first SIREN layer)
        h = self.lifting(x_input)  # [B*T_in, width, H, W]

        # Hidden SIREN Blocks
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
    print("Testing MLP Baseline Model (SIREN-style)")
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
            'omega_0': 30.0,
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
