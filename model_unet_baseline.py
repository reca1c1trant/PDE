"""
UNet Baseline Model for PDE Prediction.

Based on unet.py from train_PINN_transient:
- Tanh activation (not ReLU)
- BatchNorm2d normalization
- 4-level encoder-decoder with skip connections

Key features:
- Exact same UNet architecture as train_PINN_transient.py
- Optional positional embedding
- Handles temporal dimension: [B, T, H, W, C] -> [B, T-1, H, W, C]
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Dict


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


class UNet2d(nn.Module):
    """
    UNet2d with Tanh activation and BatchNorm.

    Exact architecture from train_PINN_transient.py:
    - 4 encoder levels with MaxPool
    - Bottleneck
    - 4 decoder levels with ConvTranspose + skip connections
    - Each block: Conv3x3 -> BN -> Tanh -> Conv3x3 -> BN -> Tanh
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1, init_features: int = 32):
        super(UNet2d, self).__init__()

        features = init_features
        self.encoder1 = UNet2d._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet2d._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet2d._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet2d._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet2d._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet2d._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet2d._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet2d._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet2d._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)

    @staticmethod
    def _block(in_channels: int, features: int, name: str) -> nn.Sequential:
        """
        UNet block: Conv3x3 -> BN -> Tanh -> Conv3x3 -> BN -> Tanh

        Note: Uses Tanh activation (not ReLU) - same as train_PINN_transient.py
        """
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "tanh1", nn.Tanh()),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "tanh2", nn.Tanh()),
                ]
            )
        )


class UNetBaseline(nn.Module):
    """
    UNet Baseline Model for PDE time-series prediction.

    Wraps UNet2d to handle temporal dimension.

    Architecture:
        Input [B, T, H, W, C]
        -> Pos Embed (optional) -> [B*T_in, C+2, H, W]
        -> UNet2d
        -> Output [B, T-1, H, W, C]
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        model_cfg = config.get('model', {})
        self.in_channels = model_cfg.get('in_channels', 6)
        self.out_channels = model_cfg.get('out_channels', 6)
        self.init_features = model_cfg.get('init_features', 32)

        # Architecture options
        self.use_positional_embedding = model_cfg.get('use_positional_embedding', True)

        # Grid size for positional embedding
        grid_h = model_cfg.get('grid_h', 128)
        grid_w = model_cfg.get('grid_w', 128)

        # Positional embedding
        if self.use_positional_embedding:
            self.pos_embed = PositionalEmbedding2D((grid_h, grid_w))
            unet_in_channels = self.in_channels + 2  # +2 for (x, y) coords
        else:
            self.pos_embed = None
            unet_in_channels = self.in_channels

        # UNet2d core
        self.unet = UNet2d(
            in_channels=unet_in_channels,
            out_channels=self.out_channels,
            init_features=self.init_features,
        )

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

        # UNet forward
        out = self.unet(x_input)  # [B*T_in, C_out, H, W]

        # Reshape back: [B, T-1, H, W, C]
        out = out.reshape(B, T_in, self.out_channels, H, W)
        out = out.permute(0, 1, 3, 4, 2)

        return out

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component."""
        pos_params = sum(p.numel() for p in self.pos_embed.parameters()) if self.pos_embed else 0
        unet_params = sum(p.numel() for p in self.unet.parameters())
        total = pos_params + unet_params

        return {
            'positional_embedding': pos_params,
            'unet': unet_params,
            'total': total,
        }


def create_unet_baseline(config: dict) -> UNetBaseline:
    """Factory function to create UNet baseline model."""
    return UNetBaseline(config)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing UNet Baseline Model (Tanh + BatchNorm)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Test config
    config = {
        'model': {
            'type': 'unet',
            'in_channels': 6,
            'out_channels': 6,
            'init_features': 32,
            'use_positional_embedding': True,
            'grid_h': 128,
            'grid_w': 128,
        }
    }

    # Create model
    model = create_unet_baseline(config).to(device)

    # Count parameters
    params = model.count_parameters()
    print(f"\nParameter counts:")
    print(f"  Pos Embed: {params['positional_embedding']:>10,}")
    print(f"  UNet:      {params['unet']:>10,}")
    print(f"  Total:     {params['total']:>10,}")

    # Test forward pass
    print(f"\nTesting forward pass...")
    x = torch.randn(2, 17, 128, 128, 6).to(device)

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
    print("UNet Baseline test passed!")
    print("=" * 60)
