"""
V2 PDE Foundation Model with Neighborhood Attention.

Architecture:
    Input [B, T, H, W, C]
    -> PatchifyEncoder (CNN + IntraPatchAttn + AttentionPool)
    -> NA Transformer (N layers of NeighborhoodAttention + SwiGLU)
    -> PatchifyDecoder (CNN upsample)
    -> Output [B, T, H, W, C]

Key features:
    - Variable resolution support (128x128, 256x256, 512x512, etc.)
    - Neighborhood Attention for efficiency
    - Intra-patch temporal attention for fine-grained features
    - Adaptive NA kernel for non-square inputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import math
import yaml

from pretrain.attention_v2 import (
    NATransformerLayer,
    NATransformerLayerND,
    RMSNorm,
)
from pretrain.encoder_v2 import (
    PatchifyEncoder, PatchifyEncoder1D, PatchifyEncoder3D, create_encoder_v2,
)
from pretrain.decoder_v2 import (
    PatchifyDecoder, PatchifyDecoder1D, PatchifyDecoder3D, create_decoder_v2,
)


class NATransformer(nn.Module):
    """
    Neighborhood Attention Transformer.

    Stack of NATransformerLayers with adaptive kernel size for non-square inputs.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 24,
        num_heads: int = 12,
        base_kernel: int = 7,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.base_kernel = base_kernel

        # Pre-create layers for common aspect ratios
        # For non-square inputs, we'll select the appropriate layer
        self.layers_1x1 = nn.ModuleList([
            NATransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                kernel_size=(base_kernel, base_kernel, base_kernel),
                is_causal=True,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # For 1:2 ratio (e.g., 128x256)
        # Formula: (base_kernel - 1) * ratio + 1 to keep kernel odd
        # e.g., base=7 -> (7-1)*2+1 = 13
        kernel_w_1x2 = (base_kernel - 1) * 2 + 1
        self.layers_1x2 = nn.ModuleList([
            NATransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                kernel_size=(base_kernel, base_kernel, kernel_w_1x2),
                is_causal=True,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # For 1:4 ratio (e.g., 128x512)
        # e.g., base=7 -> (7-1)*4+1 = 25
        kernel_w_1x4 = (base_kernel - 1) * 4 + 1
        self.layers_1x4 = nn.ModuleList([
            NATransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                kernel_size=(base_kernel, base_kernel, kernel_w_1x4),
                is_causal=True,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = RMSNorm(hidden_dim)

    def _select_layers(self, n_h: int, n_w: int) -> nn.ModuleList:
        """Select appropriate layers based on aspect ratio."""
        if n_h == 0 or n_w == 0:
            return self.layers_1x1

        ratio = max(n_h, n_w) / min(n_h, n_w)

        if ratio < 1.5:
            return self.layers_1x1
        elif ratio < 3.0:
            return self.layers_1x2
        else:
            return self.layers_1x4

    def forward(
        self,
        x: torch.Tensor,
        shape_info: Dict,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T*n_h*n_w, D] token sequence
            shape_info: dict with T, n_h, n_w

        Returns:
            out: [B, T*n_h*n_w, D]
        """
        B = x.shape[0]
        T = shape_info['T']
        n_h = shape_info['n_h']
        n_w = shape_info['n_w']

        # Reshape to 4D for NA: [B, T, n_h, n_w, D]
        x = x.reshape(B, T, n_h, n_w, self.hidden_dim)

        # Select layers based on aspect ratio
        layers = self._select_layers(n_h, n_w)

        # Apply transformer layers
        for layer in layers:
            x = layer(x)

        # Final normalization
        x = self.final_norm(x)

        # Flatten back to sequence
        x = x.reshape(B, T * n_h * n_w, self.hidden_dim)

        return x


class NATransformer1D(nn.Module):
    """
    Neighborhood Attention Transformer for 1D spatial data.

    Uses NA2D with kernel (k_t, k_x). Reshapes [B, T*n_x, D] -> [B, T, n_x, D].
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 24,
        num_heads: int = 12,
        base_kernel: int = 7,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        kernel_size = (base_kernel, base_kernel)  # (k_t, k_x)
        self.layers = nn.ModuleList([
            NATransformerLayerND(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                kernel_size=kernel_size,
                na_dim=2,
                is_causal=True,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = RMSNorm(hidden_dim)

    def forward(self, x: torch.Tensor, shape_info: Dict) -> torch.Tensor:
        """
        Args:
            x: [B, T*n_x, D] token sequence
            shape_info: dict with T, n_x
        Returns:
            out: [B, T*n_x, D]
        """
        B = x.shape[0]
        T = shape_info['T']
        n_x = shape_info['n_x']

        x = x.reshape(B, T, n_x, self.hidden_dim)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)
        x = x.reshape(B, T * n_x, self.hidden_dim)

        return x


class NATransformer3D(nn.Module):
    """
    Neighborhood Attention Transformer for 3D spatial data.

    Uses NA4D with kernel (k_t, k_d, k_h, k_w). Reshapes to [B, T, n_d, n_h, n_w, D].
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 24,
        num_heads: int = 12,
        base_kernel: int = 7,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        kernel_size = (base_kernel, base_kernel, base_kernel, base_kernel)
        self.layers = nn.ModuleList([
            NATransformerLayerND(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                kernel_size=kernel_size,
                na_dim=4,
                is_causal=True,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = RMSNorm(hidden_dim)

    def forward(self, x: torch.Tensor, shape_info: Dict) -> torch.Tensor:
        """
        Args:
            x: [B, T*n_d*n_h*n_w, D] token sequence
            shape_info: dict with T, n_d, n_h, n_w
        Returns:
            out: [B, T*n_d*n_h*n_w, D]
        """
        B = x.shape[0]
        T = shape_info['T']
        n_d = shape_info['n_d']
        n_h = shape_info['n_h']
        n_w = shape_info['n_w']

        x = x.reshape(B, T, n_d, n_h, n_w, self.hidden_dim)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)
        x = x.reshape(B, T * n_d * n_h * n_w, self.hidden_dim)

        return x


class PDEModelV2(nn.Module):
    """
    V2 PDE Foundation Model.

    Complete model with encoder, NA transformer, and decoder.
    Supports variable resolution inputs.
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        model_cfg = config.get('model', {})
        self.in_channels = model_cfg.get('in_channels', 18)
        self.hidden_dim = model_cfg.get('hidden_dim', 768)
        self.patch_size = model_cfg.get('patch_size', 16)

        # Encoder
        self.encoder = self._build_encoder(config)

        # NA Transformer
        self.transformer = self._build_transformer(config)

        # Decoder
        self.decoder = self._build_decoder(config)

        # Initialize weights
        self.apply(self._init_weights)

    def _build_encoder(self, config: Dict) -> PatchifyEncoder:
        """Build encoder from config."""
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

    def _build_transformer(self, config: Dict) -> NATransformer:
        """Build NA transformer from config."""
        model_cfg = config.get('model', {})
        na_cfg = model_cfg.get('na', {})

        return NATransformer(
            hidden_dim=model_cfg.get('hidden_dim', 768),
            num_layers=model_cfg.get('num_layers', 24),
            num_heads=model_cfg.get('num_heads', 12),
            base_kernel=na_cfg.get('base_kernel', 7),
            dropout=model_cfg.get('dropout', 0.0),
        )

    def _build_decoder(self, config: Dict) -> PatchifyDecoder:
        """Build decoder from config."""
        model_cfg = config.get('model', {})
        decoder_cfg = model_cfg.get('decoder', {})

        return PatchifyDecoder(
            out_channels=model_cfg.get('in_channels', 18),
            hidden_dim=model_cfg.get('hidden_dim', 768),
            patch_size=model_cfg.get('patch_size', 16),
            stem_channels=decoder_cfg.get('stem_channels', 256),
            decoder_hidden=decoder_cfg.get('hidden_channels', 128),
        )

    def _init_weights(self, module: nn.Module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        channel_mask: Optional[torch.Tensor] = None,
        return_normalized: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, T, H, W, C] input tensor
            channel_mask: [B, C] valid channel mask (optional)
            return_normalized: If True, return (output, mean, std) for normalized loss

        Returns:
            output: [B, T, H, W, C] or (output, mean, std) if return_normalized
        """
        B, T, H, W, C = x.shape

        # Always normalize: model weights are trained on normalized data
        mean = x.mean(dim=(1, 2, 3), keepdim=True)  # [B, 1, 1, 1, C]
        std = x.std(dim=(1, 2, 3), keepdim=True) + 1e-6  # [B, 1, 1, 1, C]
        x_norm = (x - mean) / std

        # Encode
        tokens, shape_info = self.encoder(x_norm)  # [B, T*n_h*n_w, D]

        # Transform
        tokens = self.transformer(tokens, shape_info)  # [B, T*n_h*n_w, D]

        # Decode
        output = self.decoder(tokens, shape_info)  # [B, T, H, W, C]

        if return_normalized:
            return output, mean, std
        else:
            return output * std + mean

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return the number of parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and hasattr(self.encoder, 'pos_enc'):
            n_params -= sum(p.numel() for p in self.encoder.pos_enc.parameters())
        return n_params


def create_model_v2(config: Dict) -> PDEModelV2:
    """Create model from config dict."""
    return PDEModelV2(config)


def load_config(config_path: str) -> Dict:
    """Load config from yaml file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    """Test model."""
    print("=" * 60)
    print("Testing PDEModelV2")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create config
    config = {
        'model': {
            'in_channels': 18,
            'hidden_dim': 768,
            'patch_size': 16,
            'num_layers': 24,
            'num_heads': 12,
            'dropout': 0.0,
            'encoder': {
                'stem_hidden': 128,
                'stem_out': 256,
                'use_cnn_pool': False,
            },
            'intra_patch': {
                'num_layers': 2,
                'temporal_window': 3,
                'num_heads': 8,
            },
            'na': {
                'base_kernel': 7,
            },
            'decoder': {
                'stem_channels': 256,
                'hidden_channels': 128,
            },
        }
    }

    # Create smaller model for testing
    test_config = {
        'model': {
            'in_channels': 18,
            'hidden_dim': 256,
            'patch_size': 16,
            'num_layers': 4,
            'num_heads': 4,
            'dropout': 0.0,
            'encoder': {
                'stem_hidden': 64,
                'stem_out': 128,
            },
            'intra_patch': {
                'num_layers': 1,
                'temporal_window': 3,
                'num_heads': 4,
            },
            'na': {
                'base_kernel': 5,
            },
            'decoder': {
                'stem_channels': 128,
                'hidden_channels': 64,
            },
        }
    }

    model = PDEModelV2(test_config).to(device)
    print(f"\nModel parameters: {model.get_num_params():,}")

    # Test different input sizes
    test_cases = [
        (2, 8, 128, 128, 18),
        (1, 8, 256, 256, 18),
        (1, 8, 128, 512, 18),
    ]

    print()
    for B, T, H, W, C in test_cases:
        x = torch.randn(B, T, H, W, C, device=device)

        # Test forward
        output = model(x)
        print(f"Input: [{B}, {T}, {H}, {W}, {C}]")
        print(f"  -> Output: {output.shape}")
        assert output.shape == x.shape, f"Shape mismatch!"

        # Test with return_normalized
        output_norm, mean, std = model(x, return_normalized=True)
        print(f"  -> Normalized output: {output_norm.shape}")
        print(f"  -> Mean: {mean.shape}, Std: {std.shape}")

        print(f"  ✓ Passed")
        print()

    # Test gradient flow
    print("Testing gradient flow...")
    x = torch.randn(1, 8, 128, 128, 18, device=device, requires_grad=True)
    output = model(x)
    loss = output.sum()
    loss.backward()
    print(f"  Input grad: {x.grad is not None}")
    print(f"  ✓ Gradient flow OK")

    print("\n" + "=" * 60)
    print("All model tests passed!")
    print("=" * 60)

    # Print full model config
    print("\nFull model (Base) config:")
    print(f"  Hidden dim: 768")
    print(f"  Layers: 24")
    print(f"  Heads: 12")
    print(f"  Intra-patch layers: 2")
    print(f"  Temporal window: 3")
    print(f"  NA kernel: (7, 7, 7)")

    # Estimate full model params
    full_model = PDEModelV2(config).to(device)
    print(f"\n  Total parameters: {full_model.get_num_params():,}")
    print("=" * 60)
