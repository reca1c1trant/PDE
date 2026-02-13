"""
V3 Unified PDE Foundation Model with Shared FFN Transformer.

Supports 1D, 2D, and 3D spatial data with:
    - Separate encoders/decoders per spatial dimension
    - Shared transformer: FFN weights shared across dims, only NA layers are separate

Architecture:
    Input [B, T, *spatial, C]
    -> Encoder (1D/2D/3D selected by input shape)
    -> Shared NA Transformer (FFN shared, NA per dim)
    -> Decoder (1D/2D/3D)
    -> Output [B, T, *spatial, C]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from typing import Dict, Optional
import math

from natten import NeighborhoodAttention2D as NattenNA2D
from natten import NeighborhoodAttention3D as NattenNA3D
from natten import NeighborhoodAttention4D as NattenNA4D

from pretrain.attention_v2 import RMSNorm, SwiGLU
from pretrain.encoder_v2 import (
    PatchifyEncoder, PatchifyEncoder1D, PatchifyEncoder3D,
)
from pretrain.decoder_v2 import (
    PatchifyDecoder, PatchifyDecoder1D, PatchifyDecoder3D,
)


class SharedNATransformerLayer(nn.Module):
    """
    Transformer layer with shared FFN and separate NA per spatial dimension.

    FFN (SwiGLU) + LayerNorm are shared across 1D/2D/3D.
    Only NA attention layers are dimension-specific.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        base_kernel: int,
        dropout: float = 0.0,
        enable_1d: bool = True,
        enable_3d: bool = True,
    ):
        super().__init__()

        # Shared components
        self.norm1 = RMSNorm(hidden_dim)
        self.norm2 = RMSNorm(hidden_dim)
        self.ffn = SwiGLU(hidden_dim, hidden_dim * 4)
        self.dropout = nn.Dropout(dropout)

        # 1D spatial: NA2D (time + 1 spatial)
        if enable_1d:
            self.na_1d = NattenNA2D(
                embed_dim=hidden_dim, num_heads=num_heads,
                kernel_size=(base_kernel, base_kernel),
                dilation=1, is_causal=(True, False),
            )

        # 2D spatial: NA3D with adaptive kernels for different aspect ratios
        self.na_2d_1x1 = NattenNA3D(
            embed_dim=hidden_dim, num_heads=num_heads,
            kernel_size=(base_kernel, base_kernel, base_kernel),
            dilation=1, is_causal=(True, False, False),
        )

        kernel_w_1x2 = (base_kernel - 1) * 2 + 1
        self.na_2d_1x2 = NattenNA3D(
            embed_dim=hidden_dim, num_heads=num_heads,
            kernel_size=(base_kernel, base_kernel, kernel_w_1x2),
            dilation=1, is_causal=(True, False, False),
        )

        kernel_w_1x4 = (base_kernel - 1) * 4 + 1
        self.na_2d_1x4 = NattenNA3D(
            embed_dim=hidden_dim, num_heads=num_heads,
            kernel_size=(base_kernel, base_kernel, kernel_w_1x4),
            dilation=1, is_causal=(True, False, False),
        )

        # 3D spatial: NA4D (time + 3 spatial)
        if enable_3d:
            self.na_3d = NattenNA4D(
                embed_dim=hidden_dim, num_heads=num_heads,
                kernel_size=(base_kernel, base_kernel, base_kernel, base_kernel),
                dilation=1, is_causal=(True, False, False, False),
            )

    def forward(self, x: torch.Tensor, na_key: str = '2d_1x1') -> torch.Tensor:
        """
        Args:
            x: [B, *spatial_dims, D]
            na_key: which NA to use ('1d', '2d_1x1', '2d_1x2', '2d_1x4', '3d')
        """
        na_module = getattr(self, f'na_{na_key}')

        residual = x
        x = self.norm1(x)
        x = na_module(x)
        x = residual + self.dropout(x)

        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ffn(x))

        return x


class SharedNATransformer(nn.Module):
    """
    Shared NA Transformer that routes to the correct NA layer based on input shape.

    FFN weights are shared across all spatial dimensions.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 24,
        num_heads: int = 12,
        base_kernel: int = 7,
        dropout: float = 0.0,
        enable_1d: bool = True,
        enable_3d: bool = True,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.base_kernel = base_kernel
        self.gradient_checkpointing = gradient_checkpointing

        self.layers = nn.ModuleList([
            SharedNATransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                base_kernel=base_kernel,
                dropout=dropout,
                enable_1d=enable_1d,
                enable_3d=enable_3d,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = RMSNorm(hidden_dim)

    def _select_na_key_2d(self, n_h: int, n_w: int) -> str:
        """Select NA variant based on 2D aspect ratio."""
        if n_h == 0 or n_w == 0:
            return '2d_1x1'
        ratio = max(n_h, n_w) / min(n_h, n_w)
        if ratio < 1.5:
            return '2d_1x1'
        elif ratio < 3.0:
            return '2d_1x2'
        else:
            return '2d_1x4'

    def forward(self, x: torch.Tensor, shape_info: Dict) -> torch.Tensor:
        """
        Args:
            x: [B, num_tokens, D] flattened token sequence
            shape_info: dict from encoder with spatial dimensions

        Returns:
            out: [B, num_tokens, D]
        """
        B = x.shape[0]

        if 'n_x' in shape_info and 'n_h' not in shape_info:
            # 1D: reshape to [B, T, n_x, D]
            T, n_x = shape_info['T'], shape_info['n_x']
            x = x.reshape(B, T, n_x, self.hidden_dim)
            na_key = '1d'

            for layer in self.layers:
                if self.gradient_checkpointing and self.training:
                    x = grad_checkpoint(layer, x, na_key, use_reentrant=False)
                else:
                    x = layer(x, na_key)

            x = self.final_norm(x)
            x = x.reshape(B, T * n_x, self.hidden_dim)

        elif 'n_d' in shape_info:
            # 3D: reshape to [B, T, n_d, n_h, n_w, D]
            T = shape_info['T']
            n_d, n_h, n_w = shape_info['n_d'], shape_info['n_h'], shape_info['n_w']
            x = x.reshape(B, T, n_d, n_h, n_w, self.hidden_dim)
            na_key = '3d'

            for layer in self.layers:
                if self.gradient_checkpointing and self.training:
                    x = grad_checkpoint(layer, x, na_key, use_reentrant=False)
                else:
                    x = layer(x, na_key)

            x = self.final_norm(x)
            x = x.reshape(B, T * n_d * n_h * n_w, self.hidden_dim)

        else:
            # 2D: reshape to [B, T, n_h, n_w, D]
            T, n_h, n_w = shape_info['T'], shape_info['n_h'], shape_info['n_w']
            x = x.reshape(B, T, n_h, n_w, self.hidden_dim)
            na_key = self._select_na_key_2d(n_h, n_w)

            for layer in self.layers:
                if self.gradient_checkpointing and self.training:
                    x = grad_checkpoint(layer, x, na_key, use_reentrant=False)
                else:
                    x = layer(x, na_key)

            x = self.final_norm(x)
            x = x.reshape(B, T * n_h * n_w, self.hidden_dim)

        return x


class PDEModelV3(nn.Module):
    """
    V3 Unified PDE Foundation Model.

    Supports 1D [B, T, X, C], 2D [B, T, H, W, C], and 3D [B, T, D, H, W, C].
    Routes to appropriate encoder/decoder based on input shape.
    Transformer FFN is shared across all dimensions.
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        model_cfg = config.get('model', {})
        self.in_channels = model_cfg.get('in_channels', 18)
        self.hidden_dim = model_cfg.get('hidden_dim', 768)

        enable_1d = model_cfg.get('enable_1d', True)
        enable_3d = model_cfg.get('enable_3d', False)

        # 2D encoder/decoder (always enabled)
        self.encoder = self._build_encoder_2d(config)
        self.decoder = self._build_decoder_2d(config)

        # 1D encoder/decoder
        if enable_1d:
            self.encoder_1d = self._build_encoder_1d(config)
            self.decoder_1d = self._build_decoder_1d(config)

        # 3D encoder/decoder
        if enable_3d:
            self.encoder_3d = self._build_encoder_3d(config)
            self.decoder_3d = self._build_decoder_3d(config)

        # Shared transformer
        self.transformer = self._build_transformer(config, enable_1d, enable_3d)

        self.apply(self._init_weights)

    def _build_encoder_2d(self, config: Dict) -> PatchifyEncoder:
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

    def _build_encoder_1d(self, config: Dict) -> PatchifyEncoder1D:
        model_cfg = config.get('model', {})
        encoder_cfg = model_cfg.get('encoder', {})
        intra_cfg = model_cfg.get('intra_patch', {})
        return PatchifyEncoder1D(
            in_channels=model_cfg.get('in_channels', 18),
            hidden_dim=model_cfg.get('hidden_dim', 768),
            patch_size=model_cfg.get('patch_size_1d', model_cfg.get('patch_size', 16)),
            stem_hidden=encoder_cfg.get('stem_hidden', 128),
            stem_out=encoder_cfg.get('stem_out', 256),
            intra_patch_layers=intra_cfg.get('num_layers', 2),
            intra_patch_heads=intra_cfg.get('num_heads', 8),
        )

    def _build_encoder_3d(self, config: Dict) -> PatchifyEncoder3D:
        model_cfg = config.get('model', {})
        encoder_cfg = model_cfg.get('encoder', {})
        intra_cfg = model_cfg.get('intra_patch', {})
        return PatchifyEncoder3D(
            in_channels=model_cfg.get('in_channels', 18),
            hidden_dim=model_cfg.get('hidden_dim', 768),
            patch_size=model_cfg.get('patch_size_3d', 8),
            stem_hidden=encoder_cfg.get('stem_hidden', 128),
            stem_out=encoder_cfg.get('stem_out', 256),
            intra_patch_layers=intra_cfg.get('num_layers', 2),
            intra_patch_heads=intra_cfg.get('num_heads', 8),
        )

    def _build_decoder_2d(self, config: Dict) -> PatchifyDecoder:
        model_cfg = config.get('model', {})
        decoder_cfg = model_cfg.get('decoder', {})
        return PatchifyDecoder(
            out_channels=model_cfg.get('in_channels', 18),
            hidden_dim=model_cfg.get('hidden_dim', 768),
            patch_size=model_cfg.get('patch_size', 16),
            stem_channels=decoder_cfg.get('stem_channels', 256),
            decoder_hidden=decoder_cfg.get('hidden_channels', 128),
        )

    def _build_decoder_1d(self, config: Dict) -> PatchifyDecoder1D:
        model_cfg = config.get('model', {})
        decoder_cfg = model_cfg.get('decoder', {})
        return PatchifyDecoder1D(
            out_channels=model_cfg.get('in_channels', 18),
            hidden_dim=model_cfg.get('hidden_dim', 768),
            patch_size=model_cfg.get('patch_size_1d', model_cfg.get('patch_size', 16)),
            stem_channels=decoder_cfg.get('stem_channels', 256),
            decoder_hidden=decoder_cfg.get('hidden_channels', 128),
        )

    def _build_decoder_3d(self, config: Dict) -> PatchifyDecoder3D:
        model_cfg = config.get('model', {})
        decoder_cfg = model_cfg.get('decoder', {})
        return PatchifyDecoder3D(
            out_channels=model_cfg.get('in_channels', 18),
            hidden_dim=model_cfg.get('hidden_dim', 768),
            patch_size=model_cfg.get('patch_size_3d', 8),
            stem_channels=decoder_cfg.get('stem_channels', 256),
            decoder_hidden=decoder_cfg.get('hidden_channels', 128),
        )

    def _build_transformer(self, config: Dict, enable_1d: bool, enable_3d: bool) -> SharedNATransformer:
        model_cfg = config.get('model', {})
        na_cfg = model_cfg.get('na', {})
        return SharedNATransformer(
            hidden_dim=model_cfg.get('hidden_dim', 768),
            num_layers=model_cfg.get('num_layers', 24),
            num_heads=model_cfg.get('num_heads', 12),
            base_kernel=na_cfg.get('base_kernel', 7),
            dropout=model_cfg.get('dropout', 0.0),
            enable_1d=enable_1d,
            enable_3d=enable_3d,
            gradient_checkpointing=model_cfg.get('gradient_checkpointing', False),
        )

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        channel_mask: Optional[torch.Tensor] = None,
        return_normalized: bool = False,
    ):
        """
        Forward pass with automatic dimension detection.

        Args:
            x: [B, T, X, C] for 1D, [B, T, H, W, C] for 2D, [B, T, D, H, W, C] for 3D
            return_normalized: If True, return (output, mean, std) for normalized loss

        Returns:
            output or (output, mean, std)
        """
        ndim = x.ndim

        # Always normalize: model is trained on normalized data
        spatial_dims = tuple(range(1, ndim - 1))
        mean = x.mean(dim=spatial_dims, keepdim=True)
        std = x.std(dim=spatial_dims, keepdim=True) + 1e-6
        x_norm = (x - mean) / std

        if ndim == 4:
            # 1D: [B, T, X, C]
            if not hasattr(self, 'encoder_1d'):
                raise RuntimeError("Received 1D input but enable_1d=False. Set enable_1d: true in config.")
            tokens, shape_info = self.encoder_1d(x_norm)
            tokens = self.transformer(tokens, shape_info)
            output = self.decoder_1d(tokens, shape_info)
        elif ndim == 5:
            # 2D: [B, T, H, W, C]
            tokens, shape_info = self.encoder(x_norm)
            tokens = self.transformer(tokens, shape_info)
            output = self.decoder(tokens, shape_info)
        elif ndim == 6:
            # 3D: [B, T, D, H, W, C]
            if not hasattr(self, 'encoder_3d'):
                raise RuntimeError("Received 3D input but enable_3d=False. Set enable_3d: true in config.")
            tokens, shape_info = self.encoder_3d(x_norm)
            tokens = self.transformer(tokens, shape_info)
            output = self.decoder_3d(tokens, shape_info)
        else:
            raise ValueError(f"Unsupported input ndim={ndim}, expected 4 (1D), 5 (2D), or 6 (3D)")

        if return_normalized:
            return output, mean, std
        else:
            return output * std + mean

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
