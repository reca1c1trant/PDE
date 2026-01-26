"""
Transolver-style Model for PDE Prediction.

Key innovation: Physics-Attention
- Slice: Soft-cluster spatial points into learnable physics slices
- Attend: Self-attention among slice tokens (not individual points)
- Deslice: Broadcast back to original spatial resolution

This reduces complexity from O(N²) to O(N×S) where S << N.

Reference: "Transolver: A Fast Transformer Solver for PDEs on General Geometries" (ICML 2024)
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


def _is_main_process():
    return os.environ.get('LOCAL_RANK', '0') == '0'


class PhysicsAttention(nn.Module):
    """
    Physics-Attention: Slice → Attend → Deslice

    Instead of attention over all N points (O(N²)), we:
    1. Slice: Soft-cluster N points into S slices (O(N×S))
    2. Attend: Self-attention among S slice tokens (O(S²))
    3. Deslice: Broadcast back to N points (O(N×S))

    Total: O(N×S + S²) ≈ O(N×S) when S << N

    Args:
        dim: Feature dimension
        num_heads: Number of attention heads
        num_slices: Number of physics slices (S)
        dropout: Attention dropout
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_slices: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_slices = num_slices
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Slice projection: project to slice assignments
        self.to_slice = nn.Linear(dim, num_slices)
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)  # Learnable temperature

        # Attention projections
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

        # Layer norm for slice tokens
        self.slice_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, C] input features (N = H*W spatial points)

        Returns:
            out: [B, N, C] output features
        """
        B, N, C = x.shape
        H = self.num_heads
        S = self.num_slices

        # ============ Slice ============
        # Compute soft slice assignments
        # [B, N, S]
        slice_logits = self.to_slice(x) / (self.temperature.clamp(min=0.1))
        slice_weights = F.softmax(slice_logits, dim=-1)

        # Aggregate points into slice tokens
        # slice_tokens[b, s, c] = sum_n(slice_weights[b, n, s] * x[b, n, c])
        # [B, S, C]
        slice_tokens = torch.einsum('bns,bnc->bsc', slice_weights, x)

        # Normalize by slice population to prevent scaling issues
        slice_count = slice_weights.sum(dim=1, keepdim=True).transpose(1, 2)  # [B, S, 1]
        slice_tokens = slice_tokens / (slice_count + 1e-6)

        slice_tokens = self.slice_norm(slice_tokens)

        # ============ Attend ============
        # Multi-head self-attention among slice tokens
        q = self.to_q(slice_tokens)  # [B, S, C]
        k = self.to_k(slice_tokens)
        v = self.to_v(slice_tokens)

        # Reshape for multi-head attention
        q = rearrange(q, 'b s (h d) -> b h s d', h=H)
        k = rearrange(k, 'b s (h d) -> b h s d', h=H)
        v = rearrange(v, 'b s (h d) -> b h s d', h=H)

        # Attention scores
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [B, H, S, S]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out_slices = torch.matmul(attn, v)  # [B, H, S, D]
        out_slices = rearrange(out_slices, 'b h s d -> b s (h d)')  # [B, S, C]

        # ============ Deslice ============
        # Broadcast slice tokens back to original points
        # out[b, n, c] = sum_s(slice_weights[b, n, s] * out_slices[b, s, c])
        out = torch.einsum('bns,bsc->bnc', slice_weights, out_slices)

        out = self.to_out(out)

        return out


class PhysicsAttentionBlock(nn.Module):
    """
    Transformer block with Physics-Attention.

    Structure: LayerNorm → PhysicsAttention → Residual → LayerNorm → FFN → Residual
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_slices: int = 64,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = PhysicsAttention(dim, num_heads, num_slices, dropout)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TemporalMixing(nn.Module):
    """
    Temporal mixing layer for causal autoregressive prediction.

    Uses causal 1D convolution along time dimension.
    """

    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(
            dim, dim,
            kernel_size=kernel_size,
            padding=kernel_size - 1,  # Causal padding
            groups=dim  # Depthwise
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, T: int) -> torch.Tensor:
        """
        Args:
            x: [B, T*N, C] where N = H*W
            T: number of timesteps
        """
        B, TN, C = x.shape
        N = TN // T

        # Reshape to [B*N, T, C]
        x = rearrange(x, 'b (t n) c -> (b n) t c', t=T, n=N)

        # Apply causal conv: [B*N, C, T]
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x[..., :T]  # Remove causal padding
        x = x.transpose(1, 2)

        # Reshape back: [B, T*N, C]
        x = rearrange(x, '(b n) t c -> b (t n) c', b=B, n=N)
        x = self.norm(x)

        return x


class TransolverBlock(nn.Module):
    """
    Combined spatial (Physics-Attention) and temporal mixing block.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_slices: int = 64,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Spatial mixing via Physics-Attention
        self.spatial_block = PhysicsAttentionBlock(
            dim, num_heads, num_slices, mlp_ratio, dropout
        )
        # Temporal mixing via causal conv
        self.temporal_mix = TemporalMixing(dim)

    def forward(self, x: torch.Tensor, T: int) -> torch.Tensor:
        """
        Args:
            x: [B, T*N, C]
            T: number of timesteps
        """
        B, TN, C = x.shape
        N = TN // T

        # Spatial attention (per timestep)
        # Reshape to [B*T, N, C]
        x = rearrange(x, 'b (t n) c -> (b t) n c', t=T, n=N)
        x = self.spatial_block(x)
        # Reshape back to [B, T*N, C]
        x = rearrange(x, '(b t) n c -> b (t n) c', b=B, t=T)

        # Temporal mixing
        x = x + self.temporal_mix(x, T)

        return x


class PDETransolver(nn.Module):
    """
    Transolver-style model for PDE time-series prediction.

    Architecture:
        Input [B, T, H, W, C_in]
          ↓
        Patch Embedding (optional) or Point Embedding
          ↓
        N × TransolverBlock (Physics-Attention + Temporal Mixing)
          ↓
        Output Projection
          ↓
        (Optional) Temporal Residual: output = input + delta
          ↓
        Output [B, T, H, W, C_in]

    Args:
        config: Model configuration dict
    """

    def __init__(self, config: dict):
        super().__init__()

        model_cfg = config['model']
        transolver_cfg = model_cfg.get('transolver', {})

        self.in_channels = model_cfg['in_channels']
        self.hidden_dim = transolver_cfg.get('hidden_dim', 256)
        self.num_layers = transolver_cfg.get('num_layers', 8)
        self.num_heads = transolver_cfg.get('num_heads', 8)
        self.num_slices = transolver_cfg.get('num_slices', 64)
        self.mlp_ratio = transolver_cfg.get('mlp_ratio', 4.0)
        self.dropout = transolver_cfg.get('dropout', 0.0)
        self.use_temporal_residual = transolver_cfg.get('use_temporal_residual', True)

        # Patch embedding (convert spatial points to tokens)
        self.patch_size = transolver_cfg.get('patch_size', 4)
        self.patch_embed = nn.Conv2d(
            self.in_channels,
            self.hidden_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        # Positional embedding
        # For 128x128 with patch_size=4: 32x32 = 1024 patches per timestep
        max_patches = (128 // self.patch_size) ** 2  # 1024
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches, self.hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transolver blocks
        self.blocks = nn.ModuleList([
            TransolverBlock(
                dim=self.hidden_dim,
                num_heads=self.num_heads,
                num_slices=self.num_slices,
                mlp_ratio=self.mlp_ratio,
                dropout=self.dropout,
            )
            for _ in range(self.num_layers)
        ])

        # Output projection (upsample back to original resolution)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 2, self.in_channels * self.patch_size * self.patch_size),
        )

        # Temporal residual projection (zero-init)
        if self.use_temporal_residual:
            self.residual_proj = nn.Conv2d(self.in_channels, self.in_channels, 1)
            nn.init.zeros_(self.residual_proj.weight)
            nn.init.zeros_(self.residual_proj.bias)

        self._init_weights()
        self._log_info()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _log_info(self):
        if not _is_main_process():
            return

        total_params = sum(p.numel() for p in self.parameters())

        print(f"\n{'='*60}")
        print(f"PDETransolver (Physics-Attention)")
        print(f"{'='*60}")
        print(f"Hidden Dim: {self.hidden_dim}")
        print(f"Num Layers: {self.num_layers}")
        print(f"Num Heads: {self.num_heads}")
        print(f"Num Slices: {self.num_slices}")
        print(f"Patch Size: {self.patch_size}")
        print(f"MLP Ratio: {self.mlp_ratio}")
        print(f"Temporal Residual: {self.use_temporal_residual}")
        print(f"Total Parameters: {total_params:,}")
        print(f"{'='*60}\n")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, H, W, C] input tensor

        Returns:
            output: [B, T, H, W, C] predicted next timesteps
        """
        B, T, H, W, C = x.shape
        x_input = x  # Save for residual

        # Patch embedding: [B, T, H, W, C] -> [B*T, C, H, W] -> [B*T, D, H', W']
        x = rearrange(x, 'b t h w c -> (b t) c h w')
        x = self.patch_embed(x)  # [B*T, D, H', W']

        _, D, H_p, W_p = x.shape
        N = H_p * W_p  # Number of patches per timestep

        # Flatten spatial: [B*T, D, H', W'] -> [B*T, N, D]
        x = rearrange(x, 'bt d h w -> bt (h w) d')

        # Add positional embedding
        x = x + self.pos_embed[:, :N, :]

        # Reshape for temporal processing: [B, T*N, D]
        x = rearrange(x, '(b t) n d -> b (t n) d', b=B, t=T)

        # Apply Transolver blocks
        for block in self.blocks:
            x = block(x, T)

        # Output projection
        x = self.norm(x)
        x = self.output_proj(x)  # [B, T*N, C*P*P]

        # Reshape to patches: [B, T, H', W', C, P, P]
        x = rearrange(
            x, 'b (t h w) (c p1 p2) -> b t h w c p1 p2',
            t=T, h=H_p, w=W_p, c=C, p1=self.patch_size, p2=self.patch_size
        )

        # Unpatchify: [B, T, H, W, C]
        x = rearrange(x, 'b t h w c p1 p2 -> b t (h p1) (w p2) c')

        # Temporal residual
        if self.use_temporal_residual:
            # Apply zero-init projection
            x_flat = rearrange(x, 'b t h w c -> (b t) c h w')
            delta = self.residual_proj(x_flat)
            delta = rearrange(delta, '(b t) c h w -> b t h w c', b=B, t=T)
            x = x_input + delta

        return x


def compute_nrmse_loss(pred: torch.Tensor, target: torch.Tensor, channel_mask: torch.Tensor):
    """Compute nRMSE loss (same as train_e2e.py)."""
    eps = 1e-8

    if channel_mask.dim() == 2:
        valid_mask = channel_mask[0].bool()
    else:
        valid_mask = channel_mask.bool()

    pred_valid = pred[..., valid_mask]
    target_valid = target[..., valid_mask]

    B, T, H, W, C = pred_valid.shape

    pred_flat = pred_valid.permute(0, 4, 1, 2, 3).reshape(B, C, -1)
    target_flat = target_valid.permute(0, 4, 1, 2, 3).reshape(B, C, -1)

    mse_per_bc = ((pred_flat - target_flat) ** 2).mean(dim=2)
    rmse_per_bc = torch.sqrt(mse_per_bc + eps)

    rms_per_bc = torch.sqrt((target_flat ** 2).mean(dim=2) + eps)

    nrmse_per_bc = rmse_per_bc / rms_per_bc

    nrmse = nrmse_per_bc.mean()
    rmse = rmse_per_bc.mean()

    return nrmse, rmse


if __name__ == "__main__":
    # Test model
    config = {
        'model': {
            'in_channels': 6,
            'transolver': {
                'hidden_dim': 256,
                'num_layers': 8,
                'num_heads': 8,
                'num_slices': 64,
                'patch_size': 4,
                'mlp_ratio': 4.0,
                'dropout': 0.0,
                'use_temporal_residual': True,
            }
        }
    }

    model = PDETransolver(config)

    # Test forward pass
    x = torch.randn(2, 16, 128, 128, 6)
    with torch.no_grad():
        y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
