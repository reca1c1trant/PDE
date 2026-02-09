"""
V2 Attention Modules for PDE Foundation Model.

Contains:
1. IntraPatchTemporalAttention: Causal attention within patches across time
2. AttentionPool: Learnable aggregation from 4x4 to 1x1
3. NeighborhoodAttention3D: Wrapper for natten (with fallback)
4. RMSNorm, SwiGLU: From Llama architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from natten import NeighborhoodAttention3D as NattenNA3D


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (from Llama)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class SwiGLU(nn.Module):
    """SwiGLU activation (from Llama)."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class AttentionPool(nn.Module):
    """
    Learnable attention pooling: 4x4 -> 1x1.

    Uses a learnable query to compute attention weights over spatial positions.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Learnable query vector
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        # Project keys
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = math.sqrt(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, n_h, n_w, 4, 4, D]
        Returns:
            out: [B, T, n_h, n_w, D]
        """
        B, T, n_h, n_w, h, w, D = x.shape

        # Reshape to [B*T*n_h*n_w, 16, D]
        x = x.reshape(B * T * n_h * n_w, h * w, D)

        # Compute attention weights
        keys = self.key_proj(x)  # [N, 16, D]
        query = self.query.expand(x.shape[0], -1, -1)  # [N, 1, D]

        attn_weights = torch.bmm(query, keys.transpose(1, 2)) / self.scale  # [N, 1, 16]
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Weighted sum
        out = torch.bmm(attn_weights, x)  # [N, 1, D]
        out = out.squeeze(1)  # [N, D]

        # Reshape back
        out = out.reshape(B, T, n_h, n_w, D)
        return out


class IntraPatchTemporalAttention(nn.Module):
    """
    Causal self-attention within patches across time steps.

    For each spatial position (h, w), collects tokens from multiple time steps
    and applies causal self-attention.

    Args:
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        num_layers: Number of attention layers (default 2)
        temporal_window: How many time steps to look back (default 3)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        temporal_window: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.temporal_window = temporal_window
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Multi-layer attention
        self.layers = nn.ModuleList([
            IntraPatchAttentionLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, n_h, n_w, 4, 4, D]
        Returns:
            out: [B, T, n_h, n_w, 4, 4, D]
        """
        B, T, n_h, n_w, h, w, D = x.shape
        sub_patches = h * w  # 16

        # Reshape: group by spatial position (h_idx, w_idx)
        # [B, T, n_h, n_w, 16, D] -> [B, n_h, n_w, T, 16, D]
        x = x.reshape(B, T, n_h, n_w, sub_patches, D)
        x = x.permute(0, 2, 3, 1, 4, 5)  # [B, n_h, n_w, T, 16, D]

        # Merge batch dims: [B*n_h*n_w, T, 16, D]
        x = x.reshape(B * n_h * n_w, T, sub_patches, D)

        # Flatten time and sub-patches: [B*n_h*n_w, T*16, D]
        x = x.reshape(B * n_h * n_w, T * sub_patches, D)

        # Create causal mask with temporal window
        # Each token at time t can only see tokens at time max(0, t-window+1) to t
        causal_mask = self._create_temporal_causal_mask(T, sub_patches, x.device)

        # Apply attention layers
        for layer in self.layers:
            x = layer(x, causal_mask)

        # Reshape back: [B*n_h*n_w, T*16, D] -> [B, T, n_h, n_w, 4, 4, D]
        x = x.reshape(B * n_h * n_w, T, sub_patches, D)
        x = x.reshape(B, n_h, n_w, T, h, w, D)
        x = x.permute(0, 3, 1, 2, 4, 5, 6)  # [B, T, n_h, n_w, 4, 4, D]

        return x

    def _create_temporal_causal_mask(
        self, T: int, sub_patches: int, device: torch.device
    ) -> torch.Tensor:
        """
        Create causal mask with temporal window.

        For token at (t, s), it can see tokens at (t', s') where:
        - t' <= t (causal)
        - t' >= t - temporal_window + 1 (window constraint)

        Returns:
            mask: [T*sub_patches, T*sub_patches] where True means MASKED (cannot attend)
        """
        seq_len = T * sub_patches

        # Create position indices
        positions = torch.arange(seq_len, device=device)
        time_indices = positions // sub_patches  # Which time step each token belongs to

        # For each pair (i, j), check if j can attend to i
        # j can attend to i if: time[i] <= time[j] AND time[i] >= time[j] - window + 1
        time_i = time_indices.unsqueeze(0)  # [1, seq_len]
        time_j = time_indices.unsqueeze(1)  # [seq_len, 1]

        # Causal: can only attend to past or current time
        causal_ok = time_i <= time_j

        # Window: can only attend within window
        window_ok = time_i >= (time_j - self.temporal_window + 1)

        # Combined: both conditions must be true
        can_attend = causal_ok & window_ok

        # Mask is True where we CANNOT attend
        mask = ~can_attend

        return mask


class IntraPatchAttentionLayer(nn.Module):
    """Single attention layer for intra-patch attention."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.norm1 = RMSNorm(hidden_dim)
        self.norm2 = RMSNorm(hidden_dim)
        self.ffn = SwiGLU(hidden_dim, hidden_dim * 4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, D]
            mask: [seq_len, seq_len] where True means MASKED
        Returns:
            out: [B, seq_len, D]
        """
        B, seq_len, D = x.shape

        # Self-attention with pre-norm
        residual = x
        x = self.norm1(x)

        q = self.q_proj(x).reshape(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [B, heads, seq, seq]

        # Apply mask
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)  # [B, heads, seq, head_dim]
        out = out.transpose(1, 2).reshape(B, seq_len, D)
        out = self.o_proj(out)

        x = residual + self.dropout(out)

        # FFN with pre-norm
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ffn(x))

        return x


class NeighborhoodAttention3D(nn.Module):
    """
    3D Neighborhood Attention wrapper using natten.

    Args:
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        kernel_size: (k_t, k_h, k_w) neighborhood size
        is_causal: Whether to apply causal mask on time dimension
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        kernel_size: Tuple[int, int, int] = (7, 7, 7),
        is_causal: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.is_causal = is_causal

        self.na = NattenNA3D(
            dim=hidden_dim,
            num_heads=num_heads,
            kernel_size=kernel_size,
            dilation=1,
            is_causal=(is_causal, False, False),  # Causal only on time dim
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, n_h, n_w, D]
        Returns:
            out: [B, T, n_h, n_w, D]
        """
        return self.na(x)


class NATransformerLayer(nn.Module):
    """
    Single transformer layer with Neighborhood Attention.

    Structure: NA -> RMSNorm -> SwiGLU -> RMSNorm (Pre-norm style)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        kernel_size: Tuple[int, int, int] = (7, 7, 7),
        is_causal: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_dim)
        self.na = NeighborhoodAttention3D(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            kernel_size=kernel_size,
            is_causal=is_causal,
            dropout=dropout,
        )
        self.norm2 = RMSNorm(hidden_dim)
        self.ffn = SwiGLU(hidden_dim, hidden_dim * 4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, n_h, n_w, D]
        Returns:
            out: [B, T, n_h, n_w, D]
        """
        # NA with pre-norm
        residual = x
        x = self.norm1(x)
        x = self.na(x)
        x = residual + self.dropout(x)

        # FFN with pre-norm
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ffn(x))

        return x


if __name__ == "__main__":
    """Test attention modules."""
    print("=" * 60)
    print("Testing Attention V2 Modules")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, T, n_h, n_w, h, w, D = 2, 8, 8, 8, 4, 4, 256

    # Test IntraPatchTemporalAttention
    print("\n1. IntraPatchTemporalAttention")
    x = torch.randn(B, T, n_h, n_w, h, w, D, device=device)
    intra_attn = IntraPatchTemporalAttention(
        hidden_dim=D, num_heads=8, num_layers=2, temporal_window=3
    ).to(device)
    out = intra_attn(x)
    print(f"   Input:  {x.shape}")
    print(f"   Output: {out.shape}")
    assert out.shape == x.shape, "Shape mismatch!"
    print("   ✓ Passed")

    # Test AttentionPool
    print("\n2. AttentionPool")
    attn_pool = AttentionPool(hidden_dim=D).to(device)
    out_pool = attn_pool(x)
    print(f"   Input:  {x.shape}")
    print(f"   Output: {out_pool.shape}")
    assert out_pool.shape == (B, T, n_h, n_w, D), "Shape mismatch!"
    print("   ✓ Passed")

    # Test NeighborhoodAttention3D
    print("\n3. NeighborhoodAttention3D")
    x_na = torch.randn(B, T, n_h, n_w, D, device=device)
    na = NeighborhoodAttention3D(
        hidden_dim=D, num_heads=8, kernel_size=(7, 7, 7), is_causal=True
    ).to(device)
    out_na = na(x_na)
    print(f"   Input:  {x_na.shape}")
    print(f"   Output: {out_na.shape}")
    assert out_na.shape == x_na.shape, "Shape mismatch!"
    print("   ✓ Passed")

    # Test NATransformerLayer
    print("\n4. NATransformerLayer")
    na_layer = NATransformerLayer(
        hidden_dim=D, num_heads=8, kernel_size=(7, 7, 7), is_causal=True
    ).to(device)
    out_layer = na_layer(x_na)
    print(f"   Input:  {x_na.shape}")
    print(f"   Output: {out_layer.shape}")
    assert out_layer.shape == x_na.shape, "Shape mismatch!"
    print("   ✓ Passed")

    # Count parameters
    print("\n" + "=" * 60)
    print("Parameter counts:")
    print(f"   IntraPatchTemporalAttention: {sum(p.numel() for p in intra_attn.parameters()):,}")
    print(f"   AttentionPool: {sum(p.numel() for p in attn_pool.parameters()):,}")
    print(f"   NeighborhoodAttention3D: {sum(p.numel() for p in na.parameters()):,}")
    print(f"   NATransformerLayer: {sum(p.numel() for p in na_layer.parameters()):,}")
    print("=" * 60)
