"""
Encoder and Decoder modules for PDE data with mixed dimensions.
Supports 1D, 2D, and 3D spatial data with temporal dynamics.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict


# ============================================================
# 残差块 (用于方案 B)
# ============================================================
class ResBlock2D(nn.Module):
    """残差块 with GroupNorm"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode='replicate'),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode='replicate'),
        )

    def forward(self, x):
        return x + self.conv(x)


# ============================================================
# 新版 Encoder/Decoder (方案 A+B: 可配置 + 残差块)
# ============================================================
class PDE2DEncoderV2(nn.Module):
    """
    升级版 2D Encoder (方案 A+B):
    - 可配置的中间维度 (mid_channels)
    - 残差块增强表达能力
    - 支持从 config 传入参数

    Input: [B, T, H, W, C] where T=16, H=W=128, C=6
    Output: [B, seq_len, hidden_dim] where seq_len=4096
    """
    def __init__(
        self,
        in_channels: int = 6,
        hidden_dim: int = 768,
        mid_channels: int = 256,  # 中间特征维度 (方案 A)
        use_resblock: bool = True,  # 是否使用残差块 (方案 B)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.mid_channels = mid_channels

        # 每个分支输出 mid_channels // 2
        branch_out = mid_channels // 2

        # Vector branch: 128 → 64 → 32 → 16
        if use_resblock:
            self.vector_conv = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, padding_mode='replicate'),
                nn.GELU(),
                ResBlock2D(64),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, padding_mode='replicate'),
                nn.GELU(),
                ResBlock2D(128),
                nn.Conv2d(128, branch_out, kernel_size=3, stride=2, padding=1, padding_mode='replicate'),
                nn.GELU(),
                ResBlock2D(branch_out),
            )
        else:
            self.vector_conv = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, padding_mode='replicate'),
                nn.GELU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, padding_mode='replicate'),
                nn.GELU(),
                nn.Conv2d(64, branch_out, kernel_size=3, stride=2, padding=1, padding_mode='replicate'),
                nn.GELU(),
            )

        # Scalar branch: 同上
        if use_resblock:
            self.scalar_conv = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, padding_mode='replicate'),
                nn.GELU(),
                ResBlock2D(64),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, padding_mode='replicate'),
                nn.GELU(),
                ResBlock2D(128),
                nn.Conv2d(128, branch_out, kernel_size=3, stride=2, padding=1, padding_mode='replicate'),
                nn.GELU(),
                ResBlock2D(branch_out),
            )
        else:
            self.scalar_conv = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, padding_mode='replicate'),
                nn.GELU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, padding_mode='replicate'),
                nn.GELU(),
                nn.Conv2d(64, branch_out, kernel_size=3, stride=2, padding=1, padding_mode='replicate'),
                nn.GELU(),
            )

        # Fusion: mid_channels → mid_channels
        if use_resblock:
            self.fusion = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, kernel_size=1),
                nn.GELU(),
                ResBlock2D(mid_channels),
            )
        else:
            self.fusion = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, kernel_size=1),
                nn.GELU(),
            )

        # Project: mid_channels → hidden_dim
        self.proj = nn.Linear(mid_channels, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = x.shape

        # Split vector and scalar
        x_vec = x[..., :3].permute(0, 1, 4, 2, 3).reshape(B * T, 3, H, W)
        x_sca = x[..., 3:].permute(0, 1, 4, 2, 3).reshape(B * T, 3, H, W)

        # Separate branch processing
        x_vec = self.vector_conv(x_vec)  # [B*T, mid//2, 16, 16]
        x_sca = self.scalar_conv(x_sca)  # [B*T, mid//2, 16, 16]

        # Concatenate and fuse
        x = torch.cat([x_vec, x_sca], dim=1)  # [B*T, mid, 16, 16]
        x = self.fusion(x)

        # Reshape and project
        x = x.permute(0, 2, 3, 1).reshape(B, T * 16 * 16, self.mid_channels)
        x = self.proj(x)  # [B, 4096, hidden_dim]

        return x


class PDE2DDecoderV2(nn.Module):
    """
    升级版 2D Decoder (对应 EncoderV2):
    - 可配置的中间维度
    - 残差块增强

    Input: [B, seq_len, hidden_dim] where seq_len=4096
    Output: [B, T, H, W, C] where T=16, H=W=128, C=6
    """
    def __init__(
        self,
        out_channels: int = 6,
        hidden_dim: int = 768,
        mid_channels: int = 256,
        use_resblock: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.mid_channels = mid_channels

        branch_in = mid_channels // 2

        # Project: hidden_dim → mid_channels
        self.proj = nn.Linear(hidden_dim, mid_channels)

        # Split layer
        if use_resblock:
            self.split = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, kernel_size=1),
                nn.GELU(),
                ResBlock2D(mid_channels),
            )
        else:
            self.split = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, kernel_size=1),
                nn.GELU(),
            )

        # Vector branch: 16 → 32 → 64 → 128
        if use_resblock:
            self.vector_conv = nn.Sequential(
                ResBlock2D(branch_in),
                nn.ConvTranspose2d(branch_in, 128, kernel_size=4, stride=2, padding=1),
                nn.GELU(),
                ResBlock2D(128),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.GELU(),
                ResBlock2D(64),
                nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            )
        else:
            self.vector_conv = nn.Sequential(
                nn.ConvTranspose2d(branch_in, 64, kernel_size=4, stride=2, padding=1),
                nn.GELU(),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.GELU(),
                nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            )

        # Scalar branch: 同上
        if use_resblock:
            self.scalar_conv = nn.Sequential(
                ResBlock2D(branch_in),
                nn.ConvTranspose2d(branch_in, 128, kernel_size=4, stride=2, padding=1),
                nn.GELU(),
                ResBlock2D(128),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.GELU(),
                ResBlock2D(64),
                nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            )
        else:
            self.scalar_conv = nn.Sequential(
                nn.ConvTranspose2d(branch_in, 64, kernel_size=4, stride=2, padding=1),
                nn.GELU(),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.GELU(),
                nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, seq_len, D = x.shape
        T = seq_len // (16 * 16)

        # Project
        x = self.proj(x)  # [B, 4096, mid_channels]

        # Reshape
        x = x.reshape(B * T, 16, 16, self.mid_channels).permute(0, 3, 1, 2)

        # Split
        x = self.split(x)  # [B*T, mid, 16, 16]
        branch_ch = self.mid_channels // 2
        x_vec = x[:, :branch_ch, :, :]
        x_sca = x[:, branch_ch:, :, :]

        # Upsample
        x_vec = self.vector_conv(x_vec)  # [B*T, 3, 128, 128]
        x_sca = self.scalar_conv(x_sca)  # [B*T, 3, 128, 128]

        # Concatenate
        x = torch.cat([x_vec, x_sca], dim=1)  # [B*T, 6, 128, 128]

        # Reshape back
        x = x.permute(0, 2, 3, 1).reshape(B, T, 128, 128, self.out_channels)

        return x


def create_encoder_v2(config: Dict) -> PDE2DEncoderV2:
    """从 config 创建 EncoderV2"""
    enc_config = config['model'].get('encoder', {})
    return PDE2DEncoderV2(
        in_channels=config['model']['in_channels'],
        hidden_dim=config['model']['transformer']['hidden_size'],
        mid_channels=enc_config.get('mid_channels', 256),
        use_resblock=enc_config.get('use_resblock', True),
    )


def create_decoder_v2(config: Dict) -> PDE2DDecoderV2:
    """从 config 创建 DecoderV2"""
    enc_config = config['model'].get('encoder', {})
    return PDE2DDecoderV2(
        out_channels=config['model']['in_channels'],
        hidden_dim=config['model']['transformer']['hidden_size'],
        mid_channels=enc_config.get('mid_channels', 256),
        use_resblock=enc_config.get('use_resblock', True),
    )


# ============================================================
# 原始版本 (保持向后兼容)
# ============================================================
class PDE1DEncoder(nn.Module):
    """
    Encoder for 1D PDE data with multi-stage downsampling.
    Input: [B, T, H, C] where T=16, H=1024, C=6 (vector=3, scalar=3)
    Output: [B, seq_len, D] where seq_len=4096, D=hidden_dim

    Processing:
        1. Separate vector [vx,vy,vz] and scalar [p,ρ,T] branches
        2. Downsample: 1024 → 512 → 256 (2 stages, stride=2 each)
        3. Cross-branch fusion via 1x1 conv
        4. Flatten: T*H = 16*256 = 4096 tokens
        5. Project to hidden_dim
    """

    def __init__(self, in_channels: int = 6, hidden_dim: int = 768):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.vector_channels = 3  # vx, vy, vz
        self.scalar_channels = 3  # p, ρ, T

        # Vector branch: 1024 → 512 → 256
        self.vector_conv = nn.Sequential(
            nn.Conv1d(self.vector_channels, 32, kernel_size=3, stride=2, padding=1, padding_mode='replicate'),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, padding_mode='replicate'),
            nn.GELU(),
        )

        # Scalar branch: 1024 → 512 → 256
        self.scalar_conv = nn.Sequential(
            nn.Conv1d(self.scalar_channels, 32, kernel_size=3, stride=2, padding=1, padding_mode='replicate'),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, padding_mode='replicate'),
            nn.GELU(),
        )

        # Cross-branch fusion: concat (128) → fused (128)
        self.fusion = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1),
            nn.GELU(),
        )

        # Project to hidden dimension
        self.proj = nn.Linear(128, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, H, C] where T=16, H=1024, C=6
        Returns:
            tokens: [B, seq_len, D] where seq_len=4096
        """
        B, T, H, C = x.shape
        # assert H == 1024, f"Expected H=1024, got {H}"
        # assert C == self.in_channels, f"Expected C={self.in_channels}, got {C}"

        # Split vector and scalar: [B, T, H, 3] each
        x_vec = x[..., :3]  # vx, vy, vz
        x_sca = x[..., 3:]  # p, ρ, T

        # Reshape for conv: [B*T, C, H]
        x_vec = x_vec.permute(0, 1, 3, 2).reshape(B * T, 3, H)
        x_sca = x_sca.permute(0, 1, 3, 2).reshape(B * T, 3, H)

        # Separate branch processing
        x_vec = self.vector_conv(x_vec)  # [B*T, 64, 256]
        x_sca = self.scalar_conv(x_sca)  # [B*T, 64, 256]

        # Concatenate and fuse
        x = torch.cat([x_vec, x_sca], dim=1)  # [B*T, 128, 256]
        x = self.fusion(x)  # [B*T, 128, 256]

        # Reshape and flatten
        x = x.permute(0, 2, 1)  # [B*T, 256, 128]
        x = x.reshape(B, T * 256, 128)  # [B, 4096, 128]

        # Project
        x = self.proj(x)  # [B, 4096, D]

        return x


class PDE2DEncoder(nn.Module):
    """
    Encoder for 2D PDE data with multi-stage downsampling.
    Input: [B, T, H, W, C] where T=16, H=W=128, C=6 (vector=3, scalar=3)
    Output: [B, seq_len, D] where seq_len=4096, D=hidden_dim

    Processing:
        1. Separate vector [vx,vy,vz] and scalar [p,ρ,T] branches
        2. Downsample: 128 → 64 → 32 → 16 (3 stages, stride=2 each)
        3. Cross-branch fusion via 1x1 conv
        4. Flatten: T*H*W = 16*16*16 = 4096 tokens
        5. Project to hidden_dim
    """

    def __init__(self, in_channels: int = 6, hidden_dim: int = 768):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.vector_channels = 3  # vx, vy, vz
        self.scalar_channels = 3  # p, ρ, T

        # Vector branch: 128 → 64 → 32 → 16
        self.vector_conv = nn.Sequential(
            nn.Conv2d(self.vector_channels, 16, kernel_size=3, stride=2, padding=1, padding_mode='replicate'),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, padding_mode='replicate'),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, padding_mode='replicate'),
            nn.GELU(),
        )

        # Scalar branch: 128 → 64 → 32 → 16
        self.scalar_conv = nn.Sequential(
            nn.Conv2d(self.scalar_channels, 16, kernel_size=3, stride=2, padding=1, padding_mode='replicate'),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, padding_mode='replicate'),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, padding_mode='replicate'),
            nn.GELU(),
        )

        # Cross-branch fusion: concat (128) → fused (128)
        self.fusion = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1),
            nn.GELU(),
        )

        # Project to hidden dimension
        self.proj = nn.Linear(128, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, H, W, C] where T=16, H=W=128, C=6
        Returns:
            tokens: [B, seq_len, D] where seq_len=4096
        """
        B, T, H, W, C = x.shape
        assert H == 128 and W == 128, f"Expected H=W=128, got {H}x{W}"
        assert C == self.in_channels, f"Expected C={self.in_channels}, got {C}"

        # Split vector and scalar: [B, T, H, W, 3] each
        x_vec = x[..., :3]  # vx, vy, vz
        x_sca = x[..., 3:]  # p, ρ, T

        # Reshape for conv: [B*T, C, H, W]
        x_vec = x_vec.permute(0, 1, 4, 2, 3).reshape(B * T, 3, H, W)
        x_sca = x_sca.permute(0, 1, 4, 2, 3).reshape(B * T, 3, H, W)

        # Separate branch processing
        x_vec = self.vector_conv(x_vec)  # [B*T, 64, 16, 16]
        x_sca = self.scalar_conv(x_sca)  # [B*T, 64, 16, 16]

        # Concatenate and fuse
        x = torch.cat([x_vec, x_sca], dim=1)  # [B*T, 128, 16, 16]
        x = self.fusion(x)  # [B*T, 128, 16, 16]

        # Reshape and flatten
        x = x.permute(0, 2, 3, 1)  # [B*T, 16, 16, 128]
        x = x.reshape(B, T * 16 * 16, 128)  # [B, 4096, 128]

        # Project
        x = self.proj(x)  # [B, 4096, D]

        return x


class PDE3DEncoder(nn.Module):
    """
    Encoder for 3D PDE data.
    Input: [B, T, H, W, D, C] where T=16, H=W=D=128, C=6
    Output: [B, seq_len, D_hidden] where seq_len=65536, D_hidden=hidden_dim
    
    Processing:
        1. Downsample: 128x128x128 → 16x16x16 (8x)
        2. Flatten: T*H*W*D = 16*16*16*16 = 65536 tokens
        3. Project to hidden_dim
    """
    
    def __init__(self, in_channels: int = 6, hidden_dim: int = 768):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        
        # Spatial downsampling: 128x128x128 → 16x16x16
        self.conv_down = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=8, stride=8, padding=0),  # 128 → 16
            nn.GELU(),
            nn.Conv3d(64, 128, kernel_size=1),
            nn.GELU(),
        )
        
        # Project to hidden dimension
        self.proj = nn.Linear(128, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, H, W, D, C] where T=16, H=W=D=128, C=6
        Returns:
            tokens: [B, seq_len, D_hidden] where seq_len=65536
        """
        B, T, H, W, D, C = x.shape
        assert H == 128 and W == 128 and D == 128, f"Expected H=W=D=128, got {H}x{W}x{D}"
        assert C == self.in_channels, f"Expected C={self.in_channels}, got {C}"
        
        # Process each timestep
        x = x.permute(0, 1, 5, 2, 3, 4)  # [B, T, C, H, W, D]
        x = x.reshape(B * T, C, H, W, D)  # [B*T, C, H, W, D]
        
        # Downsample
        x = self.conv_down(x)  # [B*T, 128, 16, 16, 16]
        
        # Reshape and flatten
        x = x.permute(0, 2, 3, 4, 1)  # [B*T, 16, 16, 16, 128]
        x = x.reshape(B, T * 16 * 16 * 16, 128)  # [B, 65536, 128]
        
        # Project
        x = self.proj(x)  # [B, 65536, D_hidden]
        
        return x


class PDE1DDecoder(nn.Module):
    """
    Decoder for 1D PDE data with multi-stage upsampling (mirrors encoder).
    Input: [B, seq_len, D] where seq_len=4096, D=hidden_dim
    Output: [B, T, H, C] where T=16, H=1024, C=6 (vector=3, scalar=3)

    Processing:
        1. Project from hidden_dim
        2. Split into vector/scalar branches
        3. Upsample: 256 → 512 → 1024 (2 stages, stride=2 each)
        4. Concatenate outputs
    """

    def __init__(self, out_channels: int = 6, hidden_dim: int = 768):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.vector_channels = 3  # vx, vy, vz
        self.scalar_channels = 3  # p, ρ, T

        # Project from hidden dimension
        self.proj = nn.Linear(hidden_dim, 128)

        # Split layer: 128 → 64 + 64
        self.split = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1),
            nn.GELU(),
        )

        # Vector branch: 256 → 512 → 1024
        self.vector_conv = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(32, self.vector_channels, kernel_size=4, stride=2, padding=1),
        )

        # Scalar branch: 256 → 512 → 1024
        self.scalar_conv = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(32, self.scalar_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, D] where seq_len=4096
        Returns:
            output: [B, T, H, C] where T=16, H=1024
        """
        B, seq_len, D = x.shape
        assert seq_len == 4096, f"Expected seq_len=4096, got {seq_len}"

        # Project
        x = self.proj(x)  # [B, 4096, 128]

        # Reshape
        x = x.reshape(B * 16, 256, 128)  # [B*T, 256, 128]
        x = x.permute(0, 2, 1)  # [B*T, 128, 256]

        # Split preparation
        x = self.split(x)  # [B*T, 128, 256]
        x_vec = x[:, :64, :]  # [B*T, 64, 256]
        x_sca = x[:, 64:, :]  # [B*T, 64, 256]

        # Separate branch upsampling
        x_vec = self.vector_conv(x_vec)  # [B*T, 3, 1024]
        x_sca = self.scalar_conv(x_sca)  # [B*T, 3, 1024]

        # Concatenate
        x = torch.cat([x_vec, x_sca], dim=1)  # [B*T, 6, 1024]

        # Reshape back
        x = x.permute(0, 2, 1)  # [B*T, 1024, 6]
        x = x.reshape(B, 16, 1024, self.out_channels)  # [B, T, H, C]

        return x


class PDE2DDecoder(nn.Module):
    """
    Decoder for 2D PDE data with multi-stage upsampling (mirrors encoder).
    Input: [B, seq_len, D] where seq_len=4096, D=hidden_dim
    Output: [B, T, H, W, C] where T=16, H=W=128, C=6 (vector=3, scalar=3)

    Processing:
        1. Project from hidden_dim
        2. Split into vector/scalar branches
        3. Upsample: 16 → 32 → 64 → 128 (3 stages, stride=2 each)
        4. Concatenate outputs
    """

    def __init__(self, out_channels: int = 6, hidden_dim: int = 768):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.vector_channels = 3  # vx, vy, vz
        self.scalar_channels = 3  # p, ρ, T

        # Project from hidden dimension
        self.proj = nn.Linear(hidden_dim, 128)

        # Split layer: 128 → 64 + 64
        self.split = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1),
            nn.GELU(),
        )

        # Vector branch: 16 → 32 → 64 → 128
        self.vector_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(16, self.vector_channels, kernel_size=4, stride=2, padding=1),
        )

        # Scalar branch: 16 → 32 → 64 → 128
        self.scalar_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(16, self.scalar_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, D] where seq_len=4096
        Returns:
            output: [B, T, H, W, C] where T=16, H=W=128
        """
        B, seq_len, D = x.shape
        # assert seq_len == 4096, f"Expected seq_len=4096, got {seq_len}"

        # Project
        x = self.proj(x)  # [B, 4096, 128]
        T = seq_len // (16 * 16)  # 16
        # Reshape
        x = x.reshape(B * T, 16, 16, 128)  # [B*T, 16, 16, 128]
        x = x.permute(0, 3, 1, 2)  # [B*T, 128, 16, 16]

        # Split preparation
        x = self.split(x)  # [B*T, 128, 16, 16]
        x_vec = x[:, :64, :, :]  # [B*T, 64, 16, 16]
        x_sca = x[:, 64:, :, :]  # [B*T, 64, 16, 16]

        # Separate branch upsampling
        x_vec = self.vector_conv(x_vec)  # [B*T, 3, 128, 128]
        x_sca = self.scalar_conv(x_sca)  # [B*T, 3, 128, 128]

        # Concatenate
        x = torch.cat([x_vec, x_sca], dim=1)  # [B*T, 6, 128, 128]

        # Reshape back
        x = x.permute(0, 2, 3, 1)  # [B*T, 128, 128, 6]
        x = x.reshape(B, T, 128, 128, self.out_channels)  # [B, T, H, W, C]

        return x


class PDE3DDecoder(nn.Module):
    """
    Decoder for 3D PDE data (mirrors encoder).
    Input: [B, seq_len, D] where seq_len=65536, D=hidden_dim
    Output: [B, T, H, W, D, C] where T=16, H=W=D=128, C=6
    """
    
    def __init__(self, out_channels: int = 6, hidden_dim: int = 768):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        
        # Project from hidden dimension
        self.proj = nn.Linear(hidden_dim, 128)
        
        # Spatial upsampling: 16x16x16 → 128x128x128
        self.conv_up = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=1),
            nn.GELU(),
            nn.ConvTranspose3d(64, out_channels, kernel_size=8, stride=8, padding=0),  # 16 → 128
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, D] where seq_len=65536
        Returns:
            output: [B, T, H, W, D, C] where T=16, H=W=D=128
        """
        B, seq_len, D = x.shape
        # assert seq_len == 65536, f"Expected seq_len=65536, got {seq_len}"
        
        # Project
        x = self.proj(x)  # [B, 65536, 128]
        
        # Reshape
        x = x.reshape(B * 16, 16, 16, 16, 128)  # [B*T, 16, 16, 16, 128]
        x = x.permute(0, 4, 1, 2, 3)  # [B*T, 128, 16, 16, 16]
        
        # Upsample
        x = self.conv_up(x)  # [B*T, C, 128, 128, 128]
        
        # Reshape back
        x = x.permute(0, 2, 3, 4, 1)  # [B*T, 128, 128, 128, C]
        x = x.reshape(B, 16, 128, 128, 128, self.out_channels)  # [B, T, H, W, D, C]
        
        return x


# ============================================================
# V3: FFT-based Encoder/Decoder
# ============================================================
class FFTBlock(nn.Module):
    """
    FFT处理块：在频域进行可学习的特征变换

    流程：
    1. rfft2 → 提取低频modes
    2. 可学习权重变换（复数乘法）
    3. 零填充 → irfft2
    4. 残差连接 + LayerNorm + FFN
    """
    def __init__(self, channels: int, modes: int = 64):
        super().__init__()
        self.channels = channels
        self.modes = modes

        # 可学习Fourier权重（实部和虚部分开存储）
        # rfft2输出形状: [B, C, H, W//2+1]，所以第二个modes维度是 modes//2+1
        scale = 1.0 / (channels * channels)
        self.weight_re = nn.Parameter(scale * torch.randn(channels, channels, modes, modes // 2 + 1))
        self.weight_im = nn.Parameter(scale * torch.randn(channels, channels, modes, modes // 2 + 1))

        # LayerNorm和FFN
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels)
        )

    def complex_mul(self, x_re: torch.Tensor, x_im: torch.Tensor,
                    w_re: torch.Tensor, w_im: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """复数乘法: (a+bi)(c+di) = (ac-bd) + (ad+bc)i"""
        # x: [B, C_in, modes, modes//2+1]
        # w: [C_in, C_out, modes, modes//2+1]
        # 输出: [B, C_out, modes, modes//2+1]
        out_re = torch.einsum('bcxy,cdxy->bdxy', x_re, w_re) - torch.einsum('bcxy,cdxy->bdxy', x_im, w_im)
        out_im = torch.einsum('bcxy,cdxy->bdxy', x_re, w_im) + torch.einsum('bcxy,cdxy->bdxy', x_im, w_re)
        return out_re, out_im

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, H, W, C] 其中 H=W=128
        Returns:
            x: [B, H, W, C]
        """
        B, H, W, C = x.shape
        residual = x

        # 保存原始dtype，FFT不支持bfloat16
        orig_dtype = x.dtype

        # 转换为 [B, C, H, W] 用于FFT
        x = x.permute(0, 3, 1, 2)

        # 禁用autocast进行FFT操作 (FFT不支持bfloat16)
        with torch.amp.autocast(device_type='cuda', enabled=False):
            # 转换为float32进行FFT操作
            x = x.float()

            # rfft2: 实数FFT，输出 [B, C, H, W//2+1] 复数
            x_ft = torch.fft.rfft2(x, norm='ortho')

            # 提取低频modes
            x_ft_modes = x_ft[:, :, :self.modes, :self.modes // 2 + 1]

            # 可学习权重变换 (权重也需要float32)
            out_re, out_im = self.complex_mul(
                x_ft_modes.real, x_ft_modes.imag,
                self.weight_re.float(), self.weight_im.float()
            )

            # 零填充回原尺寸
            out_ft = torch.zeros_like(x_ft)
            out_ft[:, :, :self.modes, :self.modes // 2 + 1] = torch.complex(out_re, out_im)

            # irfft2: 逆FFT
            x = torch.fft.irfft2(out_ft, s=(H, W), norm='ortho')

        # 转回原始dtype
        x = x.to(orig_dtype)

        # 转回 [B, H, W, C]
        x = x.permute(0, 2, 3, 1)

        # 残差连接 + LayerNorm
        x = self.norm1(x + residual)

        # FFN + 残差
        x = self.norm2(x + self.ffn(x))

        return x


class FFTEncoder2D(nn.Module):
    """
    V3 FFT Encoder: 使用FFT在频域提取特征

    Input: [B, T, 128, 128, 6]
    Output: [B, 4096, hidden_size]

    架构：
    1. 通道投影: 6 → hidden_channels
    2. FFT块 × N: 在128×128上进行频域处理
    3. 空间池化: 128×128 → 16×16
    4. 投影: hidden_channels → hidden_size
    """
    def __init__(
        self,
        in_channels: int = 6,
        hidden_channels: int = 64,
        hidden_size: int = 768,
        modes: int = 64,
        n_blocks: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.hidden_size = hidden_size

        # 通道投影
        self.proj_in = nn.Linear(in_channels, hidden_channels)

        # FFT块
        self.blocks = nn.ModuleList([
            FFTBlock(hidden_channels, modes) for _ in range(n_blocks)
        ])

        # 空间池化: 128×128 → 16×16 (stride=8)
        self.pool = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=8, stride=8)
        self.norm = nn.LayerNorm(hidden_channels)

        # 最终投影到transformer的hidden_size
        self.proj_out = nn.Linear(hidden_channels, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, H, W, C] 其中 T=16, H=W=128, C=6
        Returns:
            tokens: [B, 4096, hidden_size]
        """
        B, T, H, W, C = x.shape

        # 合并B和T维度
        x = x.reshape(B * T, H, W, C)

        # 通道投影
        x = self.proj_in(x)  # [B*T, 128, 128, hidden_channels]

        # FFT块处理
        for block in self.blocks:
            x = block(x)

        # 空间池化
        x = x.permute(0, 3, 1, 2)  # [B*T, C, 128, 128]
        x = self.pool(x)  # [B*T, C, 16, 16]
        x = x.permute(0, 2, 3, 1)  # [B*T, 16, 16, C]
        x = self.norm(x)

        # Flatten空间维度
        x = x.reshape(B * T, 256, self.hidden_channels)  # [B*T, 256, C]

        # 最终投影
        x = self.proj_out(x)  # [B*T, 256, hidden_size]

        # 重塑为序列
        x = x.reshape(B, T * 256, self.hidden_size)  # [B, 4096, hidden_size]

        return x


class FFTDecoder2D(nn.Module):
    """
    V3 FFT Decoder: 对应FFTEncoder2D的解码器

    Input: [B, 4096, hidden_size]
    Output: [B, T, 128, 128, 6]

    架构：
    1. 投影: hidden_size → hidden_channels
    2. 空间上采样: 16×16 → 128×128
    3. FFT块 × N
    4. 输出投影: hidden_channels → 6
    """
    def __init__(
        self,
        out_channels: int = 6,
        hidden_channels: int = 64,
        hidden_size: int = 768,
        modes: int = 64,
        n_blocks: int = 4,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.hidden_size = hidden_size

        # 从hidden_size投影
        self.proj_in = nn.Linear(hidden_size, hidden_channels)

        # 空间上采样: 16×16 → 128×128 (stride=8)
        self.upsample = nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=8, stride=8)
        self.norm_up = nn.LayerNorm(hidden_channels)

        # FFT块
        self.blocks = nn.ModuleList([
            FFTBlock(hidden_channels, modes) for _ in range(n_blocks)
        ])

        # 输出投影
        self.proj_out = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, B: int = None, T: int = None) -> torch.Tensor:
        """
        Args:
            x: [B, 4096, hidden_size]
            B: batch size (可选，用于reshape)
            T: temporal length (可选，用于reshape)
        Returns:
            output: [B, T, 128, 128, out_channels]
        """
        if B is None:
            B = x.shape[0]
        if T is None:
            T = x.shape[1] // 256  # 4096 / 256 = 16

        # 投影
        x = self.proj_in(x)  # [B, 4096, hidden_channels]

        # 重塑
        x = x.reshape(B * T, 256, self.hidden_channels)  # [B*T, 256, C]
        x = x.reshape(B * T, 16, 16, self.hidden_channels)  # [B*T, 16, 16, C]

        # 上采样
        x = x.permute(0, 3, 1, 2)  # [B*T, C, 16, 16]
        x = self.upsample(x)  # [B*T, C, 128, 128]
        x = x.permute(0, 2, 3, 1)  # [B*T, 128, 128, C]
        x = self.norm_up(x)

        # FFT块处理
        for block in self.blocks:
            x = block(x)

        # 输出投影
        x = self.proj_out(x)  # [B*T, 128, 128, out_channels]

        # 重塑
        x = x.reshape(B, T, 128, 128, self.out_channels)  # [B, T, 128, 128, out_channels]

        return x


def create_encoder_v3(config: Dict) -> FFTEncoder2D:
    """从config创建FFT Encoder V3"""
    enc_config = config['model'].get('encoder', {})
    return FFTEncoder2D(
        in_channels=config['model']['in_channels'],
        hidden_size=config['model']['transformer']['hidden_size'],
        hidden_channels=enc_config.get('hidden_channels', 64),
        modes=enc_config.get('modes', 64),
        n_blocks=enc_config.get('n_blocks', 4),
    )


def create_decoder_v3(config: Dict) -> FFTDecoder2D:
    """从config创建FFT Decoder V3"""
    enc_config = config['model'].get('encoder', {})
    return FFTDecoder2D(
        out_channels=config['model']['in_channels'],
        hidden_size=config['model']['transformer']['hidden_size'],
        hidden_channels=enc_config.get('hidden_channels', 64),
        modes=enc_config.get('modes', 64),
        n_blocks=enc_config.get('n_blocks', 4),
    )


class PDEAutoencoder(nn.Module):
    """
    Wrapper that selects appropriate encoder/decoder based on input dimension.
    """
    
    def __init__(self, in_channels: int = 6, hidden_dim: int = 768):
        super().__init__()
        
        # Create all encoders and decoders
        self.encoder_1d = PDE1DEncoder(in_channels, hidden_dim)
        self.encoder_2d = PDE2DEncoder(in_channels, hidden_dim)
        self.encoder_3d = PDE3DEncoder(in_channels, hidden_dim)
        
        self.decoder_1d = PDE1DDecoder(in_channels, hidden_dim)
        self.decoder_2d = PDE2DDecoder(in_channels, hidden_dim)
        self.decoder_3d = PDE3DDecoder(in_channels, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Auto-detect dimension and apply appropriate encoder/decoder.
        
        Args:
            x: Input tensor
                - 1D: [B, T, H, C]
                - 2D: [B, T, H, W, C]
                - 3D: [B, T, H, W, D, C]
        Returns:
            Reconstructed tensor with same shape as input
        """
        ndim = x.ndim - 3  # Exclude B, T, C
        
        if ndim == 1:  # 1D
            tokens = self.encoder_1d(x)
            output = self.decoder_1d(tokens)
        elif ndim == 2:  # 2D
            tokens = self.encoder_2d(x)
            output = self.decoder_2d(tokens)
        elif ndim == 3:  # 3D
            tokens = self.encoder_3d(x)
            output = self.decoder_3d(tokens)
        else:
            raise ValueError(f"Unsupported input dimension: {x.shape}")
        
        return output


if __name__ == "__main__":
    """Test encoder/decoder shapes."""

    print("=" * 60)
    print("Testing FFT Encoder/Decoder V3...")
    print("=" * 60)
    x_fft = torch.randn(2, 16, 128, 128, 6)
    encoder_fft = FFTEncoder2D(in_channels=6, hidden_channels=64, hidden_size=768, modes=64, n_blocks=4)
    decoder_fft = FFTDecoder2D(out_channels=6, hidden_channels=64, hidden_size=768, modes=64, n_blocks=4)

    # 计算参数量
    enc_params = sum(p.numel() for p in encoder_fft.parameters())
    dec_params = sum(p.numel() for p in decoder_fft.parameters())
    print(f"  Encoder params: {enc_params / 1e6:.2f}M")
    print(f"  Decoder params: {dec_params / 1e6:.2f}M")

    tokens_fft = encoder_fft(x_fft)
    print(f"  Input: {x_fft.shape} → Tokens: {tokens_fft.shape}")

    recon_fft = decoder_fft(tokens_fft)
    print(f"  Tokens: {tokens_fft.shape} → Output: {recon_fft.shape}")

    assert tokens_fft.shape == (2, 4096, 768), f"Token shape mismatch: {tokens_fft.shape}"
    assert recon_fft.shape == x_fft.shape, f"Output shape mismatch: {recon_fft.shape}"
    print("  ✓ FFT V3 test passed!")

    print("\n" + "=" * 60)
    print("Testing 1D Encoder/Decoder...")
    print("=" * 60)
    x_1d = torch.randn(2, 16, 1024, 6)
    encoder_1d = PDE1DEncoder()
    decoder_1d = PDE1DDecoder()
    tokens_1d = encoder_1d(x_1d)
    recon_1d = decoder_1d(tokens_1d)
    print(f"  Input: {x_1d.shape} → Tokens: {tokens_1d.shape} → Output: {recon_1d.shape}")
    assert recon_1d.shape == x_1d.shape, "Shape mismatch!"
    print("  ✓ 1D test passed!")

    print("\n" + "=" * 60)
    print("Testing 2D Encoder/Decoder...")
    print("=" * 60)
    x_2d = torch.randn(2, 16, 128, 128, 6)
    encoder_2d = PDE2DEncoder()
    decoder_2d = PDE2DDecoder()
    tokens_2d = encoder_2d(x_2d)
    recon_2d = decoder_2d(tokens_2d)
    print(f"  Input: {x_2d.shape} → Tokens: {tokens_2d.shape} → Output: {recon_2d.shape}")
    assert recon_2d.shape == x_2d.shape, "Shape mismatch!"
    print("  ✓ 2D test passed!")

    print("\n" + "=" * 60)
    print("Testing 2D V2 Encoder/Decoder...")
    print("=" * 60)
    encoder_v2 = PDE2DEncoderV2(in_channels=6, hidden_dim=768, mid_channels=256, use_resblock=True)
    decoder_v2 = PDE2DDecoderV2(out_channels=6, hidden_dim=768, mid_channels=256, use_resblock=True)
    tokens_v2 = encoder_v2(x_2d)
    recon_v2 = decoder_v2(tokens_v2)
    print(f"  Input: {x_2d.shape} → Tokens: {tokens_v2.shape} → Output: {recon_v2.shape}")
    assert recon_v2.shape == x_2d.shape, "Shape mismatch!"
    print("  ✓ V2 test passed!")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)