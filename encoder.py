"""
Encoder and Decoder modules for PDE data with mixed dimensions.
Supports 1D, 2D, and 3D spatial data with temporal dynamics.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


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
    
    print("Testing 1D Encoder/Decoder...")
    x_1d = torch.randn(2, 16, 1024, 6)
    encoder_1d = PDE1DEncoder()
    decoder_1d = PDE1DDecoder()
    tokens_1d = encoder_1d(x_1d)
    recon_1d = decoder_1d(tokens_1d)
    print(f"  Input: {x_1d.shape} → Tokens: {tokens_1d.shape} → Output: {recon_1d.shape}")
    assert recon_1d.shape == x_1d.shape, "Shape mismatch!"
    
    print("\nTesting 2D Encoder/Decoder...")
    x_2d = torch.randn(2, 16, 128, 128, 6)
    encoder_2d = PDE2DEncoder()
    decoder_2d = PDE2DDecoder()
    tokens_2d = encoder_2d(x_2d)
    recon_2d = decoder_2d(tokens_2d)
    print(f"  Input: {x_2d.shape} → Tokens: {tokens_2d.shape} → Output: {recon_2d.shape}")
    assert recon_2d.shape == x_2d.shape, "Shape mismatch!"
    
    print("\nTesting 3D Encoder/Decoder...")
    x_3d = torch.randn(2, 16, 128, 128, 128, 6)
    encoder_3d = PDE3DEncoder()
    decoder_3d = PDE3DDecoder()
    tokens_3d = encoder_3d(x_3d)
    recon_3d = decoder_3d(tokens_3d)
    print(f"  Input: {x_3d.shape} → Tokens: {tokens_3d.shape} → Output: {recon_3d.shape}")
    assert recon_3d.shape == x_3d.shape, "Shape mismatch!"
    
    print("\nTesting Autoencoder wrapper...")
    model = PDEAutoencoder()
    print(f"  1D: {x_1d.shape} → {model(x_1d).shape}")
    print(f"  2D: {x_2d.shape} → {model(x_2d).shape}")
    print(f"  3D: {x_3d.shape} → {model(x_3d).shape}")
    
    print("\n✓ All tests passed!")