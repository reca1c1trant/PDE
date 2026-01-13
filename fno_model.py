"""
Fourier Neural Operator (FNO) for 2D PDE solving.

Architecture:
    Input [B, H, W, C]
        ↓
    Lifting (Linear: C → width)
        ↓
    Fourier Layer × n_layers
        ├── FFT → spectral convolution → IFFT
        └── + Local Conv1x1 + GELU
        ↓
    Projection (width → out_channels)
        ↓
    Output [B, H, W, out_channels]

Reference: https://arxiv.org/abs/2010.08895
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class SpectralConv2d(nn.Module):
    """
    2D Spectral Convolution Layer.

    Performs convolution in Fourier space by:
    1. FFT to frequency domain
    2. Multiply with learnable complex weights (truncated to `modes` lowest frequencies)
    3. IFFT back to spatial domain

    Args:
        in_channels: Input channel dimension
        out_channels: Output channel dimension
        modes1: Number of Fourier modes to keep in first spatial dimension
        modes2: Number of Fourier modes to keep in second spatial dimension
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes in y direction
        self.modes2 = modes2  # Number of Fourier modes in x direction

        self.scale = 1 / (in_channels * out_channels)

        # Complex weights for spectral convolution
        # weights1: for positive frequencies in both dimensions
        # weights2: for negative frequencies in dimension 1, positive in dimension 2
        self.weights1 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Complex multiplication in Fourier space.

        Args:
            input: [B, in_channels, modes1, modes2] complex tensor
            weights: [in_channels, out_channels, modes1, modes2] complex tensor

        Returns:
            output: [B, out_channels, modes1, modes2] complex tensor
        """
        # Einstein summation: batch, input, mode1, mode2 -> batch, output, mode1, mode2
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through spectral convolution.

        Args:
            x: [B, C, H, W] tensor in spatial domain

        Returns:
            [B, out_channels, H, W] tensor in spatial domain
        """
        B, C, H, W = x.shape

        # Compute 2D FFT
        x_ft = torch.fft.rfft2(x)  # [B, C, H, W//2+1]

        # Output tensor in Fourier space
        out_ft = torch.zeros(
            B, self.out_channels, H, W // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        # Multiply relevant Fourier modes
        # modes1 corresponds to y (height), modes2 corresponds to x (width, rfft)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2],
            self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2],
            self.weights2
        )

        # Inverse FFT back to spatial domain
        x = torch.fft.irfft2(out_ft, s=(H, W))

        return x


class FourierLayer(nn.Module):
    """
    Single Fourier Layer = Spectral Conv + Local Conv + Activation.

    Architecture:
        x → SpectralConv2d → x1
        x → Conv1x1        → x2
        out = activation(x1 + x2)
    """

    def __init__(
        self,
        width: int,
        modes1: int,
        modes2: int,
        activation: str = "gelu",
    ):
        super().__init__()
        self.width = width

        # Spectral convolution (frequency domain)
        self.spectral_conv = SpectralConv2d(width, width, modes1, modes2)

        # Local convolution (spatial domain, 1x1 kernel)
        self.local_conv = nn.Conv2d(width, width, kernel_size=1)

        # Activation
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, width, H, W]
        Returns:
            [B, width, H, W]
        """
        x1 = self.spectral_conv(x)
        x2 = self.local_conv(x)
        return self.activation(x1 + x2)


class FNO2D(nn.Module):
    """
    2D Fourier Neural Operator.

    Processes spatial data without downsampling, learning in frequency domain.

    Input/Output formats supported:
    - Single frame: [B, H, W, C_in] → [B, H, W, C_out]
    - Time series: [B, T, H, W, C_in] → [B, T, H, W, C_out] (autoregressive or parallel)

    Args:
        in_channels: Input channels (e.g., 2 for Burgers u,v)
        out_channels: Output channels (same as input for prediction)
        modes: Number of Fourier modes to keep (same for both dimensions)
        width: Hidden channel width
        n_layers: Number of Fourier layers
        activation: Activation function ("gelu", "relu", "silu")
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        modes: int = 12,
        width: int = 32,
        n_layers: int = 4,
        activation: str = "gelu",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.width = width
        self.n_layers = n_layers

        # Lifting layer: in_channels → width
        self.lifting = nn.Linear(in_channels, width)

        # Fourier layers
        self.fourier_layers = nn.ModuleList([
            FourierLayer(width, modes, modes, activation)
            for _ in range(n_layers)
        ])

        # Projection layers: width → 128 → out_channels
        self.proj1 = nn.Linear(width, 128)
        self.proj2 = nn.Linear(128, out_channels)

        # Activation for projection
        if activation == "gelu":
            self.proj_activation = nn.GELU()
        elif activation == "relu":
            self.proj_activation = nn.ReLU()
        else:
            self.proj_activation = nn.SiLU()

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for single frame.

        Args:
            x: [B, H, W, C_in] or [B, C_in, H, W]

        Returns:
            [B, H, W, C_out]
        """
        # Ensure input is [B, H, W, C]
        if x.dim() == 4 and x.shape[1] == self.in_channels:
            # Input is [B, C, H, W], convert to [B, H, W, C]
            x = x.permute(0, 2, 3, 1)

        B, H, W, C = x.shape

        # Lifting: [B, H, W, C] → [B, H, W, width]
        x = self.lifting(x)

        # Permute to [B, width, H, W] for conv operations
        x = x.permute(0, 3, 1, 2)

        # Fourier layers
        for layer in self.fourier_layers:
            x = layer(x)

        # Permute back to [B, H, W, width]
        x = x.permute(0, 2, 3, 1)

        # Projection: [B, H, W, width] → [B, H, W, out_channels]
        x = self.proj1(x)
        x = self.proj_activation(x)
        x = self.proj2(x)

        return x

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "autoregressive",
    ) -> torch.Tensor:
        """
        Forward pass with support for time series.

        Args:
            x: Input tensor
               - Single frame: [B, H, W, C_in]
               - Time series: [B, T, H, W, C_in]
            mode: Processing mode for time series
               - "autoregressive": Given t[0:16], predict t[1:17] one step at a time
               - "parallel": Process all frames independently

        Returns:
            Prediction tensor with same spatial dimensions as input
        """
        if x.dim() == 4:
            # Single frame [B, H, W, C]
            return self.forward_single(x)

        elif x.dim() == 5:
            # Time series [B, T, H, W, C]
            B, T, H, W, C = x.shape

            if mode == "parallel":
                # Process all frames independently
                # Reshape to [B*T, H, W, C]
                x_flat = x.reshape(B * T, H, W, C)
                out_flat = self.forward_single(x_flat)
                return out_flat.reshape(B, T, H, W, -1)

            elif mode == "autoregressive":
                # Autoregressive: input t[0:T-1], output t[1:T]
                # For training: use teacher forcing (input = ground truth)
                outputs = []
                for t in range(T - 1):
                    # Predict next frame from current frame
                    x_t = x[:, t]  # [B, H, W, C]
                    out_t = self.forward_single(x_t)  # [B, H, W, C_out]
                    outputs.append(out_t)

                # Stack: [B, T-1, H, W, C_out]
                return torch.stack(outputs, dim=1)

            else:
                raise ValueError(f"Unknown mode: {mode}")

        else:
            raise ValueError(f"Expected 4D or 5D input, got {x.dim()}D")

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FNO2DForBurgers(nn.Module):
    """
    FNO2D wrapper for Burgers equation with 6-channel I/O.

    Handles the channel padding (6 channels) and extracts only the first 2 channels
    for actual computation.

    Input: [B, T, H, W, 6] where only first 2 channels are valid (u, v)
    Output: [B, T-1, H, W, 6] predictions padded to 6 channels
    """

    def __init__(
        self,
        modes: int = 12,
        width: int = 32,
        n_layers: int = 4,
        activation: str = "gelu",
    ):
        super().__init__()
        self.fno = FNO2D(
            in_channels=2,
            out_channels=2,
            modes=modes,
            width=width,
            n_layers=n_layers,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, H, W, 6] input with 6 channels (only first 2 used)

        Returns:
            [B, T-1, H, W, 6] predictions (only first 2 channels valid)
        """
        B, T, H, W, C = x.shape

        # Extract u, v channels
        x_uv = x[..., :2]  # [B, T, H, W, 2]

        # Run FNO autoregressive
        out_uv = self.fno(x_uv, mode="autoregressive")  # [B, T-1, H, W, 2]

        # Pad back to 6 channels
        out = torch.zeros(B, T - 1, H, W, 6, device=x.device, dtype=x.dtype)
        out[..., :2] = out_uv

        return out

    def count_parameters(self) -> int:
        return self.fno.count_parameters()


class FNO2DForFlowMixing(nn.Module):
    """
    FNO2D wrapper for Flow Mixing equation with 6-channel I/O.

    Flow Mixing uses only 1 channel (u).

    Input: [B, T, H, W, 6] where only first channel is valid
    Output: [B, T-1, H, W, 6] predictions padded to 6 channels
    """

    def __init__(
        self,
        modes: int = 12,
        width: int = 32,
        n_layers: int = 4,
        activation: str = "gelu",
    ):
        super().__init__()
        self.fno = FNO2D(
            in_channels=1,
            out_channels=1,
            modes=modes,
            width=width,
            n_layers=n_layers,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, H, W, 6] input with 6 channels (only first 1 used)

        Returns:
            [B, T-1, H, W, 6] predictions (only first 1 channel valid)
        """
        B, T, H, W, C = x.shape

        # Extract u channel
        x_u = x[..., :1]  # [B, T, H, W, 1]

        # Run FNO autoregressive
        out_u = self.fno(x_u, mode="autoregressive")  # [B, T-1, H, W, 1]

        # Pad back to 6 channels
        out = torch.zeros(B, T - 1, H, W, 6, device=x.device, dtype=x.dtype)
        out[..., :1] = out_u

        return out

    def count_parameters(self) -> int:
        return self.fno.count_parameters()


def get_fno_model(
    task: str = "burgers",
    modes: int = 12,
    width: int = 32,
    n_layers: int = 4,
    activation: str = "gelu",
) -> nn.Module:
    """
    Factory function to create FNO model for specific task.

    Args:
        task: "burgers" or "flow_mixing"
        modes: Number of Fourier modes
        width: Hidden channel width
        n_layers: Number of Fourier layers
        activation: Activation function

    Returns:
        FNO model configured for the task
    """
    if task == "burgers":
        return FNO2DForBurgers(
            modes=modes,
            width=width,
            n_layers=n_layers,
            activation=activation,
        )
    elif task in ["flow_mixing", "flow"]:
        return FNO2DForFlowMixing(
            modes=modes,
            width=width,
            n_layers=n_layers,
            activation=activation,
        )
    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing FNO2D Model")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Test basic FNO2D
    print("\n--- Basic FNO2D ---")
    fno = FNO2D(
        in_channels=2,
        out_channels=2,
        modes=12,
        width=32,
        n_layers=4,
    ).to(device)

    print(f"Parameters: {fno.count_parameters():,}")

    # Test single frame
    x_single = torch.randn(2, 128, 128, 2).to(device)
    out_single = fno.forward_single(x_single)
    print(f"Single frame: {x_single.shape} → {out_single.shape}")

    # Test time series
    x_ts = torch.randn(2, 17, 128, 128, 2).to(device)
    out_ts = fno(x_ts, mode="autoregressive")
    print(f"Time series (AR): {x_ts.shape} → {out_ts.shape}")

    out_parallel = fno(x_ts, mode="parallel")
    print(f"Time series (parallel): {x_ts.shape} → {out_parallel.shape}")

    # Test Burgers wrapper
    print("\n--- FNO2D for Burgers ---")
    fno_burgers = FNO2DForBurgers(modes=12, width=32, n_layers=4).to(device)
    print(f"Parameters: {fno_burgers.count_parameters():,}")

    x_burgers = torch.randn(2, 17, 128, 128, 6).to(device)
    out_burgers = fno_burgers(x_burgers)
    print(f"Burgers: {x_burgers.shape} → {out_burgers.shape}")

    # Test Flow Mixing wrapper
    print("\n--- FNO2D for Flow Mixing ---")
    fno_flow = FNO2DForFlowMixing(modes=12, width=32, n_layers=4).to(device)
    print(f"Parameters: {fno_flow.count_parameters():,}")

    x_flow = torch.randn(2, 17, 128, 128, 6).to(device)
    out_flow = fno_flow(x_flow)
    print(f"Flow Mixing: {x_flow.shape} → {out_flow.shape}")

    # Gradient check
    print("\n--- Gradient Check ---")
    x_test = torch.randn(1, 17, 128, 128, 6, requires_grad=True).to(device)
    out_test = fno_burgers(x_test)
    loss = out_test.sum()
    loss.backward()
    print(f"Input grad shape: {x_test.grad.shape}")
    print(f"Gradient computed successfully!")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
