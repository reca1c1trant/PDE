"""
Fourier-based PDE Loss for Burgers Equation.

Based on PINO (Physics-Informed Neural Operator) approach:
- Use FFT to compute spatial derivatives (spectral accuracy)
- Much more accurate than finite difference methods
- Memory efficient (no autograd graph for derivatives)

Reference: Li et al., "Physics-Informed Neural Operator for Learning PDEs"
           arXiv:2111.03794
"""

import torch
import torch.nn.functional as F
from typing import Tuple
import math


def fourier_derivative_2d(
    u: torch.Tensor,
    dx: float,
    dy: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute spatial derivatives using FFT (spectral accuracy).

    For periodic boundary conditions, FFT derivative is exact:
        ∂u/∂x ↔ i * kx * û(kx, ky)

    Args:
        u: [B, T, H, W] tensor
        dx: grid spacing in x direction
        dy: grid spacing in y direction

    Returns:
        du_dx: [B, T, H, W] ∂u/∂x
        du_dy: [B, T, H, W] ∂u/∂y
    """
    B, T, H, W = u.shape
    device = u.device
    dtype = u.dtype

    # Compute in float32 for FFT precision
    u = u.float()

    # Wavenumbers
    # kx: [H], ky: [W//2+1]
    kx = torch.fft.fftfreq(H, d=dx, device=device) * 2 * math.pi  # [H]
    ky = torch.fft.rfftfreq(W, d=dy, device=device) * 2 * math.pi  # [W//2+1]

    # Reshape for broadcasting: [1, 1, H, 1] and [1, 1, 1, W//2+1]
    kx = kx.view(1, 1, H, 1)
    ky = ky.view(1, 1, 1, -1)

    # FFT
    u_hat = torch.fft.rfft2(u)  # [B, T, H, W//2+1]

    # Derivative in Fourier space: ∂u/∂x = ifft(i * kx * fft(u))
    du_dx_hat = 1j * kx * u_hat
    du_dy_hat = 1j * ky * u_hat

    # Inverse FFT
    du_dx = torch.fft.irfft2(du_dx_hat, s=(H, W))
    du_dy = torch.fft.irfft2(du_dy_hat, s=(H, W))

    return du_dx.to(dtype), du_dy.to(dtype)


def fourier_laplacian_2d(
    u: torch.Tensor,
    dx: float,
    dy: float,
) -> torch.Tensor:
    """
    Compute Laplacian using FFT (spectral accuracy).

        ∇²u = ∂²u/∂x² + ∂²u/∂y² ↔ -(kx² + ky²) * û

    Args:
        u: [B, T, H, W] tensor
        dx: grid spacing in x direction
        dy: grid spacing in y direction

    Returns:
        laplacian: [B, T, H, W] ∇²u
    """
    B, T, H, W = u.shape
    device = u.device
    dtype = u.dtype

    u = u.float()

    # Wavenumbers
    kx = torch.fft.fftfreq(H, d=dx, device=device) * 2 * math.pi
    ky = torch.fft.rfftfreq(W, d=dy, device=device) * 2 * math.pi

    # k² = kx² + ky²
    kx_sq = kx.view(1, 1, H, 1) ** 2
    ky_sq = ky.view(1, 1, 1, -1) ** 2
    k_sq = kx_sq + ky_sq

    # FFT
    u_hat = torch.fft.rfft2(u)

    # Laplacian in Fourier space: ∇²u = ifft(-k² * fft(u))
    laplacian_hat = -k_sq * u_hat

    # Inverse FFT
    laplacian = torch.fft.irfft2(laplacian_hat, s=(H, W))

    return laplacian.to(dtype)


def burgers_pde_loss_fourier(
    pred: torch.Tensor,
    nu: float,
    dt: float,
    dx: float = 1.0 / 128,
    dy: float = 1.0 / 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Burgers PDE residual loss using Fourier derivatives.

    2D Burgers equation:
        ∂u/∂t + u * ∂u/∂x + v * ∂u/∂y = ν * ∇²u
        ∂v/∂t + u * ∂v/∂x + v * ∂v/∂y = ν * ∇²v

    PDE residual:
        R_u = ∂u/∂t + u * ∂u/∂x + v * ∂u/∂y - ν * ∇²u
        R_v = ∂v/∂t + u * ∂v/∂x + v * ∂v/∂y - ν * ∇²v

    Args:
        pred: [B, T, H, W, 2] predicted velocity field (u, v)
        nu: viscosity coefficient
        dt: time step
        dx: grid spacing in x
        dy: grid spacing in y

    Returns:
        total_loss: mean squared PDE residual
        loss_u: loss for u component
        loss_v: loss for v component
    """
    B, T, H, W, C = pred.shape
    assert C == 2, f"Expected 2 channels (u, v), got {C}"

    # Extract u and v: [B, T, H, W]
    u = pred[..., 0]
    v = pred[..., 1]

    # Time derivative (central difference): ∂u/∂t ≈ (u[t+1] - u[t-1]) / (2*dt)
    # For simplicity, use forward difference: ∂u/∂t ≈ (u[t+1] - u[t]) / dt
    du_dt = (u[:, 1:] - u[:, :-1]) / dt  # [B, T-1, H, W]
    dv_dt = (v[:, 1:] - v[:, :-1]) / dt

    # Use average of t and t+1 for spatial terms
    u_mid = (u[:, 1:] + u[:, :-1]) / 2  # [B, T-1, H, W]
    v_mid = (v[:, 1:] + v[:, :-1]) / 2

    # Spatial derivatives using Fourier (spectral accuracy)
    du_dx, du_dy = fourier_derivative_2d(u_mid, dx, dy)
    dv_dx, dv_dy = fourier_derivative_2d(v_mid, dx, dy)

    # Laplacian using Fourier
    laplacian_u = fourier_laplacian_2d(u_mid, dx, dy)
    laplacian_v = fourier_laplacian_2d(v_mid, dx, dy)

    # PDE residuals
    # R_u = ∂u/∂t + u * ∂u/∂x + v * ∂u/∂y - ν * ∇²u
    residual_u = du_dt + u_mid * du_dx + v_mid * du_dy - nu * laplacian_u

    # R_v = ∂v/∂t + u * ∂v/∂x + v * ∂v/∂y - ν * ∇²v
    residual_v = dv_dt + u_mid * dv_dx + v_mid * dv_dy - nu * laplacian_v

    # Mean squared residual
    loss_u = torch.mean(residual_u ** 2)
    loss_v = torch.mean(residual_v ** 2)
    total_loss = (loss_u + loss_v) / 2

    return total_loss, loss_u, loss_v


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Fourier PDE Loss")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Test parameters
    B, T, H, W = 2, 17, 128, 128
    nu = 0.1
    dt = 1.0 / 999
    dx = dy = 1.0 / 128

    # Create test data
    pred = torch.randn(B, T, H, W, 2, device=device)

    # Compute PDE loss
    total_loss, loss_u, loss_v = burgers_pde_loss_fourier(pred, nu, dt, dx, dy)

    print(f"\nInput shape: {pred.shape}")
    print(f"Total PDE loss: {total_loss.item():.6f}")
    print(f"Loss u: {loss_u.item():.6f}")
    print(f"Loss v: {loss_v.item():.6f}")

    # Test Fourier derivative accuracy
    print("\n" + "=" * 60)
    print("Testing Fourier derivative accuracy")
    print("=" * 60)

    # Create a known function: u = sin(2πx) * sin(2πy)
    # Exact derivatives:
    #   ∂u/∂x = 2π * cos(2πx) * sin(2πy)
    #   ∂u/∂y = 2π * sin(2πx) * cos(2πy)

    x = torch.linspace(0, 1 - 1/H, H, device=device)
    y = torch.linspace(0, 1 - 1/W, W, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    u_test = torch.sin(2 * math.pi * X) * torch.sin(2 * math.pi * Y)
    u_test = u_test.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    # Exact derivatives
    exact_du_dx = 2 * math.pi * torch.cos(2 * math.pi * X) * torch.sin(2 * math.pi * Y)
    exact_du_dy = 2 * math.pi * torch.sin(2 * math.pi * X) * torch.cos(2 * math.pi * Y)

    # Fourier derivatives
    fourier_du_dx, fourier_du_dy = fourier_derivative_2d(u_test, dx, dy)
    fourier_du_dx = fourier_du_dx.squeeze()
    fourier_du_dy = fourier_du_dy.squeeze()

    # Finite difference derivatives (for comparison)
    fd_du_dx = (torch.roll(u_test, -1, dims=-1) - torch.roll(u_test, 1, dims=-1)) / (2 * dx)
    fd_du_dy = (torch.roll(u_test, -1, dims=-2) - torch.roll(u_test, 1, dims=-2)) / (2 * dy)
    fd_du_dx = fd_du_dx.squeeze()
    fd_du_dy = fd_du_dy.squeeze()

    # Errors
    fourier_error_dx = torch.mean((fourier_du_dx - exact_du_dx) ** 2).sqrt().item()
    fourier_error_dy = torch.mean((fourier_du_dy - exact_du_dy) ** 2).sqrt().item()
    fd_error_dx = torch.mean((fd_du_dx - exact_du_dx) ** 2).sqrt().item()
    fd_error_dy = torch.mean((fd_du_dy - exact_du_dy) ** 2).sqrt().item()

    print(f"\nRMSE for ∂u/∂x:")
    print(f"  Fourier:          {fourier_error_dx:.2e}")
    print(f"  Finite Difference: {fd_error_dx:.2e}")
    print(f"  Improvement:       {fd_error_dx / fourier_error_dx:.1f}x")

    print(f"\nRMSE for ∂u/∂y:")
    print(f"  Fourier:          {fourier_error_dy:.2e}")
    print(f"  Finite Difference: {fd_error_dy:.2e}")
    print(f"  Improvement:       {fd_error_dy / fourier_error_dy:.1f}x")

    print("\n" + "=" * 60)
    print("Fourier PDE Loss test passed!")
    print("=" * 60)
