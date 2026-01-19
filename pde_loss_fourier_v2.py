"""
Fourier-based PDE Loss V2 for Burgers Equation.

V2 improvements:
1. Normalized PDE residual (dimensionless)
2. Boundary loss (enforce boundary conditions)

Reference: PINO (Li et al., arXiv:2111.03794)
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

    u = u.float()

    # Wavenumbers
    kx = torch.fft.fftfreq(H, d=dx, device=device) * 2 * math.pi
    ky = torch.fft.rfftfreq(W, d=dy, device=device) * 2 * math.pi

    kx = kx.view(1, 1, H, 1)
    ky = ky.view(1, 1, 1, -1)

    # FFT
    u_hat = torch.fft.rfft2(u)

    # Derivative in Fourier space
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

    kx = torch.fft.fftfreq(H, d=dx, device=device) * 2 * math.pi
    ky = torch.fft.rfftfreq(W, d=dy, device=device) * 2 * math.pi

    kx_sq = kx.view(1, 1, H, 1) ** 2
    ky_sq = ky.view(1, 1, 1, -1) ** 2
    k_sq = kx_sq + ky_sq

    u_hat = torch.fft.rfft2(u)
    laplacian_hat = -k_sq * u_hat
    laplacian = torch.fft.irfft2(laplacian_hat, s=(H, W))

    return laplacian.to(dtype)


def burgers_pde_loss_fourier_v2(
    pred: torch.Tensor,
    nu: float,
    dt: float,
    dx: float = 1.0 / 128,
    dy: float = 1.0 / 128,
    normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute NORMALIZED Burgers PDE residual loss using Fourier derivatives.

    2D Burgers equation:
        ∂u/∂t + u·∂u/∂x + v·∂u/∂y = ν·∇²u
        ∂v/∂t + u·∂v/∂x + v·∂v/∂y = ν·∇²v

    Normalization: Divide residual by characteristic scale to make it dimensionless.

    Args:
        pred: [B, T, H, W, 2] predicted velocity field (u, v)
        nu: viscosity coefficient
        dt: time step
        dx: grid spacing in x
        dy: grid spacing in y
        normalize: whether to normalize the residual

    Returns:
        total_loss: normalized mean squared PDE residual
        loss_u: loss for u component
        loss_v: loss for v component
    """
    B, T, H, W, C = pred.shape
    assert C == 2, f"Expected 2 channels (u, v), got {C}"

    u = pred[..., 0]  # [B, T, H, W]
    v = pred[..., 1]

    # Time derivative: (u[t+1] - u[t]) / dt
    du_dt = (u[:, 1:] - u[:, :-1]) / dt  # [B, T-1, H, W]
    dv_dt = (v[:, 1:] - v[:, :-1]) / dt

    # Midpoint for spatial terms
    u_mid = (u[:, 1:] + u[:, :-1]) / 2  # [B, T-1, H, W]
    v_mid = (v[:, 1:] + v[:, :-1]) / 2

    # Spatial derivatives (Fourier, spectral accuracy)
    du_dx, du_dy = fourier_derivative_2d(u_mid, dx, dy)
    dv_dx, dv_dy = fourier_derivative_2d(v_mid, dx, dy)

    # Laplacian (Fourier)
    laplacian_u = fourier_laplacian_2d(u_mid, dx, dy)
    laplacian_v = fourier_laplacian_2d(v_mid, dx, dy)

    # PDE residuals
    residual_u = du_dt + u_mid * du_dx + v_mid * du_dy - nu * laplacian_u
    residual_v = dv_dt + u_mid * dv_dx + v_mid * dv_dy - nu * laplacian_v

    if normalize:
        # Normalize by characteristic scale: |du/dt| ~ U/T
        # Use RMS of time derivative as reference
        scale_u = torch.sqrt(torch.mean(du_dt ** 2) + 1e-8)
        scale_v = torch.sqrt(torch.mean(dv_dt ** 2) + 1e-8)

        residual_u = residual_u / scale_u
        residual_v = residual_v / scale_v

    # Mean squared residual
    loss_u = torch.mean(residual_u ** 2)
    loss_v = torch.mean(residual_v ** 2)
    total_loss = (loss_u + loss_v) / 2

    return total_loss, loss_u, loss_v


def boundary_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Compute boundary loss (enforce boundary conditions).

    Boundary definition (exclude corners):
        Left:   pred[:, :, 1:-1, 0, :]
        Right:  pred[:, :, 1:-1, -1, :]
        Bottom: pred[:, :, 0, 1:-1, :]
        Top:    pred[:, :, -1, 1:-1, :]

    Args:
        pred: [B, T, H, W, C] predicted field
        target: [B, T, H, W, C] ground truth field

    Returns:
        loss: MSE on boundaries
    """
    # Left boundary: x=0, y from 1 to H-2 (exclude corners)
    pred_left = pred[:, :, 1:-1, 0, :]      # [B, T, H-2, C]
    target_left = target[:, :, 1:-1, 0, :]

    # Right boundary: x=W-1, y from 1 to H-2
    pred_right = pred[:, :, 1:-1, -1, :]    # [B, T, H-2, C]
    target_right = target[:, :, 1:-1, -1, :]

    # Bottom boundary: y=0, x from 1 to W-2
    pred_bottom = pred[:, :, 0, 1:-1, :]    # [B, T, W-2, C]
    target_bottom = target[:, :, 0, 1:-1, :]

    # Top boundary: y=H-1, x from 1 to W-2
    pred_top = pred[:, :, -1, 1:-1, :]      # [B, T, W-2, C]
    target_top = target[:, :, -1, 1:-1, :]

    # MSE for each boundary
    loss_left = F.mse_loss(pred_left, target_left)
    loss_right = F.mse_loss(pred_right, target_right)
    loss_bottom = F.mse_loss(pred_bottom, target_bottom)
    loss_top = F.mse_loss(pred_top, target_top)

    # Average
    total_loss = (loss_left + loss_right + loss_bottom + loss_top) / 4

    return total_loss


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Fourier PDE Loss V2 (Normalized + Boundary)")
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
    target = torch.randn(B, T, H, W, 2, device=device)

    # Test normalized PDE loss
    print("\n--- Normalized PDE Loss ---")
    total_loss, loss_u, loss_v = burgers_pde_loss_fourier_v2(
        pred, nu, dt, dx, dy, normalize=True
    )
    print(f"Total PDE loss (normalized): {total_loss.item():.6f}")
    print(f"Loss u: {loss_u.item():.6f}")
    print(f"Loss v: {loss_v.item():.6f}")

    # Test non-normalized (for comparison)
    print("\n--- Non-normalized PDE Loss ---")
    total_loss_raw, _, _ = burgers_pde_loss_fourier_v2(
        pred, nu, dt, dx, dy, normalize=False
    )
    print(f"Total PDE loss (raw): {total_loss_raw.item():.6f}")

    # Test boundary loss
    print("\n--- Boundary Loss ---")
    b_loss = boundary_loss(pred, target)
    print(f"Boundary loss: {b_loss.item():.6f}")

    # Verify boundary shapes
    print("\n--- Boundary Shapes ---")
    print(f"Left:   pred[:, :, 1:-1, 0, :].shape = {pred[:, :, 1:-1, 0, :].shape}")
    print(f"Right:  pred[:, :, 1:-1, -1, :].shape = {pred[:, :, 1:-1, -1, :].shape}")
    print(f"Bottom: pred[:, :, 0, 1:-1, :].shape = {pred[:, :, 0, 1:-1, :].shape}")
    print(f"Top:    pred[:, :, -1, 1:-1, :].shape = {pred[:, :, -1, 1:-1, :].shape}")

    print("\n" + "=" * 60)
    print("Fourier PDE Loss V2 test passed!")
    print("=" * 60)
