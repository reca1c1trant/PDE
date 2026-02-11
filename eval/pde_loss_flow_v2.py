"""
2D Flow Mixing PDE residual loss - V2 中心差分版本

与 train_PINN_transient.py 相同的计算方式：
- 时间导数: 后向差分
- 空间导数: 一阶中心差分 (uE - uW) / (2dx)

PDE: ∂u/∂t + a·∂u/∂x + b·∂u/∂y = 0

Author: Ziye
Date: January 2025
"""

import torch
from typing import Tuple


def compute_flow_coefficients(
    H: int,
    W: int,
    vtmax: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advection coefficients a(x,y) and b(x,y) for flow mixing.

    a(x,y) = -v_t/v_tmax · y/r = -ω·y
    b(x,y) = +v_t/v_tmax · x/r = +ω·x
    """
    # Interior grid points
    x = torch.linspace(1/256, 255/256, W, device=device, dtype=dtype)
    y = torch.linspace(1/256, 255/256, H, device=device, dtype=dtype)

    # Create meshgrid [H, W]
    Y, X = torch.meshgrid(y, x, indexing='ij')

    # Compute r = sqrt(x² + y²)
    r = torch.sqrt(X**2 + Y**2)
    r_safe = torch.clamp(r, min=1e-10)

    # v_t = sech²(r) * tanh(r)
    # Clamp r to avoid overflow in cosh for large r
    r_clamped = torch.clamp(r_safe, max=20.0)  # cosh(20) ~ 2.4e8, safe
    v_t = (1.0 / torch.cosh(r_clamped))**2 * torch.tanh(r_clamped)

    # omega = v_t / (r * vtmax)
    omega = v_t / (r_safe * vtmax + 1e-10)

    # a = -omega * y, b = omega * x
    a = -omega * Y
    b = omega * X

    return a, b


def pad_with_boundaries_1x(
    interior: torch.Tensor,
    boundary_left: torch.Tensor,
    boundary_right: torch.Tensor,
    boundary_bottom: torch.Tensor,
    boundary_top: torch.Tensor,
) -> torch.Tensor:
    """
    Pad interior grid with 1 layer of ghost cells (for central difference).

    Ghost cell extrapolation (linear):
    - ghost = 2*boundary - interior[0]

    Parameters:
    -----------
    interior : torch.Tensor [B, T, H, W]
    boundary_* : torch.Tensor

    Returns:
    --------
    padded : torch.Tensor [B, T, H+2, W+2]
    """
    B, T, H, W = interior.shape

    # Squeeze boundaries
    bnd_left = boundary_left.squeeze(-1)      # [B, T, H]
    bnd_right = boundary_right.squeeze(-1)    # [B, T, H]
    bnd_bottom = boundary_bottom.squeeze(-2)  # [B, T, W]
    bnd_top = boundary_top.squeeze(-2)        # [B, T, W]

    # Ghost cells (1 layer)
    ghost_left = 2 * bnd_left - interior[..., 0]       # [B, T, H]
    ghost_right = 2 * bnd_right - interior[..., -1]    # [B, T, H]
    ghost_bottom = 2 * bnd_bottom - interior[..., 0, :]  # [B, T, W]
    ghost_top = 2 * bnd_top - interior[..., -1, :]       # [B, T, W]

    # Build padded tensor [B, T, H+2, W+2]
    padded = torch.zeros(B, T, H + 2, W + 2, device=interior.device, dtype=interior.dtype)

    # Fill interior
    padded[:, :, 1:H+1, 1:W+1] = interior

    # Fill ghost cells
    padded[:, :, 1:H+1, 0] = ghost_left
    padded[:, :, 1:H+1, W+1] = ghost_right
    padded[:, :, 0, 1:W+1] = ghost_bottom
    padded[:, :, H+1, 1:W+1] = ghost_top

    # Corners (average)
    padded[:, :, 0, 0] = (ghost_left[:, :, 0] + ghost_bottom[:, :, 0]) / 2
    padded[:, :, 0, W+1] = (ghost_right[:, :, 0] + ghost_bottom[:, :, -1]) / 2
    padded[:, :, H+1, 0] = (ghost_left[:, :, -1] + ghost_top[:, :, 0]) / 2
    padded[:, :, H+1, W+1] = (ghost_right[:, :, -1] + ghost_top[:, :, -1]) / 2

    return padded


def central_derivative_x(
    u_padded: torch.Tensor,
    H: int,
    W: int,
    dx: float,
) -> torch.Tensor:
    """
    Compute x-derivative using central difference (same as train_PINN_transient.py).

    Formula: u_x = (u_E - u_W) / (2*dx)
    Equivalent to: uc_e = 0.5*(uE + uC), uc_w = 0.5*(uW + uC), u_x = (uc_e - uc_w)/dx

    Parameters:
    -----------
    u_padded : torch.Tensor [B, T, H+2, W+2]
    H, W : int
    dx : float

    Returns:
    --------
    du_dx : torch.Tensor [B, T, H, W]
    """
    # Extract neighbors
    uC = u_padded[:, :, 1:H+1, 1:W+1]  # Center
    uE = u_padded[:, :, 1:H+1, 2:W+2]  # East (right)
    uW = u_padded[:, :, 1:H+1, 0:W]    # West (left)

    # Central difference: (uE - uW) / (2*dx)
    # Equivalent to train_PINN_transient: (uc_e - uc_w)/dx where uc_e = 0.5*(uE+uC)
    du_dx = (uE - uW) / (2 * dx)

    return du_dx


def central_derivative_y(
    u_padded: torch.Tensor,
    H: int,
    W: int,
    dy: float,
) -> torch.Tensor:
    """
    Compute y-derivative using central difference.

    Formula: u_y = (u_N - u_S) / (2*dy)

    Parameters:
    -----------
    u_padded : torch.Tensor [B, T, H+2, W+2]
    H, W : int
    dy : float

    Returns:
    --------
    du_dy : torch.Tensor [B, T, H, W]
    """
    # Extract neighbors
    uN = u_padded[:, :, 2:H+2, 1:W+1]  # North (top)
    uS = u_padded[:, :, 0:H, 1:W+1]    # South (bottom)

    # Central difference
    du_dy = (uN - uS) / (2 * dy)

    return du_dy


def flow_mixing_pde_loss_v2(
    pred: torch.Tensor,
    boundary_left: torch.Tensor,
    boundary_right: torch.Tensor,
    boundary_bottom: torch.Tensor,
    boundary_top: torch.Tensor,
    vtmax: float,
    dt: float = 1/999,
    Lx: float = 1.0,
    Ly: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute PDE residual loss using central difference (train_PINN_transient style).

    PDE: ∂u/∂t + a·∂u/∂x + b·∂u/∂y = 0

    Parameters:
    -----------
    pred : torch.Tensor [B, T, H, W, 1]
        Predicted field (17 timesteps including t=0)
    boundary_* : torch.Tensor
        Boundary values
    vtmax : float
        Maximum tangential velocity parameter
    dt : float
        Time step
    Lx, Ly : float
        Domain size

    Returns:
    --------
    pde_loss, loss_time, loss_advection, residual
    """
    B, T, H, W, C = pred.shape
    device = pred.device
    dtype = pred.dtype

    # Grid spacing
    dx = Lx / W
    dy = Ly / H

    # Extract u channel [B, T, H, W]
    u = pred[..., 0]

    # Reshape boundaries
    bnd_left = boundary_left[..., 0]      # [B, T, H, 1]
    bnd_right = boundary_right[..., 0]    # [B, T, H, 1]
    bnd_bottom = boundary_bottom[..., 0]  # [B, T, 1, W]
    bnd_top = boundary_top[..., 0]        # [B, T, 1, W]

    # Compute flow coefficients [H, W]
    a, b = compute_flow_coefficients(H, W, vtmax, device, dtype)

    # Pad with 1-layer ghost cells [B, T, H+2, W+2]
    u_padded = pad_with_boundaries_1x(u, bnd_left, bnd_right, bnd_bottom, bnd_top)

    # Compute spatial derivatives using central difference [B, T, H, W]
    du_dx = central_derivative_x(u_padded, H, W, dx)
    du_dy = central_derivative_y(u_padded, H, W, dy)

    # Time derivative (backward difference)
    # u_t ≈ (u[t] - u[t-1]) / dt
    du_dt = (u[:, 1:] - u[:, :-1]) / dt  # [B, T-1, H, W]

    # Advection terms at t=1 to T-1
    advection = a * du_dx[:, 1:] + b * du_dy[:, 1:]  # [B, T-1, H, W]

    # PDE residual: du/dt + a*du/dx + b*du/dy = 0
    residual = du_dt + advection  # [B, T-1, H, W]

    # Losses
    loss_time = torch.mean(du_dt**2)
    loss_advection = torch.mean(advection**2)
    pde_loss = torch.mean(residual**2)

    return pde_loss, loss_time, loss_advection, residual


if __name__ == "__main__":
    """Test PDE loss computation."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    B, T, H, W = 2, 17, 128, 128
    vtmax = 0.4

    # Create dummy data
    pred = torch.randn(B, T, H, W, 1, device=device)
    bnd_left = torch.randn(B, T, H, 1, 1, device=device)
    bnd_right = torch.randn(B, T, H, 1, 1, device=device)
    bnd_bottom = torch.randn(B, T, 1, W, 1, device=device)
    bnd_top = torch.randn(B, T, 1, W, 1, device=device)

    pde_loss, loss_time, loss_advection, residual = flow_mixing_pde_loss_v2(
        pred, bnd_left, bnd_right, bnd_bottom, bnd_top, vtmax
    )

    print(f"PDE Loss (V2 central diff): {pde_loss.item():.6f}")
    print(f"Time derivative loss: {loss_time.item():.6f}")
    print(f"Advection loss: {loss_advection.item():.6f}")
    print(f"Residual shape: {residual.shape}")

    print("\nPDE loss V2 test passed!")
