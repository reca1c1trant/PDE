"""
2D Flow Mixing PDE residual loss.

PDE: ∂u/∂t + a·∂u/∂x + b·∂u/∂y = 0

where:
    a(x,y) = -v_t/v_tmax · y/r
    b(x,y) = v_t/v_tmax · x/r
    v_t = sech²(r)·tanh(r)
    r = √(x² + y²)

Derivative computation:
- Time derivative: backward difference (from index 1)
- Spatial derivatives: upwind scheme (based on a, b signs)

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

    The interior grid spans x,y ∈ [1/256, 255/256] with H×W points.

    Parameters:
    -----------
    H, W : int
        Grid dimensions
    vtmax : float
        Maximum tangential velocity parameter
    device : torch.device
        Device to create tensors on
    dtype : torch.dtype
        Data type

    Returns:
    --------
    a : torch.Tensor [H, W]
        Coefficient a(x,y) = -v_t/v_tmax · y/r
    b : torch.Tensor [H, W]
        Coefficient b(x,y) = v_t/v_tmax · x/r
    """
    # Interior grid points (matching dataset generation)
    x = torch.linspace(1/256, 255/256, W, device=device, dtype=dtype)
    y = torch.linspace(1/256, 255/256, H, device=device, dtype=dtype)

    # Create meshgrid [H, W]
    Y, X = torch.meshgrid(y, x, indexing='ij')

    # Compute r = sqrt(x² + y²), avoid division by zero
    r = torch.sqrt(X**2 + Y**2)
    r_safe = torch.clamp(r, min=1e-10)

    # v_t = sech²(r) * tanh(r)
    v_t = (1.0 / torch.cosh(r_safe))**2 * torch.tanh(r_safe)

    # omega = v_t / (r * vtmax)
    omega = v_t / (r_safe * vtmax)

    # a = -v_t/vtmax * y/r = -omega * y
    # b = v_t/vtmax * x/r = omega * x
    a = -omega * Y
    b = omega * X

    return a, b


def pad_with_boundaries(
    interior: torch.Tensor,
    boundary_left: torch.Tensor,
    boundary_right: torch.Tensor,
    boundary_bottom: torch.Tensor,
    boundary_top: torch.Tensor,
) -> torch.Tensor:
    """
    Pad interior grid with boundary values using ghost cell extrapolation.

    Parameters:
    -----------
    interior : torch.Tensor [B, T, H, W]
    boundary_left : torch.Tensor [B, T, H, 1]
    boundary_right : torch.Tensor [B, T, H, 1]
    boundary_bottom : torch.Tensor [B, T, 1, W]
    boundary_top : torch.Tensor [B, T, 1, W]

    Returns:
    --------
    padded : torch.Tensor [B, T, H+2, W+2]
    """
    B, T, H, W = interior.shape

    # Ghost cells via linear extrapolation
    ghost_left = 2 * boundary_left.squeeze(-1) - interior[..., 0]      # [B, T, H]
    ghost_right = 2 * boundary_right.squeeze(-1) - interior[..., -1]   # [B, T, H]
    ghost_bottom = 2 * boundary_bottom.squeeze(-2) - interior[..., 0, :]  # [B, T, W]
    ghost_top = 2 * boundary_top.squeeze(-2) - interior[..., -1, :]       # [B, T, W]

    # Corner points
    corner_bl = (ghost_left[:, :, 0:1] + ghost_bottom[:, :, 0:1]) / 2
    corner_br = (ghost_right[:, :, 0:1] + ghost_bottom[:, :, -1:]) / 2
    corner_tl = (ghost_left[:, :, -1:] + ghost_top[:, :, 0:1]) / 2
    corner_tr = (ghost_right[:, :, -1:] + ghost_top[:, :, -1:]) / 2

    # Build middle rows [B, T, H, W+2]
    ghost_left_col = ghost_left.unsqueeze(-1)
    ghost_right_col = ghost_right.unsqueeze(-1)
    middle_rows = torch.cat([ghost_left_col, interior, ghost_right_col], dim=-1)

    # Bottom and top rows [B, T, 1, W+2]
    bottom_row = torch.cat([corner_bl.unsqueeze(-1), ghost_bottom.unsqueeze(-2), corner_br.unsqueeze(-1)], dim=-1)
    top_row = torch.cat([corner_tl.unsqueeze(-1), ghost_top.unsqueeze(-2), corner_tr.unsqueeze(-1)], dim=-1)

    # Assemble [B, T, H+2, W+2]
    padded = torch.cat([bottom_row, middle_rows, top_row], dim=-2)

    return padded


def upwind_derivative_x(
    u_padded: torch.Tensor,
    velocity: torch.Tensor,
    H: int,
    W: int,
    dx: float,
) -> torch.Tensor:
    """
    Compute x-derivative using upwind scheme.

    Parameters:
    -----------
    u_padded : torch.Tensor [B, T, H+2, W+2]
    velocity : torch.Tensor [H, W] - coefficient a(x,y)
    H, W : int
    dx : float

    Returns:
    --------
    du_dx : torch.Tensor [B, T, H, W]
    """
    u_center = u_padded[:, :, 1:H+1, 1:W+1]
    u_left = u_padded[:, :, 1:H+1, 0:W]
    u_right = u_padded[:, :, 1:H+1, 2:W+2]

    # Backward difference where velocity > 0
    du_backward = (u_center - u_left) / dx
    # Forward difference where velocity < 0
    du_forward = (u_right - u_center) / dx

    # Upwind selection
    du_dx = torch.where(velocity > 0, du_backward, du_forward)

    return du_dx


def upwind_derivative_y(
    u_padded: torch.Tensor,
    velocity: torch.Tensor,
    H: int,
    W: int,
    dy: float,
) -> torch.Tensor:
    """
    Compute y-derivative using upwind scheme.

    Parameters:
    -----------
    u_padded : torch.Tensor [B, T, H+2, W+2]
    velocity : torch.Tensor [H, W] - coefficient b(x,y)
    H, W : int
    dy : float

    Returns:
    --------
    du_dy : torch.Tensor [B, T, H, W]
    """
    u_center = u_padded[:, :, 1:H+1, 1:W+1]
    u_bottom = u_padded[:, :, 0:H, 1:W+1]
    u_top = u_padded[:, :, 2:H+2, 1:W+1]

    # Backward difference where velocity > 0
    du_backward = (u_center - u_bottom) / dy
    # Forward difference where velocity < 0
    du_forward = (u_top - u_center) / dy

    du_dy = torch.where(velocity > 0, du_backward, du_forward)

    return du_dy


def flow_mixing_pde_loss(
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
    Compute PDE residual loss for 2D Flow Mixing equation.

    PDE: ∂u/∂t + a·∂u/∂x + b·∂u/∂y = 0

    Parameters:
    -----------
    pred : torch.Tensor [B, T, H, W, 1]
        Predicted field (17 timesteps including t=0)
    boundary_left : torch.Tensor [B, T, H, 1, 1]
        Left boundary values
    boundary_right : torch.Tensor [B, T, H, 1, 1]
        Right boundary values
    boundary_bottom : torch.Tensor [B, T, 1, W, 1]
        Bottom boundary values
    boundary_top : torch.Tensor [B, T, 1, W, 1]
        Top boundary values
    vtmax : float
        Maximum tangential velocity parameter
    dt : float
        Time step
    Lx, Ly : float
        Domain size

    Returns:
    --------
    pde_loss : torch.Tensor
        Total PDE residual loss
    loss_time : torch.Tensor
        Time derivative contribution
    loss_advection : torch.Tensor
        Advection (a*du/dx + b*du/dy) contribution
    residual : torch.Tensor [B, T-1, H, W]
        Full residual field (for visualization)
    """
    B, T, H, W, C = pred.shape
    device = pred.device
    dtype = pred.dtype

    # Grid spacing
    dx = Lx / W
    dy = Ly / H

    # Extract u channel [B, T, H, W]
    u = pred[..., 0]

    # Reshape boundaries [B, T, H, 1], [B, T, 1, W]
    bnd_left = boundary_left[..., 0]    # [B, T, H, 1]
    bnd_right = boundary_right[..., 0]  # [B, T, H, 1]
    bnd_bottom = boundary_bottom[..., 0]  # [B, T, 1, W]
    bnd_top = boundary_top[..., 0]      # [B, T, 1, W]

    # Compute flow coefficients a(x,y), b(x,y) [H, W]
    a, b = compute_flow_coefficients(H, W, vtmax, device, dtype)

    # Pad with boundaries [B, T, H+2, W+2]
    u_padded = pad_with_boundaries(u, bnd_left, bnd_right, bnd_bottom, bnd_top)

    # Compute spatial derivatives using upwind scheme [B, T, H, W]
    du_dx = upwind_derivative_x(u_padded, a, H, W, dx)
    du_dy = upwind_derivative_y(u_padded, b, H, W, dy)

    # Compute time derivative (backward difference, from t=1 to T-1)
    # u_t ≈ (u[t] - u[t-1]) / dt
    du_dt = (u[:, 1:] - u[:, :-1]) / dt  # [B, T-1, H, W]

    # Advection terms at t=1 to T-1 (use values at current time for upwind)
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

    pde_loss, loss_time, loss_advection, residual = flow_mixing_pde_loss(
        pred, bnd_left, bnd_right, bnd_bottom, bnd_top, vtmax
    )

    print(f"PDE Loss: {pde_loss.item():.6f}")
    print(f"Time derivative loss: {loss_time.item():.6f}")
    print(f"Advection loss: {loss_advection.item():.6f}")
    print(f"Residual shape: {residual.shape}")

    # Test flow coefficients
    a, b = compute_flow_coefficients(H, W, vtmax, device)
    print(f"\nCoefficient a: min={a.min():.4f}, max={a.max():.4f}")
    print(f"Coefficient b: min={b.min():.4f}, max={b.max():.4f}")

    print("\nPDE loss test passed!")
