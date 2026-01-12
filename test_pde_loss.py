"""
Test PDE Loss on Ground Truth Data.

Verify that ground truth data satisfies the PDE equation:
- Flow Mixing: ∂u/∂t + a·∂u/∂x + b·∂u/∂y = 0
- Burgers: ∂u/∂t + u·∂u/∂x + v·∂u/∂y = ν(∂²u/∂x² + ∂²u/∂y²)

If the PDE loss on ground truth is near zero, our implementation is correct.
"""

import torch
import numpy as np
from pathlib import Path


def test_flow_mixing_pde():
    """Test Flow Mixing PDE loss on ground truth."""
    from dataset_flow import FlowMixingDataset
    from pde_loss_flow import flow_mixing_pde_loss

    data_path = "./flow_mixing_vtmax0.3_0.5_res128_t1000_n150.h5"
    if not Path(data_path).exists():
        print(f"[Flow Mixing] Data file not found: {data_path}")
        return

    print("=" * 60)
    print("Testing Flow Mixing PDE Loss on Ground Truth")
    print("=" * 60)

    # Load dataset
    dataset = FlowMixingDataset(
        data_path=data_path,
        temporal_length=16,
        split='train',
        clips_per_sample=10,
    )

    # Test on several samples
    pde_losses = []
    for i in range(min(5, len(dataset))):
        sample = dataset[i]

        # Ground truth data [T=17, H, W, 6]
        data = sample['data'].unsqueeze(0)  # [1, 17, H, W, 6]
        vtmax = sample['vtmax'].item()

        # Boundaries [T=17, H, 1, 1] etc -> [1, T, H, 1, 1]
        bnd_left = sample['boundary_left'].unsqueeze(0)
        bnd_right = sample['boundary_right'].unsqueeze(0)
        bnd_bottom = sample['boundary_bottom'].unsqueeze(0)
        bnd_top = sample['boundary_top'].unsqueeze(0)

        # Extract u channel [1, 17, H, W, 1]
        gt_u = data[..., :1]

        # Compute PDE loss
        pde_loss, loss_time, loss_advection, residual = flow_mixing_pde_loss(
            pred=gt_u,
            boundary_left=bnd_left,
            boundary_right=bnd_right,
            boundary_bottom=bnd_bottom,
            boundary_top=bnd_top,
            vtmax=vtmax,
            dt=1/999,
        )

        pde_losses.append(pde_loss.item())

        print(f"Sample {i}: vtmax={vtmax:.4f}")
        print(f"  PDE Loss: {pde_loss.item():.6e}")
        print(f"  Time derivative loss: {loss_time.item():.6e}")
        print(f"  Advection loss: {loss_advection.item():.6e}")
        print(f"  Residual: min={residual.min().item():.6e}, max={residual.max().item():.6e}, mean={residual.abs().mean().item():.6e}")
        print()

    avg_loss = np.mean(pde_losses)
    print(f"Average PDE Loss on GT: {avg_loss:.6e}")
    print(f"Expected: ~0 (if correct)")
    print()


def test_burgers_pde():
    """Test Burgers PDE loss on ground truth."""
    from dataset_burgers import BurgersDataset
    from pde_loss import burgers_pde_loss_upwind

    data_path = "./burgers2d_nu0.1_0.15_res128_t1000_n100.h5"
    if not Path(data_path).exists():
        print(f"[Burgers] Data file not found: {data_path}")
        return

    print("=" * 60)
    print("Testing Burgers PDE Loss on Ground Truth")
    print("=" * 60)

    # Load dataset
    dataset = BurgersDataset(
        data_path=data_path,
        temporal_length=16,
        split='train',
        clips_per_sample=10,
    )

    # Test on several samples
    pde_losses = []
    for i in range(min(5, len(dataset))):
        sample = dataset[i]

        # Ground truth data [T=17, H, W, 6]
        data = sample['data'].unsqueeze(0)  # [1, 17, H, W, 6]
        nu = sample['nu'].item()

        # Boundaries
        bnd_left = sample['boundary_left'].unsqueeze(0)
        bnd_right = sample['boundary_right'].unsqueeze(0)
        bnd_bottom = sample['boundary_bottom'].unsqueeze(0)
        bnd_top = sample['boundary_top'].unsqueeze(0)

        # Extract u, v channels [1, 17, H, W, 2]
        gt_uv = data[..., :2]

        # Compute PDE loss
        pde_loss, loss_u, loss_v, residual_u, residual_v = burgers_pde_loss_upwind(
            pred=gt_uv,
            boundary_left=bnd_left,
            boundary_right=bnd_right,
            boundary_bottom=bnd_bottom,
            boundary_top=bnd_top,
            nu=nu,
            dt=1/999,
        )

        pde_losses.append(pde_loss.item())

        print(f"Sample {i}: nu={nu:.4f}")
        print(f"  PDE Loss: {pde_loss.item():.6e}")
        print(f"  Loss u: {loss_u.item():.6e}")
        print(f"  Loss v: {loss_v.item():.6e}")
        print(f"  Residual u: min={residual_u.min().item():.6e}, max={residual_u.max().item():.6e}")
        print(f"  Residual v: min={residual_v.min().item():.6e}, max={residual_v.max().item():.6e}")
        print()

    avg_loss = np.mean(pde_losses)
    print(f"Average PDE Loss on GT: {avg_loss:.6e}")
    print(f"Expected: ~0 (if correct)")
    print()


def test_flow_mixing_analytical():
    """
    Test Flow Mixing PDE with analytical solution directly.
    No dataset needed - compute u analytically and verify PDE.
    """
    from pde_loss_flow import compute_flow_coefficients

    print("=" * 60)
    print("Testing Flow Mixing PDE with Analytical Solution")
    print("=" * 60)

    device = torch.device('cpu')
    H, W = 128, 128
    T = 17
    vtmax = 0.4
    dt = 1 / 999

    # Grid
    x = torch.linspace(1/256, 255/256, W, device=device)
    y = torch.linspace(1/256, 255/256, H, device=device)
    t = torch.linspace(0, (T-1) * dt, T, device=device)

    Y, X = torch.meshgrid(y, x, indexing='ij')

    # Analytical solution
    r = torch.sqrt(X**2 + Y**2)
    r_safe = torch.clamp(r, min=1e-10)
    v_t = (1.0 / torch.cosh(r_safe))**2 * torch.tanh(r_safe)
    omega = v_t / (r_safe * vtmax)

    # Compute u at all timesteps [T, H, W]
    u_gt = torch.zeros(T, H, W)
    for k in range(T):
        u_gt[k] = -torch.tanh((Y / 2) * torch.cos(omega * t[k]) - (X / 2) * torch.sin(omega * t[k]))

    # Compute coefficients
    a, b = compute_flow_coefficients(H, W, vtmax, device, torch.float32)

    # Compute derivatives using finite differences
    dx = 1.0 / W
    dy = 1.0 / H

    # du/dt using backward difference (from t=1)
    du_dt = (u_gt[1:] - u_gt[:-1]) / dt  # [T-1, H, W]

    # du/dx using central difference (interior points)
    du_dx = torch.zeros(T, H, W)
    du_dx[:, :, 1:-1] = (u_gt[:, :, 2:] - u_gt[:, :, :-2]) / (2 * dx)

    # du/dy using central difference (interior points)
    du_dy = torch.zeros(T, H, W)
    du_dy[:, 1:-1, :] = (u_gt[:, 2:, :] - u_gt[:, :-2, :]) / (2 * dy)

    # PDE residual: du/dt + a*du/dx + b*du/dy = 0
    # Use du_dx and du_dy at t=1 to T-1
    advection = a * du_dx[1:] + b * du_dy[1:]
    residual = du_dt + advection

    # Only consider interior points (exclude boundaries)
    residual_interior = residual[:, 2:-2, 2:-2]

    print(f"Grid: {H}x{W}, T={T}, vtmax={vtmax}")
    print(f"dt={dt:.6f}, dx={dx:.6f}, dy={dy:.6f}")
    print()
    print(f"du/dt: min={du_dt.min():.4f}, max={du_dt.max():.4f}")
    print(f"a*du/dx + b*du/dy: min={advection.min():.4f}, max={advection.max():.4f}")
    print()
    print(f"PDE Residual (interior):")
    print(f"  min: {residual_interior.min().item():.6e}")
    print(f"  max: {residual_interior.max().item():.6e}")
    print(f"  mean: {residual_interior.mean().item():.6e}")
    print(f"  abs mean: {residual_interior.abs().mean().item():.6e}")
    print(f"  MSE: {(residual_interior**2).mean().item():.6e}")
    print()
    print(f"Expected: Residual ~0 (limited by finite difference truncation error)")


if __name__ == "__main__":
    # Test analytical first (no dataset needed)
    test_flow_mixing_analytical()
    print("\n" + "=" * 60 + "\n")

    # Test on actual dataset
    test_flow_mixing_pde()
    test_burgers_pde()
