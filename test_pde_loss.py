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
    """Test Flow Mixing PDE loss on ground truth (all samples, all clips)."""
    from dataset_flow import FlowMixingDataset
    from pde_loss_flow import flow_mixing_pde_loss
    from tqdm import tqdm

    data_path = "./flow_mixing_vtmax0.3_0.5_res128_t1000_n150.h5"
    if not Path(data_path).exists():
        print(f"[Flow Mixing] Data file not found: {data_path}")
        return

    print("=" * 60)
    print("Testing Flow Mixing PDE Loss on Ground Truth")
    print("=" * 60)

    # Load dataset with ALL clips
    dataset = FlowMixingDataset(
        data_path=data_path,
        temporal_length=16,
        split='train',
        clips_per_sample=None,  # All clips
    )

    print(f"Total samples: {len(dataset.samples)}")
    print(f"Total clips: {len(dataset)}")
    print()

    # Accumulate stats
    total_pde_loss = 0.0
    total_time_loss = 0.0
    total_advection_loss = 0.0
    residual_min = float('inf')
    residual_max = float('-inf')
    residual_abs_sum = 0.0
    total_residual_count = 0

    for i in tqdm(range(len(dataset)), desc="Processing clips"):
        sample = dataset[i]

        # Ground truth data [T=17, H, W, 6]
        data = sample['data'].unsqueeze(0)
        vtmax = sample['vtmax'].item()

        # Boundaries
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

        total_pde_loss += pde_loss.item()
        total_time_loss += loss_time.item()
        total_advection_loss += loss_advection.item()
        residual_min = min(residual_min, residual.min().item())
        residual_max = max(residual_max, residual.max().item())
        residual_abs_sum += residual.abs().sum().item()
        total_residual_count += residual.numel()

    n = len(dataset)
    print()
    print("=" * 60)
    print("Results (All Clips)")
    print("=" * 60)
    print(f"Total clips evaluated: {n}")
    print(f"Average PDE Loss: {total_pde_loss / n:.6e}")
    print(f"Average Time Loss: {total_time_loss / n:.6e}")
    print(f"Average Advection Loss: {total_advection_loss / n:.6e}")
    print(f"Residual min: {residual_min:.6e}")
    print(f"Residual max: {residual_max:.6e}")
    print(f"Residual abs mean: {residual_abs_sum / total_residual_count:.6e}")
    print()
    print(f"Expected: PDE Loss ~0 (if implementation is correct)")
    print()


def test_burgers_pde():
    """Test Burgers PDE loss on ground truth (all samples, all clips)."""
    from dataset_burgers import BurgersDataset
    from pde_loss import burgers_pde_loss_upwind
    from tqdm import tqdm

    data_path = "./burgers2d_nu0.1_0.15_res128_t1000_n100.h5"
    if not Path(data_path).exists():
        print(f"[Burgers] Data file not found: {data_path}")
        return

    print("=" * 60)
    print("Testing Burgers PDE Loss on Ground Truth")
    print("=" * 60)

    # Load dataset with ALL clips
    dataset = BurgersDataset(
        data_path=data_path,
        temporal_length=16,
        split='train',
        clips_per_sample=None,  # All clips
    )

    print(f"Total samples: {len(dataset.samples)}")
    print(f"Total clips: {len(dataset)}")
    print()

    # Accumulate stats
    total_pde_loss = 0.0
    total_loss_u = 0.0
    total_loss_v = 0.0
    residual_u_min = float('inf')
    residual_u_max = float('-inf')
    residual_v_min = float('inf')
    residual_v_max = float('-inf')

    for i in tqdm(range(len(dataset)), desc="Processing clips"):
        sample = dataset[i]

        # Ground truth data [T=17, H, W, 6]
        data = sample['data'].unsqueeze(0)
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

        total_pde_loss += pde_loss.item()
        total_loss_u += loss_u.item()
        total_loss_v += loss_v.item()
        residual_u_min = min(residual_u_min, residual_u.min().item())
        residual_u_max = max(residual_u_max, residual_u.max().item())
        residual_v_min = min(residual_v_min, residual_v.min().item())
        residual_v_max = max(residual_v_max, residual_v.max().item())

    n = len(dataset)
    print()
    print("=" * 60)
    print("Results (All Clips)")
    print("=" * 60)
    print(f"Total clips evaluated: {n}")
    print(f"Average PDE Loss: {total_pde_loss / n:.6e}")
    print(f"Average Loss u: {total_loss_u / n:.6e}")
    print(f"Average Loss v: {total_loss_v / n:.6e}")
    print(f"Residual u: min={residual_u_min:.6e}, max={residual_u_max:.6e}")
    print(f"Residual v: min={residual_v_min:.6e}, max={residual_v_max:.6e}")
    print()
    print(f"Expected: PDE Loss ~0 (if implementation is correct)")
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
