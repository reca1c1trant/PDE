"""
KP-II 2D — GT PDE residual verification.

Verifies the exact 1-line-soliton solution against the KP-II equation.

PDE (original form):
    d/dx(u_t + 6u*u_x + u_xxx) + 3*u_yy = 0

PDE (expanded form — used for verification):
    u_xt + 6*(u_x)^2 + 6*u*u_xx + u_xxxx + 3*u_yy = 0

Numerical methods:
    - Time: 2nd-order central difference
    - Space: 4th-order central difference (periodic via torch.roll)
    - u_xxxx: computed as d2x(d2x(u)) via 4th-order stencil

NOTE: The line soliton is NOT truly periodic on [0,L]^2. The soliton
      crosses the domain boundary, causing large FD artifacts at the edges.
      We compute residuals both on the full domain and on the interior
      (skip_boundary pixels from each edge).

HDF5 format:
    scalar: (N, T, H, W, 1) — u=scalar[...,0]
    scalar_indices: [0]
    nu: (N,) — amplitude parameter a
    params_a, params_b, params_x0: per-sample parameters

Grid: 256x256, L=20, dx=dy=20/256
BCs: periodic (approximately — line soliton is not truly periodic)

Usage:
    python tools/test_gt_pde_kp2_2d.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import h5py
import numpy as np


def fd_dx_4th(f: torch.Tensor, dx: float) -> torch.Tensor:
    """4th-order central 1st x-derivative (periodic). f: [..., H, W], x=dim -2."""
    return (-torch.roll(f, -2, dims=-2) + 8 * torch.roll(f, -1, dims=-2)
            - 8 * torch.roll(f, 1, dims=-2) + torch.roll(f, 2, dims=-2)) / (12 * dx)


def fd_dy_4th(f: torch.Tensor, dy: float) -> torch.Tensor:
    """4th-order central 1st y-derivative (periodic). f: [..., H, W], y=dim -1."""
    return (-torch.roll(f, -2, dims=-1) + 8 * torch.roll(f, -1, dims=-1)
            - 8 * torch.roll(f, 1, dims=-1) + torch.roll(f, 2, dims=-1)) / (12 * dy)


def fd_d2x_4th(f: torch.Tensor, dx: float) -> torch.Tensor:
    """4th-order central 2nd x-derivative (periodic). f: [..., H, W], x=dim -2."""
    return (-torch.roll(f, -2, dims=-2) + 16 * torch.roll(f, -1, dims=-2)
            - 30 * f + 16 * torch.roll(f, 1, dims=-2)
            - torch.roll(f, 2, dims=-2)) / (12 * dx**2)


def fd_d2y_4th(f: torch.Tensor, dy: float) -> torch.Tensor:
    """4th-order central 2nd y-derivative (periodic). f: [..., H, W], y=dim -1."""
    return (-torch.roll(f, -2, dims=-1) + 16 * torch.roll(f, -1, dims=-1)
            - 30 * f + 16 * torch.roll(f, 1, dims=-1)
            - torch.roll(f, 2, dims=-1)) / (12 * dy**2)


def fd_d4x(f: torch.Tensor, dx: float) -> torch.Tensor:
    """
    4th x-derivative u_xxxx computed as d2x(d2x(u)).
    Uses 4th-order accurate stencils for each d2x, composed.
    """
    u_xx = fd_d2x_4th(f, dx)
    return fd_d2x_4th(u_xx, dx)


def main():
    data_path = "/scratch-share/SONG0304/finetune/kp2_2d.hdf5"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Physics parameters
    N_GRID = 256
    L = 20.0
    dx = L / N_GRID
    dy = dx
    dt = 0.01
    skip_boundary = 20  # skip this many pixels from each edge

    print(f"KP-II GT PDE Residual Verification")
    print(f"  Grid: {N_GRID}x{N_GRID}, dx={dx:.6f}, dy={dy:.6f}")
    print(f"  dt={dt}")
    print(f"  skip_boundary={skip_boundary}")
    print(f"  Device: {device}")
    print()

    # Load data
    with h5py.File(data_path, 'r') as f:
        scalar = f['scalar'][:]   # (N, T, H, W, 1)
        nu_arr = f['nu'][:]       # (N,)
        a_arr = f['params_a'][:]
        b_arr = f['params_b'][:]
        x0_arr = f['params_x0'][:]
        print(f"  scalar shape: {scalar.shape}")
        print(f"  N samples: {scalar.shape[0]}")

    N = scalar.shape[0]
    T = scalar.shape[1]
    T_res = T - 2  # interior timesteps for 2nd-order central time derivative

    # Accumulators
    all_kp2_rms_full = []
    all_kp2_rms_interior = []

    sb = skip_boundary
    per_t_sum_sq = torch.zeros(T_res, dtype=torch.float64)
    per_t_count = torch.zeros(T_res, dtype=torch.float64)

    for sample_idx in range(N):
        # Extract u: [T, H, W]
        u = torch.from_numpy(scalar[sample_idx, :, :, :, 0]).to(
            dtype=torch.float64, device=device)

        # Compute spatial derivatives at all timesteps
        u_x = fd_dx_4th(u, dx)  # [T, H, W]

        # u_xt: time derivative of u_x at interior times t=1..T-2
        u_xt = (u_x[2:] - u_x[:-2]) / (2 * dt)  # [T-2, H, W]

        # Spatial terms at interior times
        u_mid = u[1:-1]    # [T-2, H, W]
        u_x_mid = u_x[1:-1]

        # u_xx, u_xxxx, u_yy at interior times
        u_xx = fd_d2x_4th(u_mid, dx)
        u_xxxx = fd_d4x(u_mid, dx)
        u_yy = fd_d2y_4th(u_mid, dy)

        # KP-II residual (expanded form):
        # R = u_xt + 6*(u_x)^2 + 6*u*u_xx + u_xxxx + 3*u_yy
        R = u_xt + 6 * u_x_mid**2 + 6 * u_mid * u_xx + u_xxxx + 3 * u_yy

        # Full-domain RMS
        rms_full = torch.sqrt(torch.mean(R**2)).item()
        all_kp2_rms_full.append(rms_full)

        # Interior-only RMS (skip boundary pixels)
        R_int = R[:, sb:-sb, sb:-sb]
        rms_int = torch.sqrt(torch.mean(R_int**2)).item()
        all_kp2_rms_interior.append(rms_int)

        # Per-timestep interior RMS
        rms_per_t = torch.sqrt(torch.mean(R_int**2, dim=(-2, -1)))  # [T-2]
        per_t_sum_sq += rms_per_t.cpu()**2
        per_t_count += 1.0

        if sample_idx < 5 or sample_idx == N - 1:
            print(f"  Sample {sample_idx} (a={a_arr[sample_idx]:.3f}, "
                  f"b={b_arr[sample_idx]:.3f}): "
                  f"full RMS={rms_full:.6e}, interior RMS={rms_int:.6e}")

    # Summary
    print(f"\n{'='*70}")
    print(f"Summary over {N} samples:")
    print(f"{'='*70}")

    mean_full = np.mean(all_kp2_rms_full)
    mean_int = np.mean(all_kp2_rms_interior)
    std_int = np.std(all_kp2_rms_interior)

    print(f"  KP-II equation (full domain):     RMS mean={mean_full:.6e}")
    print(f"  KP-II equation (interior, sb={sb}): RMS mean={mean_int:.6e}, std={std_int:.6e}")
    print(f"    min={np.min(all_kp2_rms_interior):.6e}, max={np.max(all_kp2_rms_interior):.6e}")

    # Per-timestep average RMS (interior)
    rms_per_t_avg = torch.sqrt(per_t_sum_sq / per_t_count)
    print(f"\n  Per-timestep interior RMS (first 10):")
    for t in range(min(10, T_res)):
        print(f"    t={t+1}: {rms_per_t_avg[t].item():.6e}")

    # Save per-timestep scales
    os.makedirs("data/gt_scales", exist_ok=True)
    scales_dict = {
        'kp2_equation': rms_per_t_avg.float(),
    }
    torch.save(scales_dict, 'data/gt_scales/kp2_2d_per_t.pt')
    print(f"\n  Saved per-timestep scales to data/gt_scales/kp2_2d_per_t.pt")

    # Print recommended eq_scales for config
    print(f"\n  Recommended eq_scales for YAML config:")
    print(f"    kp2_equation: {mean_int:.6e}")


if __name__ == "__main__":
    main()
