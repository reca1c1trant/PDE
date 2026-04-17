"""
GT PDE Residual Verification for APEBench 1D Kuramoto-Sivashinsky.

APEBench KS equation (normalized domain [0,1], dt=1):
    u_t = alpha_2 * u_xx + alpha_4 * u_xxxx + beta_2 * 1/2 * (u_x)^2

Difficulty parameters from APEBench:
    gammas = (0, 0, -1.2, 0, -15.0)
    deltas = (0, 0, -6.0)

Normalized coefficients (N=160, D=1):
    alpha_2 = gamma_2 / (N^2 * 2 * D) = -1.2 / 51200 = -2.34375e-5
    alpha_4 = gamma_4 / (N^4 * 8 * D) = -15 / 5242880000 = -2.86102e-9
    beta_2  = delta_2 / (M * N^2 * D) = -6 / 25600 = -2.34375e-4

Spatial discretization:
    - u_x:    4th-order central FD
    - u_xx:   4th-order central FD
    - u_xxxx: computed as Lap(Lap(u)), i.e., d2x(d2x(u)), 2nd-order each
              or direct 5-point stencil (2nd order)

Time discretization:
    - u_t: 2nd-order central

Uses float64, periodic BC via torch.roll.

Usage:
    python tools/test_gt_pde_ks_1d.py
"""

import torch
import h5py
import numpy as np
import os

DATA_PATH = "/scratch-share/SONG0304/finetune/apebench_1d_ks.hdf5"

# Normalized domain parameters
NX = 160
LX = 1.0
DT = 1.0
dx = LX / NX

# Normalized coefficients
ALPHA_2 = -2.34375e-5    # negative diffusion
ALPHA_4 = -2.8610229492e-9  # hyper-diffusion
BETA_2 = -2.34375e-4     # gradient norm scale


def fd_dx_4th(f: torch.Tensor) -> torch.Tensor:
    """4th-order central d/dx (periodic, dim=-1)."""
    return (
        -torch.roll(f, -2, dims=-1) + 8 * torch.roll(f, -1, dims=-1)
        - 8 * torch.roll(f, 1, dims=-1) + torch.roll(f, 2, dims=-1)
    ) / (12 * dx)


def fd_d2x_4th(f: torch.Tensor) -> torch.Tensor:
    """4th-order central d^2/dx^2 (periodic, dim=-1)."""
    return (
        -torch.roll(f, -2, dims=-1) + 16 * torch.roll(f, -1, dims=-1)
        - 30 * f
        + 16 * torch.roll(f, 1, dims=-1) - torch.roll(f, 2, dims=-1)
    ) / (12 * dx ** 2)


def fd_d2x_2nd(f: torch.Tensor) -> torch.Tensor:
    """2nd-order central d^2/dx^2 (periodic, dim=-1)."""
    return (
        torch.roll(f, -1, dims=-1) - 2 * f + torch.roll(f, 1, dims=-1)
    ) / (dx ** 2)


def fd_d4x_direct(f: torch.Tensor) -> torch.Tensor:
    """Direct 5-point stencil for d^4/dx^4 (2nd order, periodic, dim=-1).

    d^4u/dx^4 = (u[i-2] - 4u[i-1] + 6u[i] - 4u[i+1] + u[i+2]) / dx^4
    """
    return (
        torch.roll(f, -2, dims=-1) - 4 * torch.roll(f, -1, dims=-1)
        + 6 * f
        - 4 * torch.roll(f, 1, dims=-1) + torch.roll(f, 2, dims=-1)
    ) / (dx ** 4)


def fd_d4x_iterated(f: torch.Tensor) -> torch.Tensor:
    """4th derivative via iterated 2nd-order Laplacian: d2x(d2x(u))."""
    return fd_d2x_2nd(fd_d2x_2nd(f))


def compute_residual(
    u: torch.Tensor,
    method_d4: str = 'direct',
) -> torch.Tensor:
    """
    Compute PDE residual.

    u_t - alpha_2*u_xx - alpha_4*u_xxxx - beta_2*1/2*(u_x)^2 = 0

    Args:
        u: [1, T, X] full trajectory
        method_d4: 'direct' or 'iterated' for 4th derivative

    Returns:
        R: [1, T_int, X] residual at interior timesteps
    """
    u_int = u[:, 1:-1]

    # Time derivative (2nd-order central)
    du_dt = (u[:, 2:] - u[:, :-2]) / (2 * DT)

    # Spatial derivatives
    du_dx = fd_dx_4th(u_int)
    d2u_dx2 = fd_d2x_4th(u_int)

    if method_d4 == 'direct':
        d4u_dx4 = fd_d4x_direct(u_int)
    else:
        d4u_dx4 = fd_d4x_iterated(u_int)

    # Gradient norm: 1/2 * (u_x)^2
    grad_norm = 0.5 * du_dx ** 2

    # Residual: u_t - alpha_2*u_xx - alpha_4*u_xxxx - beta_2*1/2*(u_x)^2 = 0
    R = du_dt - ALPHA_2 * d2u_dx2 - ALPHA_4 * d4u_dx4 - BETA_2 * grad_norm
    return R


def main():
    print("=" * 60)
    print("GT PDE Residual Verification - 1D KS (APEBench)")
    print("=" * 60)
    print(f"  alpha_2 = {ALPHA_2:.6e} (negative diffusion)")
    print(f"  alpha_4 = {ALPHA_4:.6e} (hyper-diffusion)")
    print(f"  beta_2  = {BETA_2:.6e} (gradient norm)")
    print(f"  dx = {dx:.6e}, dt = {DT}")
    print()

    with h5py.File(DATA_PATH, 'r') as f:
        scalar = torch.tensor(f['scalar'][:], dtype=torch.float64)  # [N, T, X, 1]
        print(f"  scalar shape: {scalar.shape}")

    N_SAMPLES = scalar.shape[0]
    u_all = scalar[..., 0]  # [N, T, X]

    # Data now contains test samples only (30 samples, 201 timesteps)
    test_start = 0
    n_test = u_all.shape[0]
    print(f"  Using all {n_test} samples (test only, full 201 timesteps)")

    T_int_test = u_all.shape[1] - 2  # 199 interior timesteps

    methods = ['direct', 'iterated']
    rms_results = {m: [] for m in methods}
    per_t_results = {m: torch.zeros(T_int_test, dtype=torch.float64) for m in methods}

    print(f"\nComputing PDE residuals for {n_test} test samples...")
    print(f"  Comparing: direct 5-point d4x vs iterated d2x(d2x)")
    print()

    for i in range(test_start, test_start + n_test):
        u = u_all[i:i + 1]  # [1, T, X]

        for method in methods:
            R = compute_residual(u, method_d4=method)
            rms = torch.sqrt(torch.mean(R ** 2)).item()
            rms_results[method].append(rms)
            per_t_results[method] += torch.sqrt(torch.mean(R ** 2, dim=(0, 2)))

        if (i - test_start + 1) % 10 == 0 or i == test_start:
            print(f"  Sample {i - test_start + 1}/{n_test}: "
                  f"direct={rms_results['direct'][-1]:.6e}, "
                  f"iterated={rms_results['iterated'][-1]:.6e}")

    # Average per-timestep
    for m in methods:
        per_t_results[m] /= n_test

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Results (N={n_test} test samples)")
    print(f"{'=' * 60}")
    for m in methods:
        arr = np.array(rms_results[m])
        print(f"  {m:10s}: mean_RMS = {arr.mean():.6e}, max = {arr.max():.6e}, min = {arr.min():.6e}")

    # Pick winner
    direct_mean = np.mean(rms_results['direct'])
    iterated_mean = np.mean(rms_results['iterated'])

    if direct_mean <= iterated_mean:
        winner = 'direct'
    else:
        winner = 'iterated'

    winner_arr = np.array(rms_results[winner])
    winner_per_t = per_t_results[winner]
    print(f"\n  >>> Winner: {winner} (lower mean RMS)")

    # eq_scales (test samples only)
    overall_mean_rms = winner_arr.mean()

    print(f"\n  --- eq_scales for config ---")
    print(f"  physics:")
    print(f"    eq_scales:")
    print(f"      momentum: {overall_mean_rms:.4e}")

    # Save per-timestep scales
    gt_scales_dir = "./data/gt_scales"
    os.makedirs(gt_scales_dir, exist_ok=True)
    per_t_path = os.path.join(gt_scales_dir, "ks_1d_per_t.pt")
    torch.save({
        'momentum': winner_per_t.float(),
        'scheme': winner,
        'direct_mean_rms': float(direct_mean),
        'iterated_mean_rms': float(iterated_mean),
        'overall_mean_rms': float(overall_mean_rms),
    }, per_t_path)
    print(f"\n  Saved per-timestep scales to {per_t_path}")
    print(f"    momentum: shape={winner_per_t.shape}, mean={winner_per_t.mean():.6e}")

    # Print first/last few per-t values
    print(f"\n  Per-timestep RMS (first 10):")
    for t in range(min(10, len(winner_per_t))):
        print(f"    t={t+1}: {winner_per_t[t].item():.6e}")
    print(f"  Per-timestep RMS (last 5):")
    for t in range(max(0, len(winner_per_t) - 5), len(winner_per_t)):
        print(f"    t={t+1}: {winner_per_t[t].item():.6e}")

    print(f"\n{'=' * 60}")


if __name__ == '__main__':
    main()
