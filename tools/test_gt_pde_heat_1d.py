"""
GT PDE Residual Verification for 1D Heat/Diffusion (periodic).

Verifies:
    u_t - alpha * u_xx = 0

Uses 4th-order central for u_xx, 2nd-order central for u_t.

Usage:
    python tools/test_gt_pde_heat_1d.py
"""

import torch
import h5py
import numpy as np
import os

DATA_PATH = "/scratch-share/SONG0304/finetune/heat_1d.hdf5"
L = 2.0 * np.pi
N_GRID = 1024
DT = 0.01
dx = L / N_GRID


def fd_d2x_4th(f: torch.Tensor) -> torch.Tensor:
    """4th-order central d^2/dx^2 in x (dim=-1), periodic."""
    fE = torch.roll(f, -1, dims=-1)
    fW = torch.roll(f, 1, dims=-1)
    fEE = torch.roll(f, -2, dims=-1)
    fWW = torch.roll(f, 2, dims=-1)
    return (-fEE + 16 * fE - 30 * f + 16 * fW - fWW) / (12 * dx**2)


def compute_residual(u: torch.Tensor, alpha: float) -> torch.Tensor:
    """Compute heat equation residual: R = u_t - alpha * u_xx.

    Args:
        u: [1, T, X]
    Returns:
        R: [1, T_int, X] residual
    """
    u_int = u[:, 1:-1]

    # Time derivative (2nd-order central)
    du_dt = (u[:, 2:] - u[:, :-2]) / (2 * DT)

    # Diffusion: alpha * u_xx (4th-order central)
    u_xx = fd_d2x_4th(u_int)
    diffusion = alpha * u_xx

    # Residual: u_t - alpha * u_xx = 0
    R = du_dt - diffusion
    return R


def main():
    print("=" * 60)
    print("GT PDE Residual Verification - 1D Heat/Diffusion")
    print("=" * 60)

    # Load data
    with h5py.File(DATA_PATH, 'r') as f:
        scalar = torch.tensor(f['scalar'][:], dtype=torch.float64)  # [N, T, X, 1]
        nu_vals = f['nu'][:]
        print(f"  scalar shape: {scalar.shape}")
        print(f"  alpha range: [{nu_vals.min():.4f}, {nu_vals.max():.4f}]")

    N_SAMPLES = scalar.shape[0]
    u_all = scalar[..., 0]  # [N, T, X]

    T_int = u_all.shape[1] - 2  # interior timesteps
    rms_list = []
    per_t_accum = torch.zeros(T_int, dtype=torch.float64)

    print(f"\nComputing PDE residuals for {N_SAMPLES} samples...")
    print()

    for i in range(N_SAMPLES):
        alpha = float(nu_vals[i])
        u = u_all[i:i+1]  # [1, T, X]

        R = compute_residual(u, alpha)
        rms = torch.sqrt(torch.mean(R ** 2)).item()
        rms_list.append(rms)

        # Per-timestep RMS: sqrt(mean over B,X) for each t
        per_t_accum += torch.sqrt(torch.mean(R ** 2, dim=(0, 2)))

        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Sample {i+1}/{N_SAMPLES}: RMS = {rms:.6e}")

    # Average per-timestep
    per_t_avg = per_t_accum / N_SAMPLES

    # Summary
    rms_arr = np.array(rms_list)

    print(f"\n{'=' * 60}")
    print(f"Results (N={N_SAMPLES})")
    print(f"{'=' * 60}")
    print(f"  4th-order central:  mean_RMS = {rms_arr.mean():.6e}, max = {rms_arr.max():.6e}")

    # eq_scales for config
    print(f"\n  --- eq_scales for config (mean RMS) ---")
    print(f"  physics:")
    print(f"    eq_scales:")
    print(f"      heat: {rms_arr.mean():.4e}")

    # Save per-timestep scales
    gt_scales_dir = "./data/gt_scales"
    os.makedirs(gt_scales_dir, exist_ok=True)
    per_t_path = os.path.join(gt_scales_dir, "heat_1d_per_t.pt")
    torch.save({
        'heat': per_t_avg.float(),
        'mean_rms': float(rms_arr.mean()),
    }, per_t_path)
    print(f"\n  Saved per-timestep scales to {per_t_path}")
    print(f"    heat: shape={per_t_avg.shape}, mean={per_t_avg.mean():.6e}")

    print(f"\n{'=' * 60}")


if __name__ == '__main__':
    main()
