"""
GT PDE residual verification for 2D Wave Equation dataset.

PDE (first-order system):
    Eq 1: u_t - w = 0
    Eq 2: w_t - c^2 (u_xx + u_yy) = 0

Numerical method:
    - Spatial: 4th-order central difference Laplacian (periodic BC, np.roll)
    - Temporal: 2nd-order central difference

The wave speed c is stored per-sample in the 'nu' field.

Grid: boundary-exclusive [0, 2pi) with N=256, dx = 2pi/256.

Usage:
    python tools/test_gt_pde_wave_2d.py --data ./data/finetune/wave_2d.hdf5
"""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np


def fd_laplacian_4th(f: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """4th-order central FD Laplacian with periodic BC (np.roll).

    Args:
        f: shape [..., H, W]
        dx, dy: grid spacing
    """
    d2f_dx2 = (
        -np.roll(f, -2, axis=-2) + 16 * np.roll(f, -1, axis=-2)
        - 30 * f
        + 16 * np.roll(f, 1, axis=-2) - np.roll(f, 2, axis=-2)
    ) / (12 * dx**2)
    d2f_dy2 = (
        -np.roll(f, -2, axis=-1) + 16 * np.roll(f, -1, axis=-1)
        - 30 * f
        + 16 * np.roll(f, 1, axis=-1) - np.roll(f, 2, axis=-1)
    ) / (12 * dy**2)
    return d2f_dx2 + d2f_dy2


def compute_residuals(
    u: np.ndarray,
    w: np.ndarray,
    c: float,
    dx: float,
    dy: float,
    dt: float,
) -> dict[str, np.ndarray]:
    """Compute Wave Equation PDE residuals.

    Args:
        u, w: shape (T, H, W), float64
        c: wave speed
        dx, dy: grid spacing
        dt: time step

    Returns:
        dict with 'u_equation', 'w_equation' residual arrays [T-2, H, W]
    """
    # 2nd-order central time difference
    du_dt = (u[2:] - u[:-2]) / (2 * dt)
    dw_dt = (w[2:] - w[:-2]) / (2 * dt)

    # Spatial terms at mid-frame t=1..T-2
    u_mid = u[1:-1]
    w_mid = w[1:-1]

    # Laplacian of u
    lap_u = fd_laplacian_4th(u_mid, dx, dy)

    # Eq 1: u_t - w = 0
    R_u = du_dt - w_mid

    # Eq 2: w_t - c^2 * laplacian(u) = 0
    R_w = dw_dt - c**2 * lap_u

    return {
        'u_equation': R_u,
        'w_equation': R_w,
    }


def main():
    parser = argparse.ArgumentParser(description="GT PDE residual for 2D Wave Equation")
    parser.add_argument("--data", type=str, required=True, help="Path to HDF5 file")
    parser.add_argument("--max_samples", type=int, default=100)
    args = parser.parse_args()

    # Grid parameters: boundary-exclusive
    L = 2 * np.pi
    N_GRID = 256
    dx = L / N_GRID  # boundary-exclusive: 2pi/256
    dy = dx
    dt = 0.01

    print(f"Loading: {args.data}")
    print(f"Grid: {N_GRID}x{N_GRID} (boundary-exclusive), dx={dx:.6f}, dy={dy:.6f}, dt={dt}")

    with h5py.File(args.data, 'r') as f:
        scalar = f['scalar']
        nu_vals = f['nu'][:]  # c values
        n_samples_total = scalar.shape[0]
        n_time = scalar.shape[1]
        print(f"  Samples={n_samples_total}, T={n_time}, "
              f"Grid={scalar.shape[2]}x{scalar.shape[3]}, Scalar_ch={scalar.shape[4]}")

        n_proc = min(n_samples_total, args.max_samples)

        eq_names = ['u_equation', 'w_equation']
        all_rms = {k: [] for k in eq_names}
        all_rms_per_t = {k: [] for k in eq_names}

        for s_idx in range(n_proc):
            scl = np.array(scalar[s_idx], dtype=np.float64)  # [T, H, W, 2]
            u = scl[:, :, :, 0]  # [T, H, W]
            w = scl[:, :, :, 1]
            c_val = float(nu_vals[s_idx])

            res = compute_residuals(u, w, c_val, dx, dy, dt)

            for eq in eq_names:
                rms = np.sqrt(np.mean(res[eq] ** 2))
                all_rms[eq].append(rms)

                rms_per_t = np.sqrt(np.mean(res[eq] ** 2, axis=(-2, -1)))
                all_rms_per_t[eq].append(rms_per_t)

            if s_idx < 3 or s_idx == n_proc - 1:
                print(f"  Sample {s_idx} (c={c_val:.4f}): "
                      f"u_eq={all_rms['u_equation'][-1]:.4e}, "
                      f"w_eq={all_rms['w_equation'][-1]:.4e}")

    # Summary
    print(f"\n{'='*70}")
    print(f"Summary over {n_proc} samples")
    print(f"{'='*70}")

    print(f"\n{'Equation':<15} | {'Mean RMS':>12} | {'Std RMS':>12} | {'Min RMS':>12} | {'Max RMS':>12}")
    print("-" * 70)
    for eq in eq_names:
        arr = np.array(all_rms[eq])
        print(f"{eq:<15} | {arr.mean():>12.4e} | {arr.std():>12.4e} | "
              f"{arr.min():>12.4e} | {arr.max():>12.4e}")

    # eq_scales for config
    print(f"\n--- eq_scales for config (Mean RMS values) ---")
    print("physics:")
    print("  eq_scales:")
    for eq in eq_names:
        arr = np.array(all_rms[eq])
        print(f"    {eq}: {arr.mean():.4e}")

    # Save per-timestep scales
    import torch
    gt_scales_dir = Path("data/gt_scales")
    gt_scales_dir.mkdir(parents=True, exist_ok=True)
    per_t_data = {}
    for eq in eq_names:
        stacked = np.stack(all_rms_per_t[eq], axis=0)  # [N, T-2]
        mean_per_t = np.sqrt(np.mean(stacked**2, axis=0))  # RMS across samples [T-2]
        per_t_data[eq] = torch.tensor(mean_per_t, dtype=torch.float32)
        print(f"\n  {eq} per-t shape: {per_t_data[eq].shape}, "
              f"min={per_t_data[eq].min():.4e}, max={per_t_data[eq].max():.4e}")

    save_path = str(gt_scales_dir / "wave_2d_per_t.pt")
    torch.save(per_t_data, save_path)
    print(f"\nSaved per-timestep scales to {save_path}")


if __name__ == "__main__":
    main()
