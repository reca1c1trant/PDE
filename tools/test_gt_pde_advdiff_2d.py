"""GT PDE residual verification for 2D Advection-Diffusion dataset.

Computes per-equation RMS residuals to determine eq_scales
for normalized PDE loss.

Advection-Diffusion equation:
    u_t + a*u_x + b*u_y = nu*(u_xx + u_yy)

Numerical method:
    - Spatial: 4th-order central difference (periodic BC via np.roll)
    - Temporal: 2nd-order central difference

Usage:
    python tools/test_gt_pde_advdiff_2d.py \
        --data /scratch-share/SONG0304/finetune/advdiff_2d.hdf5
"""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import torch


def fd_dx_4th_periodic(f, dx):
    """4th-order central first derivative in x (axis=-1), periodic BC."""
    return (
        -np.roll(f, -2, axis=-1) + 8 * np.roll(f, -1, axis=-1)
        - 8 * np.roll(f, 1, axis=-1) + np.roll(f, 2, axis=-1)
    ) / (12 * dx)


def fd_dy_4th_periodic(f, dy):
    """4th-order central first derivative in y (axis=-2), periodic BC."""
    return (
        -np.roll(f, -2, axis=-2) + 8 * np.roll(f, -1, axis=-2)
        - 8 * np.roll(f, 1, axis=-2) + np.roll(f, 2, axis=-2)
    ) / (12 * dy)


def fd_laplacian_4th_periodic(f, dx, dy):
    """4th-order central FD Laplacian with periodic BC."""
    d2f_dx2 = (
        -np.roll(f, -2, axis=-1) + 16 * np.roll(f, -1, axis=-1)
        - 30 * f
        + 16 * np.roll(f, 1, axis=-1) - np.roll(f, 2, axis=-1)
    ) / (12 * dx ** 2)
    d2f_dy2 = (
        -np.roll(f, -2, axis=-2) + 16 * np.roll(f, -1, axis=-2)
        - 30 * f
        + 16 * np.roll(f, 1, axis=-2) - np.roll(f, 2, axis=-2)
    ) / (12 * dy ** 2)
    return d2f_dx2 + d2f_dy2


def compute_residuals(u, dx, dy, dt, a, b, nu):
    """Compute advection-diffusion PDE residual.

    Args:
        u: shape (T, H, W)
        dx, dy, dt: grid spacings
        a, b: advection velocities
        nu: diffusion coefficient
    Returns:
        dict with 'advdiff' residual array, shape (T-2, H, W)
    """
    # 2nd-order central time difference
    du_dt = (u[2:] - u[:-2]) / (2 * dt)

    # Spatial terms at mid-frame
    u_n = u[1:-1]

    du_dx = fd_dx_4th_periodic(u_n, dx)
    du_dy = fd_dy_4th_periodic(u_n, dy)
    lap_u = fd_laplacian_4th_periodic(u_n, dx, dy)

    # Residual: u_t + a*u_x + b*u_y - nu*(u_xx + u_yy) = 0
    R = du_dt + a * du_dx + b * du_dy - nu * lap_u

    return {'advdiff': R}


def main():
    parser = argparse.ArgumentParser(description="GT PDE residual for Advection-Diffusion 2D")
    parser.add_argument("--data", type=str, required=True, help="Path to HDF5 file")
    parser.add_argument("--max_samples", type=int, default=150)
    parser.add_argument("--a", type=float, default=1.0, help="Advection velocity x")
    parser.add_argument("--b", type=float, default=0.5, help="Advection velocity y")
    args = parser.parse_args()

    L = 2 * np.pi
    N_GRID = 256
    dx = L / N_GRID
    dy = dx
    dt = 0.01

    print(f"Loading: {args.data}")
    with h5py.File(args.data, 'r') as f:
        scalar = f['scalar']
        nu_vals = f['nu'][:]
        n_samples_total = scalar.shape[0]
        n_time = scalar.shape[1]
        a_vel = float(f.attrs.get('a', args.a))
        b_vel = float(f.attrs.get('b', args.b))
        print(f"  Samples={n_samples_total}, T={n_time}, "
              f"Grid={scalar.shape[2]}x{scalar.shape[3]}, Scalar_ch={scalar.shape[4]}")
        print(f"  a={a_vel}, b={b_vel}")
        print(f"  dx={dx:.6f}, dy={dy:.6f}, dt={dt}")

        n_proc = min(n_samples_total, args.max_samples)

        eq_names = ['advdiff']
        all_rms = {k: [] for k in eq_names}
        all_rms_per_t = {k: [] for k in eq_names}

        for s_idx in range(n_proc):
            scl = np.array(scalar[s_idx], dtype=np.float64)  # [T, H, W, 1]
            u = scl[:, :, :, 0]  # [T, H, W]
            nu = float(nu_vals[s_idx])

            res = compute_residuals(u, dx, dy, dt, a_vel, b_vel, nu)

            for eq in eq_names:
                rms = np.sqrt(np.mean(res[eq] ** 2))
                all_rms[eq].append(rms)

                # Per-timestep RMS for eq_scales_per_t
                rms_per_t = np.sqrt(np.mean(res[eq] ** 2, axis=(1, 2)))  # [T-2]
                all_rms_per_t[eq].append(rms_per_t)

            if s_idx < 3 or s_idx == n_proc - 1:
                print(f"  Sample {s_idx} (nu={nu:.4f}): "
                      f"advdiff={all_rms['advdiff'][-1]:.4e}")

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
    gt_scales_dir = os.path.join(os.path.dirname(args.data), '..', 'gt_scales')
    gt_scales_dir = './data/gt_scales'
    os.makedirs(gt_scales_dir, exist_ok=True)
    per_t_dict = {}
    for eq in eq_names:
        per_t_arr = np.array(all_rms_per_t[eq])  # [N, T-2]
        mean_per_t = np.mean(per_t_arr, axis=0)  # [T-2]
        per_t_dict[eq] = torch.tensor(mean_per_t, dtype=torch.float32)
        print(f"\n  {eq} per-t RMS: shape={mean_per_t.shape}, "
              f"min={mean_per_t.min():.4e}, max={mean_per_t.max():.4e}")

    save_path = os.path.join(gt_scales_dir, 'advdiff_2d_per_t.pt')
    torch.save(per_t_dict, save_path)
    print(f"\nSaved per-timestep scales to: {save_path}")


if __name__ == "__main__":
    main()
