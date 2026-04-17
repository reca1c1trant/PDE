"""GT PDE residual verification for 2D Anisotropic Diffusion.

Computes per-equation RMS residuals to determine eq_scales
for normalized PDE loss.

Anisotropic Diffusion equation:
    u_t = div(A * grad(u))
    u_t = A_xx * u_xx + (A_xy + A_yx) * u_xy + A_yy * u_yy

With default APEBench coefficients:
    A = [[0.001, 0.0005],
         [0.0005, 0.002]]

So:
    u_t = 0.001 * u_xx + 0.001 * u_xy + 0.002 * u_yy

Domain: [0, 1)^2, periodic BC, 160x160, dt=0.1

Numerical method:
    - Spatial: 4th-order central difference (periodic BC via np.roll)
    - Temporal: 2nd-order central difference

Usage:
    python tools/test_gt_pde_aniso_diff_2d.py \\
        --data /scratch-share/SONG0304/finetune/apebench_2d_aniso_diff.hdf5
"""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import torch


def fd_dx_4th_periodic(f: np.ndarray, dx: float) -> np.ndarray:
    """4th-order central first derivative in x (axis=-1), periodic BC."""
    return (
        -np.roll(f, -2, axis=-1) + 8 * np.roll(f, -1, axis=-1)
        - 8 * np.roll(f, 1, axis=-1) + np.roll(f, 2, axis=-1)
    ) / (12 * dx)


def fd_dy_4th_periodic(f: np.ndarray, dy: float) -> np.ndarray:
    """4th-order central first derivative in y (axis=-2), periodic BC."""
    return (
        -np.roll(f, -2, axis=-2) + 8 * np.roll(f, -1, axis=-2)
        - 8 * np.roll(f, 1, axis=-2) + np.roll(f, 2, axis=-2)
    ) / (12 * dy)


def fd_d2x_4th_periodic(f: np.ndarray, dx: float) -> np.ndarray:
    """4th-order central second derivative in x (axis=-1), periodic BC."""
    return (
        -np.roll(f, -2, axis=-1) + 16 * np.roll(f, -1, axis=-1)
        - 30 * f
        + 16 * np.roll(f, 1, axis=-1) - np.roll(f, 2, axis=-1)
    ) / (12 * dx ** 2)


def fd_d2y_4th_periodic(f: np.ndarray, dy: float) -> np.ndarray:
    """4th-order central second derivative in y (axis=-2), periodic BC."""
    return (
        -np.roll(f, -2, axis=-2) + 16 * np.roll(f, -1, axis=-2)
        - 30 * f
        + 16 * np.roll(f, 1, axis=-2) - np.roll(f, 2, axis=-2)
    ) / (12 * dy ** 2)


def fd_dxy_4th_periodic(f: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """4th-order central cross derivative d^2u/(dx dy), periodic BC.

    Apply 4th-order d/dx then d/dy (or vice versa, same for smooth fields).
    """
    # First take d/dx, then d/dy of that result
    dudx = fd_dx_4th_periodic(f, dx)
    return fd_dy_4th_periodic(dudx, dy)


def compute_residuals(
    u: np.ndarray,
    dx: float,
    dy: float,
    dt: float,
    A_xx: float,
    A_xy: float,
    A_yx: float,
    A_yy: float,
) -> dict:
    """Compute anisotropic diffusion PDE residual.

    Args:
        u: shape (T, H, W), float64
        dx, dy, dt: grid spacings
        A_xx, A_xy, A_yx, A_yy: diffusion tensor components

    Returns:
        dict with 'aniso_diff' residual array, shape (T-2, H, W)
    """
    # 2nd-order central time difference
    du_dt = (u[2:] - u[:-2]) / (2 * dt)

    # Spatial terms at mid-frame
    u_n = u[1:-1]

    u_xx = fd_d2x_4th_periodic(u_n, dx)
    u_yy = fd_d2y_4th_periodic(u_n, dy)
    u_xy = fd_dxy_4th_periodic(u_n, dx, dy)

    # Residual: u_t - A_xx*u_xx - (A_xy + A_yx)*u_xy - A_yy*u_yy = 0
    R = du_dt - A_xx * u_xx - (A_xy + A_yx) * u_xy - A_yy * u_yy

    return {"aniso_diff": R}


def main():
    parser = argparse.ArgumentParser(description="GT PDE residual for Anisotropic Diffusion 2D")
    parser.add_argument("--data", type=str, required=True, help="Path to HDF5 file")
    parser.add_argument("--max_samples", type=int, default=30)
    parser.add_argument("--A_xx", type=float, default=0.001)
    parser.add_argument("--A_xy", type=float, default=0.0005)
    parser.add_argument("--A_yx", type=float, default=0.0005)
    parser.add_argument("--A_yy", type=float, default=0.002)
    args = parser.parse_args()

    # APEBench anisotropic diffusion: domain_extent=1.0, num_points=160
    L = 1.0
    N_GRID = 160
    dx = L / N_GRID
    dy = dx
    dt = 0.1

    print(f"Loading: {args.data}")
    print(f"Diffusion matrix A:")
    print(f"  [[{args.A_xx}, {args.A_xy}],")
    print(f"   [{args.A_yx}, {args.A_yy}]]")
    print(f"Domain: L={L}, N={N_GRID}, dx={dx:.6f}, dt={dt}")

    with h5py.File(args.data, "r") as f:
        scalar = f["scalar"]
        n_samples_total = scalar.shape[0]
        n_time = scalar.shape[1]
        print(f"  Samples={n_samples_total}, T={n_time}, "
              f"Grid={scalar.shape[2]}x{scalar.shape[3]}, Scalar_ch={scalar.shape[4]}")

        n_proc = min(n_samples_total, args.max_samples)

        eq_names = ["aniso_diff"]
        all_rms = {k: [] for k in eq_names}
        all_rms_per_t = {k: [] for k in eq_names}

        for s_idx in range(n_proc):
            scl = np.array(scalar[s_idx], dtype=np.float64)  # [T, H, W, 1]
            u = scl[:, :, :, 0]  # [T, H, W]

            res = compute_residuals(
                u, dx, dy, dt,
                args.A_xx, args.A_xy, args.A_yx, args.A_yy,
            )

            for eq in eq_names:
                rms = np.sqrt(np.mean(res[eq] ** 2))
                all_rms[eq].append(rms)

                # Per-timestep RMS
                rms_per_t = np.sqrt(np.mean(res[eq] ** 2, axis=(1, 2)))  # [T-2]
                all_rms_per_t[eq].append(rms_per_t)

            if s_idx < 3 or s_idx == n_proc - 1:
                print(f"  Sample {s_idx}: aniso_diff={all_rms['aniso_diff'][-1]:.4e}")

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
    gt_scales_dir = "./data/gt_scales"
    os.makedirs(gt_scales_dir, exist_ok=True)
    per_t_dict = {}
    for eq in eq_names:
        per_t_arr = np.array(all_rms_per_t[eq])  # [N, T-2]
        mean_per_t = np.mean(per_t_arr, axis=0)   # [T-2]
        per_t_dict[eq] = torch.tensor(mean_per_t, dtype=torch.float32)
        print(f"\n  {eq} per-t RMS: shape={mean_per_t.shape}, "
              f"min={mean_per_t.min():.4e}, max={mean_per_t.max():.4e}")

    save_path = os.path.join(gt_scales_dir, "aniso_diff_2d_per_t.pt")
    torch.save(per_t_dict, save_path)
    print(f"\nSaved per-timestep scales to: {save_path}")


if __name__ == "__main__":
    main()
