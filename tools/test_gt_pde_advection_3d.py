"""GT PDE residual verification for 3D Advection dataset.

Computes per-equation RMS residuals to determine eq_scales
for normalized PDE loss.

Advection equation:
    u_t + a*u_x + b*u_y + c*u_z = 0

Numerical method:
    - Spatial: 4th-order central difference (periodic BC via np.roll)
    - Temporal: 2nd-order central difference

Usage:
    python tools/test_gt_pde_advection_3d.py \
        --data /home/msai/song0304/code/PDE/data/finetune/advection_3d.hdf5
"""

import argparse
import os

import h5py
import numpy as np
import torch


def fd_dx_4th_periodic(f: np.ndarray, dx: float) -> np.ndarray:
    """4th-order central first derivative in x (axis=-3), periodic BC."""
    return (
        -np.roll(f, -2, axis=-3) + 8 * np.roll(f, -1, axis=-3)
        - 8 * np.roll(f, 1, axis=-3) + np.roll(f, 2, axis=-3)
    ) / (12 * dx)


def fd_dy_4th_periodic(f: np.ndarray, dy: float) -> np.ndarray:
    """4th-order central first derivative in y (axis=-2), periodic BC."""
    return (
        -np.roll(f, -2, axis=-2) + 8 * np.roll(f, -1, axis=-2)
        - 8 * np.roll(f, 1, axis=-2) + np.roll(f, 2, axis=-2)
    ) / (12 * dy)


def fd_dz_4th_periodic(f: np.ndarray, dz: float) -> np.ndarray:
    """4th-order central first derivative in z (axis=-1), periodic BC."""
    return (
        -np.roll(f, -2, axis=-1) + 8 * np.roll(f, -1, axis=-1)
        - 8 * np.roll(f, 1, axis=-1) + np.roll(f, 2, axis=-1)
    ) / (12 * dz)


def compute_residuals(
    u: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    dt: float,
    a: float,
    b: float,
    c: float,
) -> dict:
    """Compute 3D advection PDE residual.

    Args:
        u: shape [T, X, Y, Z]
        dx, dy, dz, dt: grid spacings
        a, b, c: advection velocity components

    Returns:
        dict with 'advection' residual array, shape [T-2, X, Y, Z]
    """
    # 2nd-order central time derivative
    du_dt = (u[2:] - u[:-2]) / (2 * dt)

    # Spatial terms at mid-frame
    u_n = u[1:-1]

    du_dx = fd_dx_4th_periodic(u_n, dx)
    du_dy = fd_dy_4th_periodic(u_n, dy)
    du_dz = fd_dz_4th_periodic(u_n, dz)

    # Residual: u_t + a*u_x + b*u_y + c*u_z = 0
    R = du_dt + a * du_dx + b * du_dy + c * du_dz

    return {'advection': R}


def main():
    parser = argparse.ArgumentParser(description="GT PDE residual for 3D Advection")
    parser.add_argument(
        "--data", type=str,
        default="/home/msai/song0304/code/PDE/data/finetune/advection_3d.hdf5",
        help="Path to HDF5 file",
    )
    parser.add_argument("--max_samples", type=int, default=100)
    args = parser.parse_args()

    L = 2 * np.pi
    N_GRID = 64
    dx = L / N_GRID
    dy = dx
    dz = dx
    dt = 0.05

    print(f"Loading: {args.data}")
    with h5py.File(args.data, 'r') as f:
        scalar = f['scalar']
        a_vals = f['params_a'][:]
        b_vals = f['params_b'][:]
        c_vals = f['params_c'][:]
        n_samples_total = scalar.shape[0]
        n_time = scalar.shape[1]
        print(f"  Samples={n_samples_total}, T={n_time}, "
              f"Grid={scalar.shape[2]}x{scalar.shape[3]}x{scalar.shape[4]}, "
              f"Scalar_ch={scalar.shape[5]}")
        print(f"  dx={dx:.6f}, dy={dy:.6f}, dz={dz:.6f}, dt={dt}")

        n_proc = min(n_samples_total, args.max_samples)

        eq_names = ['advection']
        all_rms = {k: [] for k in eq_names}
        all_rms_per_t = {k: [] for k in eq_names}

        for s_idx in range(n_proc):
            scl = np.array(scalar[s_idx], dtype=np.float64)  # [T, X, Y, Z, 1]
            u = scl[:, :, :, :, 0]  # [T, X, Y, Z]
            a_i = float(a_vals[s_idx])
            b_i = float(b_vals[s_idx])
            c_i = float(c_vals[s_idx])

            res = compute_residuals(u, dx, dy, dz, dt, a_i, b_i, c_i)

            for eq in eq_names:
                rms = np.sqrt(np.mean(res[eq] ** 2))
                all_rms[eq].append(rms)

                # Per-timestep RMS for eq_scales_per_t
                rms_per_t = np.sqrt(np.mean(res[eq] ** 2, axis=(1, 2, 3)))  # [T-2]
                all_rms_per_t[eq].append(rms_per_t)

            if s_idx < 3 or s_idx == n_proc - 1:
                v_mag = np.sqrt(a_i**2 + b_i**2 + c_i**2)
                print(f"  Sample {s_idx} (|v|={v_mag:.4f}, a={a_i:.4f}, b={b_i:.4f}, c={c_i:.4f}): "
                      f"advection={all_rms['advection'][-1]:.4e}")

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
    gt_scales_dir = './data/gt_scales'
    os.makedirs(gt_scales_dir, exist_ok=True)
    per_t_dict = {}
    for eq in eq_names:
        per_t_arr = np.array(all_rms_per_t[eq])  # [N, T-2]
        mean_per_t = np.mean(per_t_arr, axis=0)  # [T-2]
        per_t_dict[eq] = torch.tensor(mean_per_t, dtype=torch.float32)
        print(f"\n  {eq} per-t RMS: shape={mean_per_t.shape}, "
              f"min={mean_per_t.min():.4e}, max={mean_per_t.max():.4e}")

    save_path = os.path.join(gt_scales_dir, 'advection_3d_per_t.pt')
    torch.save(per_t_dict, save_path)
    print(f"\nSaved per-timestep scales to: {save_path}")


if __name__ == "__main__":
    main()
