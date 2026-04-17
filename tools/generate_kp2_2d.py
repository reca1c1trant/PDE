"""
Generate KP-II (Kadomtsev-Petviashvili) 2D+t dataset with exact 1-line-soliton solution.

PDE: d/dx(u_t + 6u*u_x + u_xxx) + 3*u_yy = 0
Expanded: u_xt + 6*(u_x)^2 + 6*u*u_xx + u_xxxx + 3*u_yy = 0

Solution: u(x,y,t) = (a^2/2) * sech^2[a/2 * (x + b*y - omega*t - x0)]
          where omega = (3b^2 + a^2) / 4

Domain: [0, 20]^2, periodic BC
Resolution: 256x256, dx=20/256, dt=0.01, T=101 steps
Parameters: a in [0.5, 2.0], b in [-1.0, 1.0], ~150 samples
Channels: scalar=[u], scalar_indices=[0]

NOTE: The line-soliton solution is NOT exactly periodic on [0,L]^2.
      The soliton extends along the line x+by=const, cutting through the
      periodic boundary. This is the standard approach in KP-II benchmarks
      (e.g., IS-FNO). The PDE residual from periodic FD is small in the
      interior but has boundary artifacts — this is expected and acceptable
      for training neural PDE solvers.

Usage:
    python tools/generate_kp2_2d.py
"""

import numpy as np
import h5py
import os
from tqdm import tqdm


def analytical_solution(
    X: np.ndarray,
    Y: np.ndarray,
    t: float,
    a: float,
    b: float,
    x0: float,
) -> np.ndarray:
    """
    Compute exact 1-line-soliton solution of KP-II.

    u(x,y,t) = (a^2/2) * sech^2[a/2 * (x + b*y - omega*t - x0)]
    omega = (3*b^2 + a^2) / 4

    Args:
        X, Y: meshgrid arrays [H, W]
        t: time
        a: amplitude parameter (a > 0)
        b: direction parameter
        x0: initial phase shift

    Returns:
        u: [H, W] solution field
    """
    omega = (3.0 * b**2 + a**2) / 4.0
    phase = (a / 2.0) * (X + b * Y - omega * t - x0)

    # sech^2 = 1/cosh^2, clamp phase to avoid overflow
    phase_clamped = np.clip(phase, -50.0, 50.0)
    u = (a**2 / 2.0) / np.cosh(phase_clamped)**2

    return u


def generate_dataset():
    # Config
    N_GRID = 256
    N_T = 101
    DT = 0.01
    L = 20.0
    dx = L / N_GRID

    N_SAMPLES = 150
    OUTPUT_PATH = "/scratch-share/SONG0304/finetune/kp2_2d.hdf5"

    # Spatial grid (boundary-exclusive for periodic BC)
    x = np.linspace(0.0, L, N_GRID, endpoint=False)
    y = np.linspace(0.0, L, N_GRID, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='xy')  # [H, W]

    # Time grid
    t_arr = np.arange(N_T) * DT

    # Random parameters
    np.random.seed(42)
    a_values = np.random.uniform(1.0, 2.5, N_SAMPLES)  # a >= 1.0 ensures sech^2 tail < 1.8e-4 at boundary
    b_values = np.random.uniform(-1.0, 1.0, N_SAMPLES)

    # Center the soliton in the phase coordinate
    # xi = x + b*y ranges from 0 to L*(1+|b|) across the domain
    # At t=0, peak at xi=x0; at t=T, peak at xi=x0+omega*T
    # Center mid-trajectory at the center of xi-range
    x0_values = np.zeros(N_SAMPLES, dtype=np.float64)
    for i in range(N_SAMPLES):
        omega = (3.0 * b_values[i]**2 + a_values[i]**2) / 4.0
        t_max = (N_T - 1) * DT
        xi_center = L * (1.0 + abs(b_values[i])) / 2.0
        x0_values[i] = xi_center - omega * t_max / 2.0

    print(f"KP-II Dataset Generation")
    print(f"  Grid: {N_GRID}x{N_GRID}, dx={dx:.6f}")
    print(f"  Time: T={N_T}, dt={DT}")
    print(f"  Domain: [0, {L}]^2")
    print(f"  Samples: {N_SAMPLES}")
    print(f"  a range: [{a_values.min():.3f}, {a_values.max():.3f}]")
    print(f"  b range: [{b_values.min():.3f}, {b_values.max():.3f}]")
    print()

    # Generate — scalar only [N, T, H, W, 1]
    all_scalar = np.zeros((N_SAMPLES, N_T, N_GRID, N_GRID, 1), dtype=np.float32)
    nu_values = np.zeros(N_SAMPLES, dtype=np.float32)

    for i in tqdm(range(N_SAMPLES), desc="Generating KP-II samples"):
        a = a_values[i]
        b = b_values[i]
        x0 = x0_values[i]

        nu_values[i] = a  # store amplitude parameter

        for k in range(N_T):
            u = analytical_solution(X, Y, t_arr[k], a, b, x0)
            all_scalar[i, k, :, :, 0] = u.astype(np.float32)

    # Sanity checks
    print(f"\nSanity checks:")
    print(f"  scalar shape: {all_scalar.shape}")
    print(f"  scalar range: [{all_scalar.min():.6f}, {all_scalar.max():.6f}]")
    print(f"  NaN count: {np.isnan(all_scalar).sum()}")
    print(f"  Inf count: {np.isinf(all_scalar).sum()}")

    # Report statistics for a few samples
    for i in [0, N_SAMPLES // 2, N_SAMPLES - 1]:
        u_all_t = all_scalar[i, :, :, :, 0]
        u0 = u_all_t[0]
        print(f"  Sample {i} (a={a_values[i]:.3f}, b={b_values[i]:.3f}):")
        print(f"    u range: [{u_all_t.min():.6f}, {u_all_t.max():.6f}]")
        print(f"    u(t=0) mean={u0.mean():.6f}, max={u0.max():.6f}")

    # Save HDF5
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with h5py.File(OUTPUT_PATH, 'w') as f:
        f.create_dataset('scalar', data=all_scalar, dtype=np.float32)
        f.create_dataset('scalar_indices', data=np.array([0], dtype=np.int64))
        f.create_dataset('nu', data=nu_values, dtype=np.float32)

        # Store parameters for reproducibility
        f.create_dataset('params_a', data=a_values.astype(np.float32))
        f.create_dataset('params_b', data=b_values.astype(np.float32))
        f.create_dataset('params_x0', data=x0_values.astype(np.float32))

    file_size = os.path.getsize(OUTPUT_PATH) / 1e9
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"  File size: {file_size:.2f} GB")

    # Also create a symlink in the local data directory
    local_path = "./data/finetune/kp2_2d.hdf5"
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if os.path.exists(local_path) or os.path.islink(local_path):
        os.remove(local_path)
    os.symlink(OUTPUT_PATH, local_path)
    print(f"  Symlink: {local_path} -> {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_dataset()
