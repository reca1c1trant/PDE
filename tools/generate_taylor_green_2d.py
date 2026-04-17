"""Generate Taylor-Green Vortex 2D+t dataset with exact analytical solution.

PDE: Incompressible Navier-Stokes
    u_t + u*u_x + v*u_y = -p_x + nu*(u_xx + u_yy)
    v_t + u*v_x + v*v_y = -p_y + nu*(v_xx + v_yy)
    u_x + v_y = 0

Exact solution:
    u(x,y,t) = -cos(x)*sin(y)*exp(-2*nu*t)
    v(x,y,t) =  sin(x)*cos(y)*exp(-2*nu*t)
    p(x,y,t) = -1/4*(cos(2x)+cos(2y))*exp(-4*nu*t)

Domain: [0, 2*pi]^2, periodic BC
Resolution: 256x256, dt=0.01, T=101 steps
Parameters: nu in [0.01, 0.1], ~50 samples

Channels: vector=[u,v,0], scalar=[p], scalar_indices=[12] (pressure at ch15)
"""

import numpy as np
import h5py
import os
from tqdm import tqdm


def analytical_solution(
    X: np.ndarray,
    Y: np.ndarray,
    t: float,
    nu: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact Taylor-Green vortex solution.

    Args:
        X: x-coordinates meshgrid [H, W]
        Y: y-coordinates meshgrid [H, W]
        t: time
        nu: kinematic viscosity

    Returns:
        u, v, p fields each [H, W]
    """
    decay_vel = np.exp(-2 * nu * t)
    decay_pres = np.exp(-4 * nu * t)

    u = -np.cos(X) * np.sin(Y) * decay_vel
    v = np.sin(X) * np.cos(Y) * decay_vel
    p = -0.25 * (np.cos(2 * X) + np.cos(2 * Y)) * decay_pres

    return u, v, p


def generate_dataset():
    # Config
    N_GRID = 256
    N_T = 101
    DT = 0.01
    L = 2 * np.pi
    N_SAMPLES = 50
    NU_MIN = 0.01
    NU_MAX = 0.1

    OUTPUT_PATH = "/scratch-share/SONG0304/finetune/taylor_green_2d.hdf5"
    LOCAL_LINK = "./data/finetune/taylor_green_2d.hdf5"
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Spatial grid (boundary-exclusive for periodic BC)
    x = np.linspace(0, L, N_GRID, endpoint=False)
    y = np.linspace(0, L, N_GRID, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='xy')  # [H, W]

    # Time grid
    t_arr = np.arange(N_T) * DT

    # Random parameters
    np.random.seed(42)
    nu_values = np.random.uniform(NU_MIN, NU_MAX, N_SAMPLES).astype(np.float32)

    # Allocate arrays
    vector_data = np.zeros((N_SAMPLES, N_T, N_GRID, N_GRID, 3), dtype=np.float32)
    scalar_data = np.zeros((N_SAMPLES, N_T, N_GRID, N_GRID, 1), dtype=np.float32)

    print(f"Generating Taylor-Green Vortex 2D dataset")
    print(f"  Grid: {N_GRID}x{N_GRID}, T={N_T}, dt={DT}")
    print(f"  Domain: [0, {L:.4f}]^2")
    print(f"  dx = dy = {L / N_GRID:.6f}")
    print(f"  nu range: [{NU_MIN}, {NU_MAX}], N_samples={N_SAMPLES}")

    for i in tqdm(range(N_SAMPLES), desc="Generating samples"):
        nu = float(nu_values[i])
        for k in range(N_T):
            u, v, p = analytical_solution(X, Y, t_arr[k], nu)
            vector_data[i, k, :, :, 0] = u
            vector_data[i, k, :, :, 1] = v
            # vector_data[i, k, :, :, 2] = 0  already zero
            scalar_data[i, k, :, :, 0] = p

    # Sanity checks
    print("\nSanity checks:")
    print(f"  vector range: [{vector_data.min():.6f}, {vector_data.max():.6f}]")
    print(f"  scalar range: [{scalar_data.min():.6f}, {scalar_data.max():.6f}]")
    print(f"  NaN count: vector={np.isnan(vector_data).sum()}, scalar={np.isnan(scalar_data).sum()}")
    print(f"  Inf count: vector={np.isinf(vector_data).sum()}, scalar={np.isinf(scalar_data).sum()}")

    # Verify continuity at t=0 for first sample
    nu0 = float(nu_values[0])
    u0 = vector_data[0, 0, :, :, 0]
    v0 = vector_data[0, 0, :, :, 1]
    dx = L / N_GRID
    # Simple central diff for quick check
    du_dx = (np.roll(u0, -1, axis=1) - np.roll(u0, 1, axis=1)) / (2 * dx)
    dv_dy = (np.roll(v0, -1, axis=0) - np.roll(v0, 1, axis=0)) / (2 * dx)
    div_max = np.max(np.abs(du_dx + dv_dy))
    print(f"  Continuity check (sample 0, t=0): max|div(u)| = {div_max:.6e}")

    # Save HDF5
    with h5py.File(OUTPUT_PATH, 'w') as f:
        f.create_dataset('vector', data=vector_data, dtype=np.float32)
        f.create_dataset('scalar', data=scalar_data, dtype=np.float32)
        f.create_dataset('scalar_indices', data=np.array([12], dtype=np.int64))
        f.create_dataset('nu', data=nu_values, dtype=np.float32)

    file_size = os.path.getsize(OUTPUT_PATH) / 1e9
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"File size: {file_size:.2f} GB")
    print(f"  vector shape: {vector_data.shape}")
    print(f"  scalar shape: {scalar_data.shape}")
    print(f"  scalar_indices: [12]")
    print(f"  nu shape: {nu_values.shape}")

    # Symlink in local data dir
    os.makedirs(os.path.dirname(LOCAL_LINK), exist_ok=True)
    if os.path.islink(LOCAL_LINK) or os.path.exists(LOCAL_LINK):
        os.remove(LOCAL_LINK)
    os.symlink(os.path.abspath(OUTPUT_PATH), LOCAL_LINK)
    print(f"  Symlink: {LOCAL_LINK} -> {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_dataset()
