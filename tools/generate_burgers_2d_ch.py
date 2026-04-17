"""
Generate 2D Burgers (Cole-Hopf, periodic) dataset with exact analytical solution.

Solution B (separable product form):
    theta(x,y,t) = f(x,t) * g(y,t)
    f(x,t) = 1 + eps_x * exp(-nu*kx^2*t) * cos(kx*x)
    g(y,t) = 1 + eps_y * exp(-nu*ky^2*t) * cos(ky*y)

    u = -2*nu * f_x / f
    v = -2*nu * g_y / g

PDE: u_t + u*u_x + v*u_y = nu*(u_xx + u_yy)
     v_t + u*v_x + v*v_y = nu*(v_xx + v_yy)
Constraint: irrotational (v_x - u_y = 0)

Domain: [0, 2*pi]^2, periodic BC
Resolution: 256x256, dt=0.01, T=101
"""

import numpy as np
import h5py
import os
from tqdm import tqdm
import time
from typing import Tuple


def analytical_solution(
    X: np.ndarray,
    Y: np.ndarray,
    t: float,
    nu: float,
    eps_x: float,
    eps_y: float,
    kx: float,
    ky: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute exact Cole-Hopf solution (separable product form)."""
    decay_x = np.exp(-nu * kx**2 * t)
    f_val = 1.0 + eps_x * decay_x * np.cos(kx * X)
    f_x = -eps_x * kx * decay_x * np.sin(kx * X)

    decay_y = np.exp(-nu * ky**2 * t)
    g_val = 1.0 + eps_y * decay_y * np.cos(ky * Y)
    g_y = -eps_y * ky * decay_y * np.sin(ky * Y)

    u = -2.0 * nu * f_x / f_val
    v = -2.0 * nu * g_y / g_val

    return u, v


def verify_sample(data, nu, eps_x, eps_y, kx, ky, sample_idx):
    """Verify the generated sample satisfies basic constraints."""
    u_data = data[..., 0]
    v_data = data[..., 1]

    assert np.all(np.isfinite(u_data)), f"Sample {sample_idx}: u has NaN/Inf"
    assert np.all(np.isfinite(v_data)), f"Sample {sample_idx}: v has NaN/Inf"

    max_u = np.abs(u_data).max()
    max_v = np.abs(v_data).max()
    assert max_u < 10.0, f"Sample {sample_idx}: |u| too large: {max_u}"
    assert max_v < 10.0, f"Sample {sample_idx}: |v| too large: {max_v}"

    amp_u_0 = np.abs(u_data[0]).max()
    amp_u_T = np.abs(u_data[-1]).max()
    if amp_u_0 > 1e-10:
        assert amp_u_T <= amp_u_0 * 1.01, (
            f"Sample {sample_idx}: u not decaying: {amp_u_0:.6e} -> {amp_u_T:.6e}"
        )
    return True


def generate_dataset():
    """Generate Burgers 2D Cole-Hopf dataset."""
    print("=" * 70)
    print("2D Burgers (Cole-Hopf, Periodic) Dataset Generation")
    print("Solution B: Separable Product Form")
    print("=" * 70)

    N_GRID = 256
    N_T = 101
    DT = 0.01
    L = 2.0 * np.pi
    DOMAIN = [0.0, L]
    dx = L / N_GRID

    OUTPUT_DIR = "/scratch-share/SONG0304/finetune"
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "burgers_2d_ch.hdf5")
    LOCAL_LINK = "./data/finetune/burgers_2d_ch.hdf5"

    NU_MIN, NU_MAX = 0.01, 0.1
    EPS_MIN, EPS_MAX = 0.1, 0.5
    KX_MODES = [1, 2, 3]
    KY_MODES = [1, 2, 3]
    SEED = 42

    np.random.seed(SEED)

    samples_per_combo = 6
    param_list = []
    for kx_mode in KX_MODES:
        for ky_mode in KY_MODES:
            for _ in range(samples_per_combo):
                nu = np.random.uniform(NU_MIN, NU_MAX)
                eps_x = np.random.uniform(EPS_MIN, EPS_MAX)
                eps_y = np.random.uniform(EPS_MIN, EPS_MAX)
                param_list.append({
                    'nu': nu, 'eps_x': eps_x, 'eps_y': eps_y,
                    'kx': float(kx_mode), 'ky': float(ky_mode),
                })

    N_SAMPLES = len(param_list)

    print(f"\nConfiguration:")
    print(f"  Number of samples: {N_SAMPLES}")
    print(f"  Spatial resolution: {N_GRID} x {N_GRID}")
    print(f"  Domain: [0, 2*pi]^2")
    print(f"  Grid spacing: dx = dy = {dx:.6f}")
    print(f"  Temporal steps: {N_T}, dt = {DT}")
    print(f"  Output file: {OUTPUT_FILE}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    x = np.linspace(DOMAIN[0], DOMAIN[1], N_GRID, endpoint=False)
    y = np.linspace(DOMAIN[0], DOMAIN[1], N_GRID, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='xy')

    t_arr = np.arange(N_T) * DT

    print(f"\nGenerating {N_SAMPLES} samples...")
    all_vector = np.zeros((N_SAMPLES, N_T, N_GRID, N_GRID, 3), dtype=np.float32)
    nu_values = np.zeros(N_SAMPLES, dtype=np.float32)

    start_time = time.time()
    for i in tqdm(range(N_SAMPLES), desc="  Progress"):
        params = param_list[i]
        nu = params['nu']
        eps_x = params['eps_x']
        eps_y = params['eps_y']
        kx = params['kx']
        ky = params['ky']
        nu_values[i] = nu

        for k in range(N_T):
            u, v = analytical_solution(X, Y, t_arr[k], nu, eps_x, eps_y, kx, ky)
            all_vector[i, k, :, :, 0] = u.astype(np.float32)
            all_vector[i, k, :, :, 1] = v.astype(np.float32)

        verify_sample(all_vector[i], nu, eps_x, eps_y, kx, ky, i)

    elapsed = time.time() - start_time
    print(f"  Generated in {elapsed:.1f}s")

    u_data = all_vector[..., 0]
    v_data = all_vector[..., 1]
    print(f"\n  u range: [{u_data.min():.6f}, {u_data.max():.6f}]")
    print(f"  v range: [{v_data.min():.6f}, {v_data.max():.6f}]")
    print(f"  u std: {u_data.std():.6f}")
    print(f"  v std: {v_data.std():.6f}")

    print(f"\n[Saving to HDF5...]")
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    with h5py.File(OUTPUT_FILE, 'w') as f:
        f.create_dataset('vector', data=all_vector, dtype=np.float32)
        f.create_dataset('nu', data=nu_values, dtype=np.float32)

        eps_x_arr = np.array([p['eps_x'] for p in param_list], dtype=np.float32)
        eps_y_arr = np.array([p['eps_y'] for p in param_list], dtype=np.float32)
        kx_arr = np.array([p['kx'] for p in param_list], dtype=np.float32)
        ky_arr = np.array([p['ky'] for p in param_list], dtype=np.float32)
        f.create_dataset('eps_x', data=eps_x_arr, dtype=np.float32)
        f.create_dataset('eps_y', data=eps_y_arr, dtype=np.float32)
        f.create_dataset('kx', data=kx_arr, dtype=np.float32)
        f.create_dataset('ky', data=ky_arr, dtype=np.float32)

        f.attrs['description'] = '2D Burgers Cole-Hopf (separable product, periodic)'
        f.attrs['n_samples'] = N_SAMPLES
        f.attrs['grid_size'] = N_GRID
        f.attrs['dx'] = dx
        f.attrs['dt'] = DT
        f.attrs['temporal_steps'] = N_T
        f.attrs['domain'] = [0.0, L]

    # Symlink in local data dir
    os.makedirs(os.path.dirname(LOCAL_LINK), exist_ok=True)
    if os.path.islink(LOCAL_LINK) or os.path.exists(LOCAL_LINK):
        os.remove(LOCAL_LINK)
    os.symlink(os.path.abspath(OUTPUT_FILE), LOCAL_LINK)
    print(f"  Symlink: {LOCAL_LINK} -> {OUTPUT_FILE}")

    with h5py.File(OUTPUT_FILE, 'r') as f:
        print(f"  vector shape: {f['vector'].shape}")
        print(f"  nu shape: {f['nu'].shape}")

    file_size = os.path.getsize(OUTPUT_FILE) / (1024**3)
    print(f"  File size: {file_size:.2f} GB")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)

    return OUTPUT_FILE


if __name__ == "__main__":
    generate_dataset()
