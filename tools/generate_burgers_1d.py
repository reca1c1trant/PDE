"""
Generate 1D Burgers (Cole-Hopf, periodic) dataset with exact analytical solution.

PDE:  u_t + u * u_x = nu * u_xx

Cole-Hopf transform: u = -2*nu * (ln theta)_x
where theta satisfies heat equation: theta_t = nu * theta_xx

For periodic domain with Fourier IC:
    theta(x,t) = 1 + sum_k eps_k * exp(-nu*k^2*t) * cos(k*x + phi_k)
    u(x,t) = 2*nu * sum_k eps_k*k*sin(k*x + phi_k)*exp(-nu*k^2*t) / theta(x,t)

Domain: [0, 2*pi], periodic, endpoint=False
Resolution: 1024, dt=0.01, T=101
"""

import numpy as np
import h5py
import os
from tqdm import tqdm
import time
from typing import List, Tuple


def analytical_solution(
    x: np.ndarray,
    t: float,
    nu: float,
    k_modes: np.ndarray,
    eps_modes: np.ndarray,
    phi_modes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute exact Cole-Hopf solution for 1D Burgers.

    Returns:
        u: velocity field [X]
        theta: heat kernel [X] (for positivity check)
    """
    # theta = 1 + sum_k eps_k * exp(-nu*k^2*t) * cos(k*x + phi_k)
    theta = np.ones_like(x)
    theta_x = np.zeros_like(x)
    for k, eps, phi in zip(k_modes, eps_modes, phi_modes):
        decay = np.exp(-nu * k**2 * t)
        theta += eps * decay * np.cos(k * x + phi)
        theta_x += -eps * k * decay * np.sin(k * x + phi)

    # u = -2*nu * theta_x / theta
    u = -2.0 * nu * theta_x / theta
    return u, theta


def verify_sample(
    u_data: np.ndarray,
    theta_min: float,
    sample_idx: int,
) -> None:
    """Verify generated sample is valid."""
    assert np.all(np.isfinite(u_data)), f"Sample {sample_idx}: u has NaN/Inf"
    assert theta_min > 0, f"Sample {sample_idx}: theta non-positive (min={theta_min:.6e})"

    max_u = np.abs(u_data).max()
    assert max_u < 100.0, f"Sample {sample_idx}: |u| too large: {max_u:.4f}"

    # Viscous decay: amplitude at T should not exceed initial amplitude (within tolerance)
    amp_0 = np.abs(u_data[0]).max()
    amp_T = np.abs(u_data[-1]).max()
    if amp_0 > 1e-10:
        assert amp_T <= amp_0 * 1.01, (
            f"Sample {sample_idx}: u not decaying: {amp_0:.6e} -> {amp_T:.6e}"
        )


def generate_dataset() -> str:
    """Generate Burgers 1D Cole-Hopf dataset."""
    print("=" * 70)
    print("1D Burgers (Cole-Hopf, Periodic) Dataset Generation")
    print("=" * 70)

    # Grid parameters
    N_GRID = 256
    N_T = 101
    DT = 0.01
    L = 2.0 * np.pi
    dx = L / N_GRID  # 2*pi/256 = 0.024544, matches physics.dx in config

    # Output
    OUTPUT_DIR = "/scratch-share/SONG0304/finetune"
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "burgers_1d.hdf5")
    LOCAL_LINK = "./data/finetune/burgers_1d.hdf5"

    # Parameter ranges
    N_SAMPLES = 1000
    NU_MIN, NU_MAX = 0.005, 0.1
    K_MIN, K_MAX = 1, 5
    N_MODES_MIN, N_MODES_MAX = 3, 8
    EPS_MIN, EPS_MAX = 0.1, 0.5
    SEED = 42

    np.random.seed(SEED)

    print(f"\nConfiguration:")
    print(f"  N_samples: {N_SAMPLES}")
    print(f"  Resolution: {N_GRID}")
    print(f"  Domain: [0, 2*pi], periodic, endpoint=False")
    print(f"  dx = {dx:.8f}")
    print(f"  Temporal steps: {N_T}, dt = {DT}")
    print(f"  nu range: [{NU_MIN}, {NU_MAX}]")
    print(f"  Fourier modes: {N_MODES_MIN}-{N_MODES_MAX}, k in [{K_MIN}, {K_MAX}]")
    print(f"  eps range: [{EPS_MIN}, {EPS_MAX}]")
    print(f"  Output: {OUTPUT_FILE}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    x = np.linspace(0, L, N_GRID, endpoint=False)
    t_arr = np.arange(N_T) * DT

    # Generate parameter list
    # IMPORTANT: theta = 1 + sum_k eps_k * cos(...) must stay positive.
    # Worst case: all cos terms = -1 => theta_min = 1 - sum(eps_k).
    # So we need sum(eps_k) < 1. We cap total amplitude at MAX_TOTAL_EPS.
    MAX_TOTAL_EPS = 0.9
    param_list: List[dict] = []
    for _ in range(N_SAMPLES):
        nu = np.random.uniform(NU_MIN, NU_MAX)
        n_modes = np.random.randint(N_MODES_MIN, N_MODES_MAX + 1)
        k_modes = np.random.randint(K_MIN, K_MAX + 1, size=n_modes).astype(np.float64)
        eps_raw = np.random.uniform(EPS_MIN, EPS_MAX, size=n_modes)
        # Rescale so sum(eps) = target_total, where target_total ~ U[0.3, MAX_TOTAL_EPS]
        target_total = np.random.uniform(0.3, MAX_TOTAL_EPS)
        eps_modes = eps_raw * (target_total / eps_raw.sum())
        phi_modes = np.random.uniform(0, 2 * np.pi, size=n_modes)
        param_list.append({
            'nu': nu,
            'k_modes': k_modes,
            'eps_modes': eps_modes,
            'phi_modes': phi_modes,
        })

    # Allocate arrays
    # HDF5 format: vector [N, T, X, 3], scalar [N, T, X, 1]
    all_scalar = np.zeros((N_SAMPLES, N_T, N_GRID, 0), dtype=np.float32)
    all_vector = np.zeros((N_SAMPLES, N_T, N_GRID, 3), dtype=np.float32)
    nu_values = np.zeros(N_SAMPLES, dtype=np.float32)

    theta_min_global = float('inf')

    print(f"\nGenerating {N_SAMPLES} samples...")
    start_time = time.time()
    for i in tqdm(range(N_SAMPLES), desc="  Progress"):
        params = param_list[i]
        nu = params['nu']
        nu_values[i] = nu

        sample_theta_min = float('inf')
        for k in range(N_T):
            u, theta = analytical_solution(
                x, t_arr[k], nu,
                params['k_modes'], params['eps_modes'], params['phi_modes'],
            )
            all_vector[i, k, :, 0] = u.astype(np.float32)  # u is velocity -> vector[0]
            sample_theta_min = min(sample_theta_min, theta.min())

        theta_min_global = min(theta_min_global, sample_theta_min)
        verify_sample(all_vector[i, :, :, 0], sample_theta_min, i)

    elapsed = time.time() - start_time
    print(f"  Generated in {elapsed:.1f}s")

    u_data = all_vector[..., 0]
    print(f"\n  u range: [{u_data.min():.6f}, {u_data.max():.6f}]")
    print(f"  u std: {u_data.std():.6f}")
    print(f"  theta_min (global): {theta_min_global:.6e}")

    # Save HDF5
    print(f"\n[Saving to HDF5...]")
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    with h5py.File(OUTPUT_FILE, 'w') as f:
        f.create_dataset('vector', data=all_vector, dtype=np.float32)
        f.create_dataset('scalar', data=all_scalar, dtype=np.float32)
        f.create_dataset('scalar_indices', data=np.array([], dtype=np.int64))
        f.create_dataset('nu', data=nu_values, dtype=np.float32)

        f.attrs['description'] = '1D Burgers Cole-Hopf (periodic, Fourier IC)'
        f.attrs['n_samples'] = N_SAMPLES
        f.attrs['grid_size'] = N_GRID
        f.attrs['dx'] = dx
        f.attrs['dt'] = DT
        f.attrs['temporal_steps'] = N_T
        f.attrs['domain'] = [0.0, L]

    # Symlink
    os.makedirs(os.path.dirname(LOCAL_LINK), exist_ok=True)
    if os.path.islink(LOCAL_LINK) or os.path.exists(LOCAL_LINK):
        os.remove(LOCAL_LINK)
    os.symlink(os.path.abspath(OUTPUT_FILE), LOCAL_LINK)
    print(f"  Symlink: {LOCAL_LINK} -> {OUTPUT_FILE}")

    # Verify
    with h5py.File(OUTPUT_FILE, 'r') as f:
        print(f"  vector shape: {f['vector'].shape}")
        print(f"  scalar shape: {f['scalar'].shape}")
        print(f"  scalar_indices: {f['scalar_indices'][:]}")
        print(f"  nu shape: {f['nu'].shape}")

    file_size = os.path.getsize(OUTPUT_FILE) / (1024**3)
    print(f"  File size: {file_size:.2f} GB")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)

    return OUTPUT_FILE


if __name__ == "__main__":
    generate_dataset()
