"""
Generate 1D Heat/Diffusion dataset with exact analytical solution.

PDE:  u_t = alpha * u_xx

Exact solution on periodic domain [0, L]:
    u(x, t) = sum_k A_k * exp(-alpha * k^2 * (2*pi/L)^2 * t) * cos(k * (2*pi/L) * x + phi_k)

Each Fourier mode decays exponentially. L = 2*pi => (2*pi/L) = 1.

Domain: [0, 2*pi], periodic, endpoint=False
Resolution: 1024, dt=0.01, T=101
"""

import numpy as np
import h5py
import os
from tqdm import tqdm
import time
from typing import List


def analytical_solution(
    x: np.ndarray,
    t: float,
    alpha: float,
    k_modes: np.ndarray,
    amp_modes: np.ndarray,
    phi_modes: np.ndarray,
) -> np.ndarray:
    """
    Compute exact Fourier solution for 1D heat equation on [0, 2*pi].

    u(x,t) = sum_k A_k * exp(-alpha * k^2 * t) * cos(k*x + phi_k)

    Returns:
        u: temperature field [X]
    """
    u = np.zeros_like(x)
    for k, A, phi in zip(k_modes, amp_modes, phi_modes):
        decay = np.exp(-alpha * k**2 * t)
        u += A * decay * np.cos(k * x + phi)
    return u


def verify_sample(
    u_data: np.ndarray,
    alpha: float,
    sample_idx: int,
) -> None:
    """Verify generated sample is valid."""
    assert np.all(np.isfinite(u_data)), f"Sample {sample_idx}: u has NaN/Inf"

    # Diffusion must decay: amplitude at T should not exceed initial
    amp_0 = np.abs(u_data[0]).max()
    amp_T = np.abs(u_data[-1]).max()
    if amp_0 > 1e-10:
        assert amp_T <= amp_0 * 1.001, (
            f"Sample {sample_idx}: u not decaying: {amp_0:.6e} -> {amp_T:.6e}"
        )


def generate_dataset() -> str:
    """Generate Heat 1D dataset."""
    print("=" * 70)
    print("1D Heat/Diffusion (Periodic, Fourier IC) Dataset Generation")
    print("=" * 70)

    # Grid parameters
    N_GRID = 1024
    N_T = 101
    DT = 0.01
    L = 2.0 * np.pi
    dx = L / N_GRID

    # Output
    OUTPUT_DIR = "/scratch-share/SONG0304/finetune"
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "heat_1d.hdf5")
    LOCAL_LINK = "./data/finetune/heat_1d.hdf5"

    # Parameter ranges
    N_SAMPLES = 1000
    ALPHA_MIN, ALPHA_MAX = 0.01, 0.5
    K_MIN, K_MAX = 1, 10
    N_MODES_MIN, N_MODES_MAX = 5, 15
    AMP_MIN, AMP_MAX = 0.1, 1.0
    SEED = 42

    np.random.seed(SEED)

    print(f"\nConfiguration:")
    print(f"  N_samples: {N_SAMPLES}")
    print(f"  Resolution: {N_GRID}")
    print(f"  Domain: [0, 2*pi], periodic, endpoint=False")
    print(f"  dx = {dx:.8f}")
    print(f"  Temporal steps: {N_T}, dt = {DT}")
    print(f"  alpha range: [{ALPHA_MIN}, {ALPHA_MAX}]")
    print(f"  Fourier modes: {N_MODES_MIN}-{N_MODES_MAX}, k in [{K_MIN}, {K_MAX}]")
    print(f"  Amplitude range: [{AMP_MIN}, {AMP_MAX}]")
    print(f"  Output: {OUTPUT_FILE}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    x = np.linspace(0, L, N_GRID, endpoint=False)
    t_arr = np.arange(N_T) * DT

    # Generate parameter list
    param_list: List[dict] = []
    for _ in range(N_SAMPLES):
        alpha = np.random.uniform(ALPHA_MIN, ALPHA_MAX)
        n_modes = np.random.randint(N_MODES_MIN, N_MODES_MAX + 1)
        k_modes = np.random.randint(K_MIN, K_MAX + 1, size=n_modes).astype(np.float64)
        amp_modes = np.random.uniform(AMP_MIN, AMP_MAX, size=n_modes)
        phi_modes = np.random.uniform(0, 2 * np.pi, size=n_modes)
        param_list.append({
            'alpha': alpha,
            'k_modes': k_modes,
            'amp_modes': amp_modes,
            'phi_modes': phi_modes,
        })

    # Allocate arrays
    # HDF5 format: vector [N, T, X, 3], scalar [N, T, X, 1]
    all_scalar = np.zeros((N_SAMPLES, N_T, N_GRID, 1), dtype=np.float32)
    all_vector = np.zeros((N_SAMPLES, N_T, N_GRID, 3), dtype=np.float32)
    alpha_values = np.zeros(N_SAMPLES, dtype=np.float32)

    print(f"\nGenerating {N_SAMPLES} samples...")
    start_time = time.time()
    for i in tqdm(range(N_SAMPLES), desc="  Progress"):
        params = param_list[i]
        alpha = params['alpha']
        alpha_values[i] = alpha

        for k in range(N_T):
            u = analytical_solution(
                x, t_arr[k], alpha,
                params['k_modes'], params['amp_modes'], params['phi_modes'],
            )
            all_scalar[i, k, :, 0] = u.astype(np.float32)

        verify_sample(all_scalar[i, :, :, 0], alpha, i)

    elapsed = time.time() - start_time
    print(f"  Generated in {elapsed:.1f}s")

    u_data = all_scalar[..., 0]
    print(f"\n  u range: [{u_data.min():.6f}, {u_data.max():.6f}]")
    print(f"  u std: {u_data.std():.6f}")
    print(f"  alpha range: [{alpha_values.min():.4f}, {alpha_values.max():.4f}]")

    # Save HDF5
    print(f"\n[Saving to HDF5...]")
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    with h5py.File(OUTPUT_FILE, 'w') as f:
        f.create_dataset('vector', data=all_vector, dtype=np.float32)
        f.create_dataset('scalar', data=all_scalar, dtype=np.float32)
        f.create_dataset('scalar_indices', data=np.array([14], dtype=np.int64))
        f.create_dataset('nu', data=alpha_values, dtype=np.float32)

        f.attrs['description'] = '1D Heat/Diffusion (periodic, Fourier IC)'
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
