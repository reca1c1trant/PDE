"""
Generate 1D Advection dataset with exact analytical solution.

PDE:  u_t + a * u_x = 0

Exact solution: u(x, t) = u_0(x - a*t)  (translated IC, periodic wrapping)

IC: Multi-mode Fourier
    u_0(x) = sum_{j=1}^{n_modes} A_j * sin(k_j * x + phi_j)
    where A_j ~ U[0.3, 1.5] / k_j (1/k decay), phi_j ~ U[0, 2*pi]
    n_modes ~ U{5, 12}, k_j ~ U{1, 8}

Domain: [0, 2*pi], periodic, endpoint=False
Resolution: 1024, dt=0.01, T=101
Channels: scalar=[u], scalar_indices=[11] (ch14, matches CH_U=14 in train script)
"""

import numpy as np
import h5py
import os
from tqdm import tqdm
import time
from typing import List


def generate_ic_fourier(
    x: np.ndarray,
    n_modes: int,
    k_modes: np.ndarray,
    amp_modes: np.ndarray,
    phi_modes: np.ndarray,
) -> np.ndarray:
    """
    Generate multi-mode Fourier IC.

    Returns:
        u0: initial condition [X]
    """
    u0 = np.zeros_like(x)
    for k, amp, phi in zip(k_modes, amp_modes, phi_modes):
        u0 += amp * np.sin(k * x + phi)
    return u0


def exact_solution(
    x: np.ndarray,
    t: float,
    a: float,
    n_modes: int,
    k_modes: np.ndarray,
    amp_modes: np.ndarray,
    phi_modes: np.ndarray,
) -> np.ndarray:
    """
    Exact solution: u(x,t) = u_0(x - a*t) with periodic wrapping.

    Since IC is Fourier sum of sin(k*x + phi), the translated solution is:
        u(x,t) = sum_j A_j * sin(k_j * (x - a*t) + phi_j)

    This naturally handles periodic wrapping for integer wavenumbers.

    Returns:
        u: field at time t [X]
    """
    u = np.zeros_like(x)
    for k, amp, phi in zip(k_modes, amp_modes, phi_modes):
        u += amp * np.sin(k * (x - a * t) + phi)
    return u


def verify_sample(u_data: np.ndarray, sample_idx: int) -> None:
    """Verify generated sample is valid."""
    assert np.all(np.isfinite(u_data)), f"Sample {sample_idx}: u has NaN/Inf"

    max_u = np.abs(u_data).max()
    assert max_u < 200.0, f"Sample {sample_idx}: |u| too large: {max_u:.4f}"

    # Advection preserves amplitude exactly in float64; float32 storage
    # introduces ~1e-3 relative error via discretization, so use loose tolerance
    amp_0 = np.abs(u_data[0]).max()
    amp_T = np.abs(u_data[-1]).max()
    if amp_0 > 1e-10:
        assert abs(amp_T - amp_0) / amp_0 < 5e-3, (
            f"Sample {sample_idx}: amplitude changed: {amp_0:.6e} -> {amp_T:.6e}"
        )


def generate_dataset() -> str:
    """Generate Advection 1D dataset."""
    print("=" * 70)
    print("1D Advection (Periodic, Fourier IC) Dataset Generation")
    print("=" * 70)

    # Grid parameters
    N_GRID = 256
    N_T = 101
    DT = 0.01
    L = 2.0 * np.pi
    dx = L / N_GRID  # 2*pi/256 = 0.024544, matches physics.dx in config

    # Output
    OUTPUT_DIR = "/scratch-share/SONG0304/finetune"
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "advection_1d.hdf5")
    LOCAL_LINK = "./data/finetune/advection_1d.hdf5"

    # Parameter ranges
    N_SAMPLES = 1000
    A_MIN, A_MAX = 0.5, 3.0
    K_MIN, K_MAX = 1, 8
    N_MODES_MIN, N_MODES_MAX = 5, 12
    SEED = 42

    np.random.seed(SEED)

    print(f"\nConfiguration:")
    print(f"  N_samples: {N_SAMPLES}")
    print(f"  Resolution: {N_GRID}")
    print(f"  Domain: [0, 2*pi], periodic, endpoint=False")
    print(f"  dx = {dx:.8f}")
    print(f"  Temporal steps: {N_T}, dt = {DT}")
    print(f"  Advection speed a: [{A_MIN}, {A_MAX}]")
    print(f"  Fourier modes: {N_MODES_MIN}-{N_MODES_MAX}, k in [{K_MIN}, {K_MAX}]")
    print(f"  Output: {OUTPUT_FILE}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    x = np.linspace(0, L, N_GRID, endpoint=False)
    t_arr = np.arange(N_T) * DT

    # Generate parameter list
    param_list: List[dict] = []
    for _ in range(N_SAMPLES):
        a_speed = np.random.uniform(A_MIN, A_MAX)
        n_modes = np.random.randint(N_MODES_MIN, N_MODES_MAX + 1)
        k_modes = np.random.randint(K_MIN, K_MAX + 1, size=n_modes).astype(np.float64)
        # Amplitudes with 1/k decay
        raw_amp = np.random.uniform(0.3, 1.5, size=n_modes)
        amp_modes = raw_amp / k_modes
        phi_modes = np.random.uniform(0, 2 * np.pi, size=n_modes)
        param_list.append({
            'a': a_speed,
            'k_modes': k_modes,
            'amp_modes': amp_modes,
            'phi_modes': phi_modes,
        })

    # Allocate arrays
    # HDF5 format: vector [N, T, X, 3], scalar [N, T, X, 1]
    all_scalar = np.zeros((N_SAMPLES, N_T, N_GRID, 1), dtype=np.float32)
    all_vector = np.zeros((N_SAMPLES, N_T, N_GRID, 3), dtype=np.float32)
    a_values = np.zeros(N_SAMPLES, dtype=np.float32)

    print(f"\nGenerating {N_SAMPLES} samples...")
    start_time = time.time()
    for i in tqdm(range(N_SAMPLES), desc="  Progress"):
        params = param_list[i]
        a_speed = params['a']
        a_values[i] = a_speed

        for k in range(N_T):
            u = exact_solution(
                x, t_arr[k], a_speed,
                len(params['k_modes']),
                params['k_modes'], params['amp_modes'], params['phi_modes'],
            )
            all_scalar[i, k, :, 0] = u.astype(np.float32)

        verify_sample(all_scalar[i, :, :, 0], i)

    elapsed = time.time() - start_time
    print(f"  Generated in {elapsed:.1f}s")

    u_data = all_scalar[..., 0]
    print(f"\n  u range: [{u_data.min():.6f}, {u_data.max():.6f}]")
    print(f"  u std: {u_data.std():.6f}")
    print(f"  a range: [{a_values.min():.4f}, {a_values.max():.4f}]")

    # Save HDF5
    print(f"\n[Saving to HDF5...]")
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    with h5py.File(OUTPUT_FILE, 'w') as f:
        f.create_dataset('vector', data=all_vector, dtype=np.float32)
        f.create_dataset('scalar', data=all_scalar, dtype=np.float32)
        f.create_dataset('scalar_indices', data=np.array([11], dtype=np.int64))  # scalar[11] -> ch14, matches CH_U=14 in train script
        f.create_dataset('nu', data=a_values, dtype=np.float32)

        f.attrs['description'] = '1D Advection (periodic, Fourier IC)'
        f.attrs['n_samples'] = N_SAMPLES
        f.attrs['grid_size'] = N_GRID
        f.attrs['dx'] = dx
        f.attrs['dt'] = DT
        f.attrs['temporal_steps'] = N_T
        f.attrs['domain'] = [0.0, L]
        f.attrs['a_range'] = [A_MIN, A_MAX]

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
