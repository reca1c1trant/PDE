"""
Generate 2D Wave Equation dataset with exact analytical solution.

PDE: u_tt = c^2 (u_xx + u_yy)
First-order system: u_t = w,  w_t = c^2 * laplacian(u)

Solution (multi-mode superposition, periodic BC on [0, 2pi)^2):
    u(x,y,t) = sum_{m,n} A_mn cos(kx_m*x + ky_n*y + phi_mn) cos(omega_mn*t)
    w(x,y,t) = -sum_{m,n} A_mn omega_mn cos(kx_m*x + ky_n*y + phi_mn) sin(omega_mn*t)

where kx_m = m, ky_n = n  (wavenumbers on [0,2pi] domain)
      omega_mn = c * sqrt(kx_m^2 + ky_n^2)

Grid: boundary-EXCLUSIVE [0, 2pi) with N=256, dx = 2pi/256.
      This ensures np.roll / torch.roll correctly handles periodicity.

Channels:
    scalar: [u, w] with scalar_indices=[0, 1]
    u -> ch3, w -> ch4 in 18-channel layout
    vector_dim = 0

Parameters:
    c in [1.0, 3.0], random mode amplitudes, 100 samples
"""

import numpy as np
import h5py
import os
import time
from tqdm import tqdm


def analytical_solution(
    X: np.ndarray,
    Y: np.ndarray,
    t: float,
    c: float,
    modes: list[tuple[int, int]],
    amplitudes: np.ndarray,
    phases: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute exact u(x,y,t) and w(x,y,t) = u_t(x,y,t).

    Args:
        X, Y: meshgrid arrays [H, W]
        t: time
        c: wave speed
        modes: list of (m, n) mode pairs
        amplitudes: A_mn for each mode
        phases: phi_mn for each mode

    Returns:
        u, w: arrays [H, W]
    """
    u = np.zeros_like(X)
    w = np.zeros_like(X)

    for idx, (m, n) in enumerate(modes):
        kx = float(m)
        ky = float(n)
        omega = c * np.sqrt(kx**2 + ky**2)
        A = amplitudes[idx]
        phi = phases[idx]

        spatial = np.cos(kx * X + ky * Y + phi)

        u += A * spatial * np.cos(omega * t)
        w += -A * omega * spatial * np.sin(omega * t)

    return u, w


def generate_modes(max_mode: int = 3) -> list[tuple[int, int]]:
    """Generate all (m, n) mode pairs with |m|, |n| <= max_mode, excluding (0,0)."""
    modes = []
    for m in range(-max_mode, max_mode + 1):
        for n in range(-max_mode, max_mode + 1):
            if m == 0 and n == 0:
                continue
            modes.append((m, n))
    return modes


def generate_dataset():
    # Config
    N_SAMPLES = 100
    N_GRID = 256
    N_T = 101
    DT = 0.01
    L = 2 * np.pi
    C_MIN = 1.0
    C_MAX = 3.0
    MAX_MODE = 3
    SEED = 42

    OUTPUT_DIR = "/scratch-share/SONG0304/finetune"
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, "wave_2d.hdf5")

    # Boundary-exclusive for periodic BC compatibility with roll
    dx = L / N_GRID

    print("=" * 70)
    print("2D Wave Equation Dataset Generation")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  N_samples: {N_SAMPLES}")
    print(f"  Grid: {N_GRID} x {N_GRID} (boundary-exclusive)")
    print(f"  Domain: [0, 2pi), dx = 2pi/{N_GRID} = {dx:.6f}")
    print(f"  Timesteps: {N_T}, dt = {DT}")
    print(f"  Wave speed c: [{C_MIN}, {C_MAX}]")
    print(f"  Max mode: |m|,|n| <= {MAX_MODE}")
    print(f"  Output: {OUTPUT_PATH}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Spatial grid (boundary-exclusive for periodic BC)
    x = np.linspace(0, L, N_GRID, endpoint=False)
    y = np.linspace(0, L, N_GRID, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='xy')  # [H, W]

    # Time grid
    t_arr = np.arange(N_T) * DT

    # All modes
    modes = generate_modes(MAX_MODE)
    n_modes = len(modes)
    print(f"  Number of modes: {n_modes}")

    # Random parameters
    np.random.seed(SEED)
    c_values = np.random.uniform(C_MIN, C_MAX, N_SAMPLES).astype(np.float32)

    # Generate data: scalar [N, T, H, W, 2] (u, w)
    all_scalar = np.zeros((N_SAMPLES, N_T, N_GRID, N_GRID, 2), dtype=np.float32)

    print(f"\nGenerating {N_SAMPLES} samples...")
    start_time = time.time()

    for i in tqdm(range(N_SAMPLES), desc="  Samples"):
        c_val = c_values[i]

        # Random amplitudes with decay: A_mn ~ 1/sqrt(m^2+n^2) * uniform(0.1, 1.0)
        raw_amp = np.random.uniform(0.1, 1.0, n_modes)
        decay = np.array([1.0 / np.sqrt(m**2 + n**2) for (m, n) in modes])
        amplitudes = raw_amp * decay

        # Random spatial phases
        phases = np.random.uniform(0, 2 * np.pi, n_modes)

        for k in range(N_T):
            u, w = analytical_solution(X, Y, t_arr[k], c_val, modes, amplitudes, phases)
            all_scalar[i, k, :, :, 0] = u
            all_scalar[i, k, :, :, 1] = w

    elapsed = time.time() - start_time
    print(f"  Generated in {elapsed:.1f}s")

    # Verify
    print("\nVerification:")
    print(f"  u range: [{all_scalar[:,:,:,:,0].min():.4f}, {all_scalar[:,:,:,:,0].max():.4f}]")
    print(f"  w range: [{all_scalar[:,:,:,:,1].min():.4f}, {all_scalar[:,:,:,:,1].max():.4f}]")
    assert not np.any(np.isnan(all_scalar)), "NaN detected!"
    assert not np.any(np.isinf(all_scalar)), "Inf detected!"

    # At t=0, w should be 0 (sin(0)=0)
    for i in range(min(3, N_SAMPLES)):
        w_t0_max = np.abs(all_scalar[i, 0, :, :, 1]).max()
        print(f"  Sample {i}: w(t=0) max = {w_t0_max:.2e} (should be ~0)")
        assert w_t0_max < 1e-6, f"w(t=0) not zero for sample {i}!"

    print(f"  c range: [{c_values.min():.4f}, {c_values.max():.4f}]")

    # Save HDF5
    print(f"\nSaving to {OUTPUT_PATH}...")
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)

    with h5py.File(OUTPUT_PATH, 'w') as f:
        # Zero vector (required by FinetuneDataset format)
        f.create_dataset('vector', shape=(N_SAMPLES, N_T, N_GRID, N_GRID, 3),
                         dtype=np.float32, fillvalue=0,
                         chunks=(1, 1, N_GRID, N_GRID, 3), compression='gzip', compression_opts=1)
        f.create_dataset('scalar', data=all_scalar, dtype=np.float32,
                         compression='gzip', compression_opts=4)
        f.create_dataset('scalar_indices', data=np.array([0, 1], dtype=np.int64))
        f.create_dataset('nu', data=c_values, dtype=np.float32)

        f.attrs['description'] = '2D Wave Equation (periodic BC, multi-mode, boundary-exclusive grid)'
        f.attrs['pde'] = 'u_tt = c^2 (u_xx + u_yy)'
        f.attrs['domain'] = '[0, 2*pi)^2'
        f.attrs['n_samples'] = N_SAMPLES
        f.attrs['grid_size'] = N_GRID
        f.attrs['dx'] = dx
        f.attrs['dy'] = dx
        f.attrs['dt'] = DT
        f.attrs['T'] = N_T
        f.attrs['c_min'] = C_MIN
        f.attrs['c_max'] = C_MAX
        f.attrs['max_mode'] = MAX_MODE
        f.attrs['grid_type'] = 'boundary_exclusive'

    with h5py.File(OUTPUT_PATH, 'r') as f:
        print(f"  scalar shape: {f['scalar'].shape}")
        print(f"  scalar_indices: {f['scalar_indices'][:]}")
        print(f"  nu shape: {f['nu'].shape}")

    file_size = os.path.getsize(OUTPUT_PATH) / (1024**3)
    print(f"  File size: {file_size:.2f} GB")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)

    return OUTPUT_PATH


if __name__ == "__main__":
    generate_dataset()
