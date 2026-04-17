"""Generate 2D Advection-Diffusion dataset with exact analytical solution.

PDE: u_t + a*u_x + b*u_y = nu*(u_xx + u_yy)

Exact solution (multi-mode Fourier, periodic):
    u(x,y,t) = sum_{m,n} A_mn * exp(-nu*(km^2+kn^2)*t)
               * cos(km*(x-a*t) + kn*(y-b*t) + phi_mn)

where km = m, kn = n (since L = 2*pi, so k = 2*pi*m / L = m).

Domain: [0, 2*pi]^2, periodic BC
Resolution: 256x256, dt=0.01, T=101 steps
Parameters: a=1.0 (fixed), b=0.5 (fixed), nu in [0.01, 0.1], ~150 samples
Channels: scalar=[u], scalar_indices=[0] (ch3)
"""

import numpy as np
import h5py
import os
from tqdm import tqdm


def analytical_solution_multimode(X, Y, t, nu, a, b, modes, amplitudes, phases):
    """Compute exact multi-mode advection-diffusion solution.

    Args:
        X: x-coordinates meshgrid [H, W]
        Y: y-coordinates meshgrid [H, W]
        t: time
        nu: diffusion coefficient
        a: advection velocity in x
        b: advection velocity in y
        modes: list of (m, n) integer mode pairs
        amplitudes: amplitude for each mode [N_modes]
        phases: phase for each mode [N_modes]

    Returns:
        u field [H, W]
    """
    u = np.zeros_like(X)
    for idx, (m, n) in enumerate(modes):
        km = float(m)
        kn = float(n)
        A_mn = amplitudes[idx]
        phi_mn = phases[idx]

        decay = np.exp(-nu * (km**2 + kn**2) * t)
        arg = km * (X - a * t) + kn * (Y - b * t) + phi_mn
        u += A_mn * decay * np.cos(arg)

    return u


def generate_dataset():
    # Config
    N_GRID = 256
    N_T = 101
    DT = 0.01
    L = 2 * np.pi
    N_SAMPLES = 150
    NU_MIN = 0.01
    NU_MAX = 0.1

    # Fixed advection velocities
    A_VEL = 1.0
    B_VEL = 0.5

    # Mode range: |m|, |n| <= 3, excluding (0,0)
    mode_candidates = []
    for m in range(-3, 4):
        for n in range(-3, 4):
            if m == 0 and n == 0:
                continue
            mode_candidates.append((m, n))

    OUTPUT_PATH = "/scratch-share/SONG0304/finetune/advdiff_2d.hdf5"
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Spatial grid (boundary-exclusive for periodic BC)
    x = np.linspace(0, L, N_GRID, endpoint=False)
    y = np.linspace(0, L, N_GRID, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='xy')

    # Time grid
    t_arr = np.arange(N_T) * DT

    # Random parameters
    rng = np.random.RandomState(42)
    nu_values = rng.uniform(NU_MIN, NU_MAX, N_SAMPLES).astype(np.float32)

    # For each sample: randomly select 8-15 modes
    N_MODES_MIN = 8
    N_MODES_MAX = 15

    sample_configs = []
    for i in range(N_SAMPLES):
        n_modes = rng.randint(N_MODES_MIN, N_MODES_MAX + 1)
        chosen_indices = rng.choice(len(mode_candidates), size=n_modes, replace=False)
        modes = [mode_candidates[idx] for idx in chosen_indices]
        amplitudes = rng.uniform(0.1, 1.0, n_modes).astype(np.float64)
        phases = rng.uniform(0, 2 * np.pi, n_modes).astype(np.float64)
        sample_configs.append((modes, amplitudes, phases))

    # Allocate arrays
    scalar_data = np.zeros((N_SAMPLES, N_T, N_GRID, N_GRID, 1), dtype=np.float32)

    print(f"Generating Advection-Diffusion 2D dataset")
    print(f"  Grid: {N_GRID}x{N_GRID}, T={N_T}, dt={DT}")
    print(f"  Domain: [0, {L:.4f}]^2")
    print(f"  dx = dy = {L / N_GRID:.6f}")
    print(f"  Advection: a={A_VEL}, b={B_VEL} (fixed)")
    print(f"  nu range: [{NU_MIN}, {NU_MAX}], N_samples={N_SAMPLES}")
    print(f"  Modes per sample: {N_MODES_MIN}-{N_MODES_MAX}")

    for i in tqdm(range(N_SAMPLES), desc="Generating samples"):
        nu = float(nu_values[i])
        modes, amplitudes, phases = sample_configs[i]
        for k in range(N_T):
            u = analytical_solution_multimode(
                X, Y, t_arr[k], nu, A_VEL, B_VEL,
                modes, amplitudes, phases,
            )
            scalar_data[i, k, :, :, 0] = u

    # Sanity checks
    print("\nSanity checks:")
    print(f"  scalar range: [{scalar_data.min():.6f}, {scalar_data.max():.6f}]")
    print(f"  NaN count: {np.isnan(scalar_data).sum()}")
    print(f"  Inf count: {np.isinf(scalar_data).sum()}")

    # Quick PDE residual check
    dx = L / N_GRID
    u_test = scalar_data[0, :, :, :, 0].astype(np.float64)
    t_idx = 50
    u_t = (u_test[t_idx + 1] - u_test[t_idx - 1]) / (2 * DT)
    u_x = (np.roll(u_test[t_idx], -1, axis=1) - np.roll(u_test[t_idx], 1, axis=1)) / (2 * dx)
    u_y = (np.roll(u_test[t_idx], -1, axis=0) - np.roll(u_test[t_idx], 1, axis=0)) / (2 * dx)
    u_xx = (np.roll(u_test[t_idx], -1, axis=1) - 2 * u_test[t_idx]
            + np.roll(u_test[t_idx], 1, axis=1)) / (dx**2)
    u_yy = (np.roll(u_test[t_idx], -1, axis=0) - 2 * u_test[t_idx]
            + np.roll(u_test[t_idx], 1, axis=0)) / (dx**2)
    nu0 = float(nu_values[0])
    residual = u_t + A_VEL * u_x + B_VEL * u_y - nu0 * (u_xx + u_yy)
    rms = np.sqrt(np.mean(residual**2))
    print(f"  Quick PDE residual (sample 0, t={t_idx}, 2nd-order): RMS = {rms:.6e}")

    # Save HDF5
    with h5py.File(OUTPUT_PATH, 'w') as f:
        f.create_dataset('scalar', data=scalar_data, dtype=np.float32)
        f.create_dataset('scalar_indices', data=np.array([0], dtype=np.int64))
        f.create_dataset('nu', data=nu_values, dtype=np.float32)
        f.attrs['a'] = A_VEL
        f.attrs['b'] = B_VEL

    file_size = os.path.getsize(OUTPUT_PATH) / 1e9
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"File size: {file_size:.2f} GB")
    print(f"  scalar shape: {scalar_data.shape}")
    print(f"  scalar_indices: [0]")
    print(f"  nu shape: {nu_values.shape}")


if __name__ == "__main__":
    generate_dataset()
