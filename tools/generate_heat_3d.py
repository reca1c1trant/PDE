"""Generate 3D Heat dataset with exact analytical solution.

PDE: u_t = nu * (u_xx + u_yy + u_zz)

Exact solution (multi-mode Fourier, periodic BC on [0, 2*pi]^3):
    u(x,y,z,t) = sum_{l,m,n} A_lmn * cos(l*x + m*y + n*z + phi_lmn) * exp(-nu*(l^2+m^2+n^2)*t)

Each Fourier mode decays independently with rate nu*|k|^2, where |k|^2 = l^2 + m^2 + n^2.
This follows directly from the Fourier-space ODE: d/dt u_hat(k,t) = -nu*|k|^2 * u_hat(k,t).

Reference: APEBench Appendix B.1 (arXiv:2411.00180), which explicitly uses periodic BC.

Domain: [0, 2*pi]^3, periodic BC (dx = 2*pi/64)
Resolution: 64x64x64, dt=0.05, T=21 steps (t=0.0 to t=1.0)
Parameters: nu in [0.005, 0.05] (diffusivity), N_SAMPLES=50
Channels: scalar=[u], scalar_indices=[14] (ch17, matches CH_TEMP=17 in train script)
vector: all zeros [N, T, X, Y, Z, 3] (required 6D for 3D detection in dataset_finetune.py)
"""

import numpy as np
import h5py
import os
from tqdm import tqdm


def analytical_solution_multimode(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    t: float,
    nu: float,
    modes: list,
    amplitudes: np.ndarray,
    phases: np.ndarray,
) -> np.ndarray:
    """Compute exact multi-mode 3D heat solution.

    u(x,y,z,t) = sum_i A_i * cos(l_i*x + m_i*y + n_i*z + phi_i) * exp(-nu*(l_i^2+m_i^2+n_i^2)*t)

    Args:
        X, Y, Z: coordinate meshgrids [NX, NY, NZ]
        t: time
        nu: diffusivity
        modes: list of (l, m, n) integer wave numbers
        amplitudes: initial amplitude for each mode [n_modes]
        phases: initial phase for each mode [n_modes]

    Returns:
        u field [NX, NY, NZ]
    """
    u = np.zeros_like(X, dtype=np.float64)
    for idx, (l, m, n) in enumerate(modes):
        k_sq = float(l**2 + m**2 + n**2)
        decay = np.exp(-nu * k_sq * t)
        arg = float(l) * X + float(m) * Y + float(n) * Z + phases[idx]
        u += amplitudes[idx] * decay * np.cos(arg)
    return u


def generate_dataset() -> None:
    # Config
    N_GRID = 64
    N_T = 21
    DT = 0.05
    L = 2 * np.pi
    N_SAMPLES = 50

    NU_MIN = 0.005
    NU_MAX = 0.05

    # Mode candidates: |l|, |m|, |n| <= 2, excluding (0,0,0)
    mode_candidates = []
    for l in range(-2, 3):
        for m in range(-2, 3):
            for n in range(-2, 3):
                if l == 0 and m == 0 and n == 0:
                    continue
                mode_candidates.append((l, m, n))

    OUTPUT_PATH = "/scratch-share/SONG0304/finetune/heat_3d.hdf5"
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Spatial grid (boundary-exclusive for periodic BC)
    x = np.linspace(0, L, N_GRID, endpoint=False)
    y = np.linspace(0, L, N_GRID, endpoint=False)
    z = np.linspace(0, L, N_GRID, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')  # [NX, NY, NZ]

    # Time grid
    t_arr = np.arange(N_T) * DT  # [0.0, 0.05, ..., 1.0]

    rng = np.random.RandomState(42)

    # Per-sample diffusivity
    nu_vals = rng.uniform(NU_MIN, NU_MAX, N_SAMPLES).astype(np.float32)

    # Per-sample mode config
    N_MODES_MIN = 8
    N_MODES_MAX = 15

    sample_configs = []
    for i in range(N_SAMPLES):
        n_modes = rng.randint(N_MODES_MIN, N_MODES_MAX + 1)
        chosen_indices = rng.choice(len(mode_candidates), size=n_modes, replace=False)
        modes = [mode_candidates[idx] for idx in chosen_indices]

        # Amplitudes decayed by 1/sqrt(l^2+m^2+n^2) for physical energy spectrum
        raw_amps = rng.uniform(0.1, 1.0, n_modes).astype(np.float64)
        decayed_amps = np.array([
            raw_amps[j] / np.sqrt(modes[j][0]**2 + modes[j][1]**2 + modes[j][2]**2)
            for j in range(n_modes)
        ])
        phases = rng.uniform(0, 2 * np.pi, n_modes).astype(np.float64)
        sample_configs.append((modes, decayed_amps, phases))

    # Allocate output arrays
    scalar_data = np.zeros((N_SAMPLES, N_T, N_GRID, N_GRID, N_GRID, 1), dtype=np.float32)

    print("Generating 3D Heat dataset")
    print(f"  Grid: {N_GRID}^3, T={N_T}, dt={DT}, t_total={t_arr[-1]:.2f}")
    print(f"  Domain: [0, {L:.4f}]^3,  dx = {L / N_GRID:.6f}")
    print(f"  nu range: [{NU_MIN}, {NU_MAX}], N_samples={N_SAMPLES}")
    print(f"  Mode candidates: {len(mode_candidates)}, modes per sample: {N_MODES_MIN}-{N_MODES_MAX}")
    print(f"  Max decay at t=1: exp(-nu_max * k_max^2) = exp(-{NU_MAX}*12) = {np.exp(-NU_MAX*12):.4f}")

    for i in tqdm(range(N_SAMPLES), desc="Generating samples"):
        nu_i = float(nu_vals[i])
        modes, amplitudes, phases = sample_configs[i]
        for k_t in range(N_T):
            u = analytical_solution_multimode(
                X, Y, Z, t_arr[k_t], nu_i,
                modes, amplitudes, phases,
            )
            scalar_data[i, k_t, :, :, :, 0] = u.astype(np.float32)

    # Sanity checks
    print("\nSanity checks:")
    print(f"  scalar range: [{scalar_data.min():.6f}, {scalar_data.max():.6f}]")
    print(f"  NaN count: {np.isnan(scalar_data).sum()}")
    print(f"  Inf count: {np.isinf(scalar_data).sum()}")

    # Quick PDE residual check: u_t ≈ nu * (u_xx + u_yy + u_zz)
    dx = L / N_GRID
    u_test = scalar_data[0, :, :, :, :, 0].astype(np.float64)  # [T, X, Y, Z]
    nu0 = float(nu_vals[0])
    t_idx = 10  # interior time step

    u_t = (u_test[t_idx + 1] - u_test[t_idx - 1]) / (2 * DT)
    u_lap = (
        (np.roll(u_test[t_idx], -1, axis=0) - 2 * u_test[t_idx] + np.roll(u_test[t_idx], 1, axis=0)) / dx**2
        + (np.roll(u_test[t_idx], -1, axis=1) - 2 * u_test[t_idx] + np.roll(u_test[t_idx], 1, axis=1)) / dx**2
        + (np.roll(u_test[t_idx], -1, axis=2) - 2 * u_test[t_idx] + np.roll(u_test[t_idx], 1, axis=2)) / dx**2
    )
    residual = u_t - nu0 * u_lap
    rms = np.sqrt(np.mean(residual**2))
    u_rms = np.sqrt(np.mean(u_test[t_idx]**2))
    print(f"  PDE residual (sample 0, t_idx={t_idx}, 2nd-order FD): RMS = {rms:.6e}, relative = {rms/u_rms:.6e}")

    # Vector data (all zeros, 6D for 3D detection)
    vector_data = np.zeros((N_SAMPLES, N_T, N_GRID, N_GRID, N_GRID, 3), dtype=np.float32)

    # Save HDF5
    print(f"\nSaving to {OUTPUT_PATH}...")
    with h5py.File(OUTPUT_PATH, 'w') as f:
        f.create_dataset('vector', data=vector_data, dtype=np.float32)
        f.create_dataset('scalar', data=scalar_data, dtype=np.float32)
        f.create_dataset('scalar_indices', data=np.array([14], dtype=np.int64))  # scalar[14] -> ch17, matches CH_TEMP=17 in train script
        f.create_dataset('nu', data=nu_vals, dtype=np.float32)

    file_size = os.path.getsize(OUTPUT_PATH) / 1e9
    print(f"Saved: {OUTPUT_PATH}")
    print(f"File size: {file_size:.2f} GB")
    print(f"  vector shape: {vector_data.shape}")
    print(f"  scalar shape: {scalar_data.shape}")
    print(f"  scalar_indices: [14]  (-> ch17 in 18-ch layout)")
    print(f"  nu shape: {nu_vals.shape}, range: [{nu_vals.min():.4f}, {nu_vals.max():.4f}]")


if __name__ == "__main__":
    generate_dataset()
