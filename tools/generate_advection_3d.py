"""Generate 3D Advection dataset with exact analytical solution.

PDE: u_t + a*u_x + b*u_y + c*u_z = 0

Exact solution (multi-mode Fourier, periodic):
    u(x,y,z,t) = sum_{l,m,n} A_lmn * cos(l*(x-a*t) + m*(y-b*t) + n*(z-c*t) + phi_lmn)

The initial condition is simply translated by the velocity vector (a,b,c).

Domain: [0, 2*pi]^3, periodic BC
Resolution: 64x64x64, dt=0.05, T=21 steps (t=0 to 1.0)
Parameters: (a,b,c) random direction on unit sphere, |v| in [0.5, 2.0]
Channels: scalar=[u], scalar_indices=[11] (ch14, matches CH_U=14 in train script)
vector: all zeros [N, T, X, Y, Z, 3] (required 6D for 3D detection)
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
    a: float,
    b: float,
    c: float,
    modes: list,
    amplitudes: np.ndarray,
    phases: np.ndarray,
) -> np.ndarray:
    """Compute exact multi-mode 3D advection solution.

    u(x,y,z,t) = sum A_i * cos(l*(x-a*t) + m*(y-b*t) + n*(z-c*t) + phi_i)

    Args:
        X, Y, Z: coordinate meshgrids [NX, NY, NZ]
        t: time
        a, b, c: advection velocity components
        modes: list of (l, m, n) integer mode triples
        amplitudes: amplitude for each mode
        phases: phase for each mode

    Returns:
        u field [NX, NY, NZ]
    """
    u = np.zeros_like(X)
    for idx, (l, m, n) in enumerate(modes):
        arg = float(l) * (X - a * t) + float(m) * (Y - b * t) + float(n) * (Z - c * t) + phases[idx]
        u += amplitudes[idx] * np.cos(arg)
    return u


def generate_dataset():
    # Config
    N_GRID = 64
    N_T = 21
    DT = 0.05
    L = 2 * np.pi
    N_SAMPLES = 50

    V_MIN = 0.5
    V_MAX = 2.0

    # Mode candidates: |l|, |m|, |n| <= 2, excluding (0,0,0)
    mode_candidates = []
    for l in range(-2, 3):
        for m in range(-2, 3):
            for n in range(-2, 3):
                if l == 0 and m == 0 and n == 0:
                    continue
                mode_candidates.append((l, m, n))

    OUTPUT_PATH = "/scratch-share/SONG0304/finetune/advection_3d.hdf5"
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Spatial grid (boundary-exclusive for periodic BC)
    x = np.linspace(0, L, N_GRID, endpoint=False)
    y = np.linspace(0, L, N_GRID, endpoint=False)
    z = np.linspace(0, L, N_GRID, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')  # [NX, NY, NZ]

    # Time grid
    t_arr = np.arange(N_T) * DT

    # Random parameters
    rng = np.random.RandomState(42)

    # Per-sample velocity: random direction on unit sphere, random magnitude
    v_magnitudes = rng.uniform(V_MIN, V_MAX, N_SAMPLES).astype(np.float32)
    # Random direction via Gaussian normalized
    raw_dirs = rng.randn(N_SAMPLES, 3)
    norms = np.linalg.norm(raw_dirs, axis=1, keepdims=True)
    unit_dirs = raw_dirs / norms  # [N, 3]

    a_vals = (unit_dirs[:, 0] * v_magnitudes).astype(np.float32)
    b_vals = (unit_dirs[:, 1] * v_magnitudes).astype(np.float32)
    c_vals = (unit_dirs[:, 2] * v_magnitudes).astype(np.float32)

    # For each sample: randomly select 8-15 modes
    N_MODES_MIN = 8
    N_MODES_MAX = 15

    sample_configs = []
    for i in range(N_SAMPLES):
        n_modes = rng.randint(N_MODES_MIN, N_MODES_MAX + 1)
        chosen_indices = rng.choice(len(mode_candidates), size=n_modes, replace=False)
        modes = [mode_candidates[idx] for idx in chosen_indices]

        # Amplitudes with 1/sqrt(l^2+m^2+n^2) decay
        raw_amps = rng.uniform(0.1, 1.0, n_modes).astype(np.float64)
        decayed_amps = np.array([
            raw_amps[j] / np.sqrt(modes[j][0]**2 + modes[j][1]**2 + modes[j][2]**2)
            for j in range(n_modes)
        ])
        phases = rng.uniform(0, 2 * np.pi, n_modes).astype(np.float64)
        sample_configs.append((modes, decayed_amps, phases))

    # Allocate arrays
    # vector: 6D [N, T, X, Y, Z, 3] for 3D detection (all zeros)
    # scalar: [N, T, X, Y, Z, 1]
    scalar_data = np.zeros((N_SAMPLES, N_T, N_GRID, N_GRID, N_GRID, 1), dtype=np.float32)

    print(f"Generating 3D Advection dataset")
    print(f"  Grid: {N_GRID}x{N_GRID}x{N_GRID}, T={N_T}, dt={DT}")
    print(f"  Domain: [0, {L:.4f}]^3")
    print(f"  dx = dy = dz = {L / N_GRID:.6f}")
    print(f"  |v| range: [{V_MIN}, {V_MAX}], N_samples={N_SAMPLES}")
    print(f"  Mode candidates: {len(mode_candidates)}, modes per sample: {N_MODES_MIN}-{N_MODES_MAX}")

    for i in tqdm(range(N_SAMPLES), desc="Generating samples"):
        a_i = float(a_vals[i])
        b_i = float(b_vals[i])
        c_i = float(c_vals[i])
        modes, amplitudes, phases = sample_configs[i]
        for k in range(N_T):
            u = analytical_solution_multimode(
                X, Y, Z, t_arr[k], a_i, b_i, c_i,
                modes, amplitudes, phases,
            )
            scalar_data[i, k, :, :, :, 0] = u

    # Sanity checks
    print("\nSanity checks:")
    print(f"  scalar range: [{scalar_data.min():.6f}, {scalar_data.max():.6f}]")
    print(f"  NaN count: {np.isnan(scalar_data).sum()}")
    print(f"  Inf count: {np.isinf(scalar_data).sum()}")

    # Quick PDE residual check on sample 0
    dx = L / N_GRID
    u_test = scalar_data[0, :, :, :, :, 0].astype(np.float64)  # [T, X, Y, Z]
    t_idx = 10
    u_t = (u_test[t_idx + 1] - u_test[t_idx - 1]) / (2 * DT)
    u_x = (np.roll(u_test[t_idx], -1, axis=0) - np.roll(u_test[t_idx], 1, axis=0)) / (2 * dx)
    u_y = (np.roll(u_test[t_idx], -1, axis=1) - np.roll(u_test[t_idx], 1, axis=1)) / (2 * dx)
    u_z = (np.roll(u_test[t_idx], -1, axis=2) - np.roll(u_test[t_idx], 1, axis=2)) / (2 * dx)
    a0, b0, c0 = float(a_vals[0]), float(b_vals[0]), float(c_vals[0])
    residual = u_t + a0 * u_x + b0 * u_y + c0 * u_z
    rms = np.sqrt(np.mean(residual**2))
    print(f"  Quick PDE residual (sample 0, t={t_idx}, 2nd-order): RMS = {rms:.6e}")

    # Create vector data (all zeros, 6D for 3D detection)
    vector_data = np.zeros((N_SAMPLES, N_T, N_GRID, N_GRID, N_GRID, 3), dtype=np.float32)

    # Save HDF5
    with h5py.File(OUTPUT_PATH, 'w') as f:
        f.create_dataset('vector', data=vector_data, dtype=np.float32)
        f.create_dataset('scalar', data=scalar_data, dtype=np.float32)
        f.create_dataset('scalar_indices', data=np.array([11], dtype=np.int64))  # scalar[11] -> ch14, matches CH_U=14 in train script
        f.create_dataset('nu', data=v_magnitudes, dtype=np.float32)  # |v| stored as "nu"
        f.create_dataset('params_a', data=a_vals, dtype=np.float32)
        f.create_dataset('params_b', data=b_vals, dtype=np.float32)
        f.create_dataset('params_c', data=c_vals, dtype=np.float32)

    file_size = os.path.getsize(OUTPUT_PATH) / 1e9
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"File size: {file_size:.2f} GB")
    print(f"  vector shape: {vector_data.shape}")
    print(f"  scalar shape: {scalar_data.shape}")
    print(f"  scalar_indices: [11]  (-> ch14 in 18-ch layout)")
    print(f"  nu (|v|) shape: {v_magnitudes.shape}, range: [{v_magnitudes.min():.4f}, {v_magnitudes.max():.4f}]")
    print(f"  params_a range: [{a_vals.min():.4f}, {a_vals.max():.4f}]")
    print(f"  params_b range: [{b_vals.min():.4f}, {b_vals.max():.4f}]")
    print(f"  params_c range: [{c_vals.min():.4f}, {c_vals.max():.4f}]")


if __name__ == "__main__":
    generate_dataset()
