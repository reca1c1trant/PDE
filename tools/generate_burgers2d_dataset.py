"""
2D Burgers Equation Dataset Generation (Boundary-Inclusive Grid)

Grid: 128 points from 0 to 1 (including boundaries)
  - Points: 0, 1/127, 2/127, ..., 126/127, 1
  - Spacing: dx = 1/127

With ghost cell extrapolation, PDE loss computed on [1:127, 1:127] (126x126 points).
Boundary points (indices 0, 127) used for ghost cell extrapolation.

Author: Ziye
Date: January 2025
"""

import numpy as np
import h5py
from tqdm import tqdm
import time
import os


def analytical_solution(x: np.ndarray, y: np.ndarray, t: float, nu: float):
    """
    Analytical solution for 2D Burgers equation.

    u_t + u*u_x + v*u_y = nu * (u_xx + u_yy)
    v_t + u*v_x + v*v_y = nu * (v_xx + v_yy)

    Solution satisfies: u + v = 1.5
    """
    exp_arg = (-4*x + 4*y - t) / (32 * nu)
    exp_term = np.exp(exp_arg)

    u = 0.75 - 0.25 / (1 + exp_term)
    v = 0.75 + 0.25 / (1 + exp_term)

    return u, v


def generate_single_sample(nu: float, x_grid: np.ndarray, y_grid: np.ndarray, t: np.ndarray):
    """
    Generate a single sample with given viscosity.

    Returns:
    --------
    data : ndarray [T, H, W, 3]
        Full domain data with [Vx, Vy, Vz] where Vz=0
    """
    T_steps = len(t)
    H, W = len(y_grid), len(x_grid)

    data = np.zeros((T_steps, H, W, 3), dtype=np.float32)

    # Create meshgrid
    X, Y = np.meshgrid(x_grid, y_grid, indexing='xy')

    for k in range(T_steps):
        u, v = analytical_solution(X, Y, t[k], nu)
        data[k, :, :, 0] = u  # Vx
        data[k, :, :, 1] = v  # Vy
        # Vz = 0 (already zeros)

    return data


def verify_sample(data: np.ndarray, nu: float, sample_idx: int) -> bool:
    """Verify the generated sample satisfies mathematical constraints."""
    u_data = data[..., 0]
    v_data = data[..., 1]

    # Check u + v = 1.5
    sum_data = u_data + v_data
    error_constraint = np.abs(sum_data - 1.5).max()
    assert error_constraint < 1e-6, f"Sample {sample_idx}: Constraint violation {error_constraint}"

    # Check physical bounds
    assert u_data.min() >= 0 and u_data.max() <= 1.5, f"Sample {sample_idx}: u out of bounds"
    assert v_data.min() >= 0 and v_data.max() <= 1.5, f"Sample {sample_idx}: v out of bounds"

    return True


def generate_dataset(n_samples: int = 5, output_suffix: str = "_test"):
    """
    Generate dataset.

    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    output_suffix : str
        Suffix for output filename (e.g., "_test" for testing)
    """
    print("=" * 70)
    print("2D Burgers Equation Dataset Generation (Boundary-Inclusive Grid)")
    print("=" * 70)

    # Configuration
    N_SAMPLES = n_samples
    N_GRID = 128  # 128 points including boundaries
    N_T = 1000
    NU_MIN = 0.005
    NU_MAX = 0.02
    SEED = 42

    # Grid spacing
    dx = 1.0 / (N_GRID - 1)  # = 1/127

    # Output path
    OUTPUT_DIR = "/scratch-share/SONG0304/finetune"
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"burgers2d_nu0.1_0.15_n{N_SAMPLES}{output_suffix}.h5")

    print(f"\nConfiguration:")
    print(f"  Number of samples: {N_SAMPLES}")
    print(f"  Spatial resolution: {N_GRID} x {N_GRID}")
    print(f"  Grid spacing: dx = dy = 1/{N_GRID-1} = {dx:.6f}")
    print(f"  Temporal steps: {N_T}")
    print(f"  Viscosity range: [{NU_MIN}, {NU_MAX}]")
    print(f"  Output file: {OUTPUT_FILE}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate viscosity coefficients
    print(f"\n[Step 1/3] Generating viscosity coefficients...")
    np.random.seed(SEED)
    nu_values = np.random.uniform(NU_MIN, NU_MAX, N_SAMPLES).astype(np.float32)
    nu_values = np.sort(nu_values)
    print(f"  nu range: [{nu_values.min():.6f}, {nu_values.max():.6f}]")

    # Create grids (boundary-inclusive)
    print(f"\n[Step 2/3] Creating grids...")
    x_grid = np.linspace(0, 1, N_GRID)  # 0, 1/127, 2/127, ..., 1
    y_grid = np.linspace(0, 1, N_GRID)
    t = np.linspace(0, 1, N_T)

    print(f"  x grid: [{x_grid[0]:.6f}, {x_grid[1]:.6f}, ..., {x_grid[-2]:.6f}, {x_grid[-1]:.6f}]")
    print(f"  y grid: [{y_grid[0]:.6f}, {y_grid[1]:.6f}, ..., {y_grid[-2]:.6f}, {y_grid[-1]:.6f}]")
    print(f"  PDE loss region (with ghost): indices [1:127, 1:127] = [{x_grid[1]:.6f}, {x_grid[126]:.6f}]")

    # Generate samples
    print(f"\n[Step 3/3] Generating {N_SAMPLES} samples...")
    all_vector = np.zeros((N_SAMPLES, N_T, N_GRID, N_GRID, 3), dtype=np.float32)

    start_time = time.time()
    for sample_idx in tqdm(range(N_SAMPLES), desc="  Progress"):
        nu = nu_values[sample_idx]
        data = generate_single_sample(nu, x_grid, y_grid, t)
        verify_sample(data, nu, sample_idx)
        all_vector[sample_idx] = data

    elapsed = time.time() - start_time
    print(f"  Generated in {elapsed:.2f}s")

    # Save to HDF5
    print(f"\n[Saving to HDF5...]")
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    with h5py.File(OUTPUT_FILE, 'w') as f:
        f.create_dataset('vector', data=all_vector, dtype=np.float32,
                        compression='gzip', compression_opts=4)
        f.create_dataset('nu', data=nu_values, dtype=np.float32)

        # Metadata
        f.attrs['description'] = '2D Burgers equation (boundary-inclusive grid)'
        f.attrs['n_samples'] = N_SAMPLES
        f.attrs['grid_size'] = N_GRID
        f.attrs['dx'] = dx
        f.attrs['temporal_steps'] = N_T
        f.attrs['nu_range'] = (NU_MIN, NU_MAX)
        f.attrs['pde_region'] = (1, 127)  # indices for PDE loss (with ghost cell)

    # Verify saved file
    with h5py.File(OUTPUT_FILE, 'r') as f:
        print(f"  vector shape: {f['vector'].shape}")
        print(f"  nu shape: {f['nu'].shape}")

    file_size = os.path.getsize(OUTPUT_FILE) / (1024**2)
    print(f"  File size: {file_size:.2f} MB")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
    print(f"\nOutput: {OUTPUT_FILE}")
    print(f"\nGrid layout (with ghost cell extrapolation):")
    print(f"  Index:    0      1       2      ...    125     126     127")
    print(f"  Coord:    0    1/127   2/127   ...  125/127  126/127    1")
    print(f"            |      |       |              |       |      |")
    print(f"          ghost  PDE    PDE...          PDE     PDE   ghost")

    return OUTPUT_FILE


if __name__ == "__main__":
    # Generate 500 samples for training/validation
    generate_dataset(n_samples=100, output_suffix="")
