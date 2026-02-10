"""
2D Burgers Equation Dataset Generation

Generates 100 samples of 2D Burgers equation solutions with varying viscosity.
Output format matches PDE Foundation Model requirements:
- vector: [N_samples, T, H, W, 3] - Vx, Vy, Vz (Vz=0 for 2D)
- boundary_left/right/bottom/top: for PDE loss ghost cell padding
- scalar: empty (no scalar fields)

Author: Ziye
Date: December 2024
"""

import numpy as np
import h5py
from tqdm import tqdm
import time
import os


def analytical_solution(x, y, t, nu):
    """
    Analytical solution for 2D Burgers equation.

    Parameters:
    -----------
    x, y, t : float or ndarray
        Spatial and temporal coordinates
    nu : float
        Kinematic viscosity

    Returns:
    --------
    u, v : float or ndarray
        Velocity components
    """
    exp_arg = (-4*x + 4*y - t) / (32 * nu)
    exp_term = np.exp(exp_arg)

    u = 0.75 - 0.25 / (1 + exp_term)
    v = 0.75 + 0.25 / (1 + exp_term)

    return u, v


def generate_single_sample(nu, x_interior, y_interior, t):
    """
    Generate a single sample with given viscosity.

    Returns:
    --------
    data : ndarray [T, H, W, 3]
        Interior domain data with [Vx, Vy, Vz] where Vz=0
    boundary_left : ndarray [T, H, 1, 2]
        Left boundary (x=0, domain edge) with [u, v] for ghost cell
    boundary_right : ndarray [T, H, 1, 2]
        Right boundary (x=1, domain edge) with [u, v] for ghost cell
    boundary_bottom : ndarray [T, 1, W, 2]
        Bottom boundary (y=0, domain edge) with [u, v] for ghost cell
    boundary_top : ndarray [T, 1, W, 2]
        Top boundary (y=1, domain edge) with [u, v] for ghost cell
    """
    T_steps, H, W = len(t), len(y_interior), len(x_interior)

    # Initialize arrays
    data = np.zeros((T_steps, H, W, 3), dtype=np.float32)
    boundary_left = np.zeros((T_steps, H, 1, 2), dtype=np.float32)
    boundary_right = np.zeros((T_steps, H, 1, 2), dtype=np.float32)
    boundary_bottom = np.zeros((T_steps, 1, W, 2), dtype=np.float32)
    boundary_top = np.zeros((T_steps, 1, W, 2), dtype=np.float32)

    # Create meshgrids for interior domain
    X_interior, Y_interior = np.meshgrid(x_interior, y_interior, indexing='xy')

    # Generate data for each timestep
    for k in range(T_steps):
        # Interior domain
        u, v = analytical_solution(X_interior, Y_interior, t[k], nu)
        data[k, :, :, 0] = u  # Vx
        data[k, :, :, 1] = v  # Vy
        # data[k, :, :, 2] = 0  # Vz (already zeros)

        # Left boundary (x=0, domain boundary for ghost cell)
        for i in range(H):
            u_bnd, v_bnd = analytical_solution(0.0, y_interior[i], t[k], nu)
            boundary_left[k, i, 0, 0] = u_bnd
            boundary_left[k, i, 0, 1] = v_bnd

        # Right boundary (x=1, domain boundary for ghost cell)
        for i in range(H):
            u_bnd, v_bnd = analytical_solution(1.0, y_interior[i], t[k], nu)
            boundary_right[k, i, 0, 0] = u_bnd
            boundary_right[k, i, 0, 1] = v_bnd

        # Bottom boundary (y=0, domain boundary for ghost cell)
        for j in range(W):
            u_bnd, v_bnd = analytical_solution(x_interior[j], 0.0, t[k], nu)
            boundary_bottom[k, 0, j, 0] = u_bnd
            boundary_bottom[k, 0, j, 1] = v_bnd

        # Top boundary (y=1, domain boundary for ghost cell)
        for j in range(W):
            u_bnd, v_bnd = analytical_solution(x_interior[j], 1.0, t[k], nu)
            boundary_top[k, 0, j, 0] = u_bnd
            boundary_top[k, 0, j, 1] = v_bnd

    return data, boundary_left, boundary_right, boundary_bottom, boundary_top


def verify_sample(data, nu, sample_idx):
    """Verify the generated sample satisfies mathematical constraints."""
    u_data = data[..., 0]
    v_data = data[..., 1]

    # Check u + v = 1.5
    sum_data = u_data + v_data
    error_constraint = np.abs(sum_data - 1.5).max()
    assert error_constraint < 1e-10, f"Constraint violation: {error_constraint}"

    # Check physical bounds
    assert 0 <= u_data.min() and u_data.max() <= 1.5
    assert 0 <= v_data.min() and v_data.max() <= 1.5

    return True


def generate_dataset():
    """Main function to generate the complete dataset."""
    print("="*70)
    print("2D Burgers Equation Dataset Generation")
    print("="*70)

    # Configuration
    N_SAMPLES = 100
    N_X = 128
    N_Y = 128
    N_T = 1000
    NU_MIN = 0.1
    NU_MAX = 0.15
    SEED = 42

    # Output path
    OUTPUT_DIR = "/scratch-share/SONG0304/finetune"
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "burgers2d_nu0.1_0.15_res128_t1000_n100.h5")

    print(f"\nConfiguration:")
    print(f"  Number of samples: {N_SAMPLES}")
    print(f"  Spatial resolution: {N_X} x {N_Y}")
    print(f"  Temporal steps: {N_T}")
    print(f"  Viscosity range: [{NU_MIN}, {NU_MAX}]")
    print(f"  Random seed: {SEED}")
    print(f"  Output file: {OUTPUT_FILE}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Generate viscosity coefficients
    print(f"\n[Step 1/4] Generating viscosity coefficients...")
    np.random.seed(SEED)
    nu_values = np.random.uniform(NU_MIN, NU_MAX, N_SAMPLES)
    nu_values = np.sort(nu_values)

    print(f"  Generated {N_SAMPLES} viscosity values")
    print(f"    Range: [{nu_values.min():.6f}, {nu_values.max():.6f}]")

    # Step 2: Create grids
    print(f"\n[Step 2/4] Creating grids...")

    # Interior domain (cell-centered): [1/256, 3/256, ..., 255/256]
    x_interior = np.linspace(1/256, 255/256, N_X)
    y_interior = np.linspace(1/256, 255/256, N_Y)
    t = np.linspace(0, 1, N_T)

    print(f"  Interior x grid: [{x_interior[0]:.8f}, {x_interior[-1]:.8f}]")
    print(f"  Interior y grid: [{y_interior[0]:.8f}, {y_interior[-1]:.8f}]")
    print(f"  Boundary: domain edges (x=0, x=1, y=0, y=1) for PDE ghost cell")

    # Step 3: Generate samples
    print(f"\n[Step 3/4] Generating {N_SAMPLES} samples...")

    # Pre-allocate arrays
    all_vector = np.zeros((N_SAMPLES, N_T, N_Y, N_X, 3), dtype=np.float32)
    all_bnd_left = np.zeros((N_SAMPLES, N_T, N_Y, 1, 2), dtype=np.float32)
    all_bnd_right = np.zeros((N_SAMPLES, N_T, N_Y, 1, 2), dtype=np.float32)
    all_bnd_bottom = np.zeros((N_SAMPLES, N_T, 1, N_X, 2), dtype=np.float32)
    all_bnd_top = np.zeros((N_SAMPLES, N_T, 1, N_X, 2), dtype=np.float32)

    start_time = time.time()

    for sample_idx in tqdm(range(N_SAMPLES), desc="  Progress"):
        nu = nu_values[sample_idx]

        # Generate sample with boundaries
        data, bnd_left, bnd_right, bnd_bottom, bnd_top = generate_single_sample(
            nu, x_interior, y_interior, t
        )

        # Verify
        verify_sample(data, nu, sample_idx)

        # Store
        all_vector[sample_idx] = data
        all_bnd_left[sample_idx] = bnd_left
        all_bnd_right[sample_idx] = bnd_right
        all_bnd_bottom[sample_idx] = bnd_bottom
        all_bnd_top[sample_idx] = bnd_top

    elapsed_time = time.time() - start_time
    print(f"\n  Generated all {N_SAMPLES} samples in {elapsed_time:.2f}s")

    # Step 4: Save to HDF5
    print(f"\n[Step 4/4] Saving to HDF5...")

    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        print(f"  Removed existing file")

    with h5py.File(OUTPUT_FILE, 'w') as f:
        # Interior data: [N, T, H, W, 3]
        f.create_dataset('vector', data=all_vector, dtype=np.float32,
                        compression='gzip', compression_opts=4)

        # Boundary data for PDE loss
        f.create_dataset('boundary_left', data=all_bnd_left, dtype=np.float32,
                        compression='gzip', compression_opts=4)
        f.create_dataset('boundary_right', data=all_bnd_right, dtype=np.float32,
                        compression='gzip', compression_opts=4)
        f.create_dataset('boundary_bottom', data=all_bnd_bottom, dtype=np.float32,
                        compression='gzip', compression_opts=4)
        f.create_dataset('boundary_top', data=all_bnd_top, dtype=np.float32,
                        compression='gzip', compression_opts=4)

        # Empty scalar (no scalar for Burgers)
        f.create_dataset('scalar', shape=(N_SAMPLES, N_T, N_Y, N_X, 0), dtype=np.float32)
        f.create_dataset('scalar_indices', data=np.array([], dtype=np.int32))

        # Nu values
        f.create_dataset('nu', data=nu_values, dtype=np.float32)

        # Metadata
        f.attrs['description'] = '2D Burgers equation with analytical solution'
        f.attrs['n_samples'] = N_SAMPLES
        f.attrs['spatial_resolution'] = (N_Y, N_X)
        f.attrs['temporal_steps'] = N_T
        f.attrs['nu_range'] = (NU_MIN, NU_MAX)
        f.attrs['vector_channels'] = 3
        f.attrs['vector_dim'] = 2
        f.attrs['has_boundary'] = True

    # Verification
    print(f"\n[Verification]")
    with h5py.File(OUTPUT_FILE, 'r') as f:
        print(f"  vector: {f['vector'].shape}")
        print(f"  boundary_left: {f['boundary_left'].shape}")
        print(f"  boundary_right: {f['boundary_right'].shape}")
        print(f"  boundary_bottom: {f['boundary_bottom'].shape}")
        print(f"  boundary_top: {f['boundary_top'].shape}")
        print(f"  nu: {f['nu'].shape}")

    file_size = os.path.getsize(OUTPUT_FILE) / (1024**3)
    print(f"\n  File size: {file_size:.2f} GB")

    print("\n" + "="*70)
    print("Dataset generation completed!")
    print("="*70)
    print(f"\nOutput: {OUTPUT_FILE}")
    print(f"\nFormat:")
    print(f"  - vector: [N, T, H, W, 3] (Vx, Vy, Vz=0)")
    print(f"  - boundary_left: [N, T, H, 1, 2] (x=0, for ghost cell)")
    print(f"  - boundary_right: [N, T, H, 1, 2] (x=1, for ghost cell)")
    print(f"  - boundary_bottom: [N, T, 1, W, 2] (y=0, for ghost cell)")
    print(f"  - boundary_top: [N, T, 1, W, 2] (y=1, for ghost cell)")
    print(f"  - nu: [N] (viscosity)")


if __name__ == "__main__":
    generate_dataset()
