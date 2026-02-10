"""
2D Burgers Equation Dataset Generation

Generates 100 samples of 2D Burgers equation solutions with varying viscosity.
Output format matches PDE Foundation Model requirements:
- vector: [N_samples, T, H, W, 3] - Vx, Vy, Vz (Vz=0 for 2D)
- scalar: empty (no scalar fields)
- scalar_indices: empty array

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
    # Compute the exponential term
    exp_arg = (-4*x + 4*y - t) / (32 * nu)
    exp_term = np.exp(exp_arg)

    # Compute u and v
    u = 0.75 - 0.25 / (1 + exp_term)
    v = 0.75 + 0.25 / (1 + exp_term)

    return u, v


def verify_sample(data, nu, sample_idx):
    """
    Verify the generated sample satisfies mathematical constraints.

    Parameters:
    -----------
    data : ndarray [T, H, W, 3]
        Sample data with [Vx, Vy, Vz]
    nu : float
        Viscosity coefficient
    sample_idx : int
        Sample index for reporting

    Returns:
    --------
    bool : True if all checks pass
    """
    print(f"\n  [Verification for Sample {sample_idx}]")

    u_data = data[..., 0]  # Vx
    v_data = data[..., 1]  # Vy
    vz_data = data[..., 2]  # Vz (should be 0)

    # Check 1: u + v = 1.5 (constant constraint)
    sum_data = u_data + v_data
    error_constraint = np.abs(sum_data - 1.5).max()

    print(f"    Constraint (u+v=1.5) max error: {error_constraint:.2e}")
    assert error_constraint < 1e-10, f"Constraint violation: {error_constraint}"

    # Check 2: Vz should be exactly 0
    vz_error = np.abs(vz_data).max()
    print(f"    Vz (should be 0) max value: {vz_error:.2e}")
    assert vz_error == 0, f"Vz should be 0 but got max={vz_error}"

    # Check 3: Physical bounds
    u_min, u_max = u_data.min(), u_data.max()
    v_min, v_max = v_data.min(), v_data.max()

    print(f"    u range: [{u_min:.4f}, {u_max:.4f}]")
    print(f"    v range: [{v_min:.4f}, {v_max:.4f}]")

    assert 0 <= u_min and u_max <= 1.5, f"u out of bounds: [{u_min}, {u_max}]"
    assert 0 <= v_min and v_max <= 1.5, f"v out of bounds: [{v_min}, {v_max}]"

    # Check 4: Temporal smoothness
    u_diff = np.abs(np.diff(u_data, axis=0)).max()
    v_diff = np.abs(np.diff(v_data, axis=0)).max()

    print(f"    Max temporal change - u: {u_diff:.4e}, v: {v_diff:.4e}")

    print(f"    Sample {sample_idx} (nu={nu:.6f}) passed all checks!")

    return True


def generate_single_sample(nu, x_interior, y_interior, t):
    """
    Generate a single sample with given viscosity.

    Parameters:
    -----------
    nu : float
        Viscosity coefficient
    x_interior : ndarray [W]
        Interior x coordinates (cell centers)
    y_interior : ndarray [H]
        Interior y coordinates (cell centers)
    t : ndarray [T]
        Time coordinates

    Returns:
    --------
    data : ndarray [T, H, W, 3]
        Interior domain data with [Vx, Vy, Vz] where Vz=0
    """
    T_steps, H, W = len(t), len(y_interior), len(x_interior)

    # Initialize array with 3 channels: [Vx, Vy, Vz]
    data = np.zeros((T_steps, H, W, 3), dtype=np.float32)

    # Create meshgrids for interior domain
    X_interior, Y_interior = np.meshgrid(x_interior, y_interior, indexing='xy')

    # Generate interior domain data
    for k in range(T_steps):
        u, v = analytical_solution(X_interior, Y_interior, t[k], nu)
        data[k, :, :, 0] = u  # Vx
        data[k, :, :, 1] = v  # Vy
        # data[k, :, :, 2] = 0  # Vz (already zeros)

    return data


def generate_dataset():
    """
    Main function to generate the complete dataset.
    """
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

    # Create output directory if needed
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Generate viscosity coefficients
    print(f"\n[Step 1/4] Generating viscosity coefficients...")
    np.random.seed(SEED)
    nu_values = np.random.uniform(NU_MIN, NU_MAX, N_SAMPLES)
    nu_values = np.sort(nu_values)  # Sort for organized storage

    print(f"  Generated {N_SAMPLES} viscosity values")
    print(f"    Range: [{nu_values.min():.6f}, {nu_values.max():.6f}]")
    print(f"    First 5: {nu_values[:5]}")
    print(f"    Last 5: {nu_values[-5:]}")

    # Step 2: Create spatial and temporal grids
    print(f"\n[Step 2/4] Creating grids...")

    # Interior domain (cell-centered)
    x_interior = np.linspace(1/256, 255/256, N_X)
    y_interior = np.linspace(1/256, 255/256, N_Y)

    # Temporal grid (uniform)
    t = np.linspace(0, 1, N_T)

    print(f"  Interior x grid: [{x_interior[0]:.8f}, {x_interior[-1]:.8f}]")
    print(f"  Interior y grid: [{y_interior[0]:.8f}, {y_interior[-1]:.8f}]")
    print(f"  Time grid: [{t[0]:.8f}, {t[-1]:.8f}]")
    print(f"  Grid points - Interior: {N_X}x{N_Y}, Time: {N_T}")

    # Step 3: Generate samples
    print(f"\n[Step 3/4] Generating {N_SAMPLES} samples...")

    # Pre-allocate arrays for all samples
    # vector: [N, T, H, W, 3] with channels [Vx, Vy, Vz]
    all_vector = np.zeros((N_SAMPLES, N_T, N_Y, N_X, 3), dtype=np.float32)

    start_time = time.time()

    for sample_idx in tqdm(range(N_SAMPLES), desc="  Progress"):
        nu = nu_values[sample_idx]

        # Generate sample
        data = generate_single_sample(nu, x_interior, y_interior, t)

        # Verify sample
        verify_sample(data, nu, sample_idx)

        # Store in array
        all_vector[sample_idx] = data

    elapsed_time = time.time() - start_time

    print(f"\n  Generated all {N_SAMPLES} samples in {elapsed_time:.2f}s")
    print(f"  Average time per sample: {elapsed_time/N_SAMPLES:.2f}s")

    # Step 4: Save to HDF5 with new format
    print(f"\n[Step 4/4] Saving to HDF5...")

    # Remove existing file if it exists
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        print(f"  Removed existing file: {OUTPUT_FILE}")

    with h5py.File(OUTPUT_FILE, 'w') as f:
        # Save vector field: [N, T, H, W, 3]
        f.create_dataset('vector', data=all_vector, dtype=np.float32,
                        compression='gzip', compression_opts=4)

        # Save empty scalar field (no scalar for Burgers equation)
        # Using shape [N, T, H, W, 0] to indicate no scalar channels
        f.create_dataset('scalar', shape=(N_SAMPLES, N_T, N_Y, N_X, 0), dtype=np.float32)

        # Save scalar_indices as empty array (no scalar channels)
        f.create_dataset('scalar_indices', data=np.array([], dtype=np.int32))

        # Save nu values for reference
        f.create_dataset('nu', data=nu_values, dtype=np.float32)

        # Save metadata
        f.attrs['description'] = '2D Burgers equation with analytical solution'
        f.attrs['n_samples'] = N_SAMPLES
        f.attrs['spatial_resolution'] = (N_Y, N_X)
        f.attrs['temporal_steps'] = N_T
        f.attrs['nu_range'] = (NU_MIN, NU_MAX)
        f.attrs['vector_channels'] = 3  # [Vx, Vy, Vz]
        f.attrs['vector_dim'] = 2  # 2D velocity (Vz=0)
        f.attrs['scalar_channels'] = 0

        print(f"  Saved to: {OUTPUT_FILE}")

    # Final verification and summary
    print(f"\n[Verification] Checking saved file...")

    with h5py.File(OUTPUT_FILE, 'r') as f:
        print(f"  vector shape: {f['vector'].shape}")
        print(f"  scalar shape: {f['scalar'].shape}")
        print(f"  scalar_indices: {f['scalar_indices'][:]}")
        print(f"  nu shape: {f['nu'].shape}")

        # Check vector shape
        assert f['vector'].shape == (N_SAMPLES, N_T, N_Y, N_X, 3), \
            f"Vector shape mismatch: {f['vector'].shape}"

        # Calculate file size
        file_size = os.path.getsize(OUTPUT_FILE) / (1024**3)  # GB
        print(f"\n  File size: {file_size:.2f} GB")

    print("\n" + "="*70)
    print("Dataset generation completed successfully!")
    print("="*70)
    print(f"\nOutput file: {OUTPUT_FILE}")
    print(f"\nFormat specification:")
    print(f"  - vector: [N={N_SAMPLES}, T={N_T}, H={N_Y}, W={N_X}, C=3]")
    print(f"    - Channel 0: Vx (u velocity)")
    print(f"    - Channel 1: Vy (v velocity)")
    print(f"    - Channel 2: Vz (always 0 for 2D)")
    print(f"  - scalar: [N={N_SAMPLES}, T={N_T}, H={N_Y}, W={N_X}, C=0] (empty)")
    print(f"  - scalar_indices: [] (no scalar channels)")
    print(f"  - nu: [{N_SAMPLES}] (viscosity coefficients)")
    print(f"\nTo load the dataset:")
    print(f"  import h5py")
    print(f"  with h5py.File('{OUTPUT_FILE}', 'r') as f:")
    print(f"      vector = f['vector'][:]  # [N, T, H, W, 3]")
    print(f"      nu = f['nu'][:]  # [N]")
    print("\n")


if __name__ == "__main__":
    generate_dataset()
