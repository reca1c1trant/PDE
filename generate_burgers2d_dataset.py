"""
2D Burgers Equation Dataset Generation

Generates 100 samples of 2D Burgers equation solutions with varying viscosity.
Each sample contains:
- Interior domain data: [T=1000, H=128, W=128, C=2]
- Right boundary (x=1.0): [T=1000, H=128, W=1, C=2]
- Top boundary (y=1.0): [T=1000, H=1, W=128, C=2]

Author: Ziye
Date: December 2024
"""

import numpy as np
import h5py
from tqdm import tqdm
import time


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


def verify_sample(data, boundary_left, boundary_right, boundary_bottom, boundary_top, 
                  nu, sample_idx):
    """
    Verify the generated sample satisfies mathematical constraints.
    
    Parameters:
    -----------
    data : ndarray [T, H, W, 2]
        Interior domain data
    boundary_left : ndarray [T, H, 1, 2]
        Left boundary data
    boundary_right : ndarray [T, H, 1, 2]
        Right boundary data
    boundary_bottom : ndarray [T, 1, W, 2]
        Bottom boundary data
    boundary_top : ndarray [T, 1, W, 2]
        Top boundary data
    nu : float
        Viscosity coefficient
    sample_idx : int
        Sample index for reporting
    
    Returns:
    --------
    bool : True if all checks pass
    """
    print(f"\n  [Verification for Sample {sample_idx}]")
    
    # Check 1: u + v = 1.5 (constant constraint)
    u_data = data[..., 0]
    v_data = data[..., 1]
    sum_data = u_data + v_data
    error_constraint = np.abs(sum_data - 1.5).max()
    
    print(f"    ✓ Constraint (u+v=1.5) max error: {error_constraint:.2e}")
    assert error_constraint < 1e-10, f"Constraint violation: {error_constraint}"
    
    # Check 2: Boundary constraint for left boundary
    u_left = boundary_left[..., 0]
    v_left = boundary_left[..., 1]
    sum_left = u_left + v_left
    error_left = np.abs(sum_left - 1.5).max()
    
    print(f"    ✓ Left boundary constraint max error: {error_left:.2e}")
    assert error_left < 1e-10, f"Left boundary violation: {error_left}"
    
    # Check 3: Boundary constraint for right boundary
    u_right = boundary_right[..., 0]
    v_right = boundary_right[..., 1]
    sum_right = u_right + v_right
    error_right = np.abs(sum_right - 1.5).max()
    
    print(f"    ✓ Right boundary constraint max error: {error_right:.2e}")
    assert error_right < 1e-10, f"Right boundary violation: {error_right}"
    
    # Check 4: Boundary constraint for bottom boundary
    u_bottom = boundary_bottom[..., 0]
    v_bottom = boundary_bottom[..., 1]
    sum_bottom = u_bottom + v_bottom
    error_bottom = np.abs(sum_bottom - 1.5).max()
    
    print(f"    ✓ Bottom boundary constraint max error: {error_bottom:.2e}")
    assert error_bottom < 1e-10, f"Bottom boundary violation: {error_bottom}"
    
    # Check 5: Boundary constraint for top boundary
    u_top = boundary_top[..., 0]
    v_top = boundary_top[..., 1]
    sum_top = u_top + v_top
    error_top = np.abs(sum_top - 1.5).max()
    
    print(f"    ✓ Top boundary constraint max error: {error_top:.2e}")
    assert error_top < 1e-10, f"Top boundary violation: {error_top}"
    
    # Check 6: Physical bounds
    u_min, u_max = u_data.min(), u_data.max()
    v_min, v_max = v_data.min(), v_data.max()
    
    print(f"    ✓ u range: [{u_min:.4f}, {u_max:.4f}]")
    print(f"    ✓ v range: [{v_min:.4f}, {v_max:.4f}]")
    
    assert 0 <= u_min and u_max <= 1.5, f"u out of bounds: [{u_min}, {u_max}]"
    assert 0 <= v_min and v_max <= 1.5, f"v out of bounds: [{v_min}, {v_max}]"
    
    # Check 7: Temporal smoothness
    u_diff = np.abs(np.diff(u_data, axis=0)).max()
    v_diff = np.abs(np.diff(v_data, axis=0)).max()
    
    print(f"    ✓ Max temporal change - u: {u_diff:.4e}, v: {v_diff:.4e}")
    
    print(f"    ✓ Sample {sample_idx} (nu={nu:.6f}) passed all checks!")
    
    return True


def generate_single_sample(nu, x_interior, y_interior, t):
    """
    Generate a single sample with given viscosity.
    
    Parameters:
    -----------
    nu : float
        Viscosity coefficient
    x_interior : ndarray [128]
        Interior x coordinates (cell centers)
    y_interior : ndarray [128]
        Interior y coordinates (cell centers)
    t : ndarray [1000]
        Time coordinates
    
    Returns:
    --------
    data : ndarray [1000, 128, 128, 2]
        Interior domain data
    boundary_left : ndarray [1000, 128, 1, 2]
        Left boundary (x=0.0) data
    boundary_right : ndarray [1000, 128, 1, 2]
        Right boundary (x=1.0) data
    boundary_bottom : ndarray [1000, 1, 128, 2]
        Bottom boundary (y=0.0) data
    boundary_top : ndarray [1000, 1, 128, 2]
        Top boundary (y=1.0) data
    """
    T, H, W = len(t), len(y_interior), len(x_interior)
    
    # Initialize arrays
    data = np.zeros((T, H, W, 2), dtype=np.float32)
    boundary_left = np.zeros((T, H, 1, 2), dtype=np.float32)
    boundary_right = np.zeros((T, H, 1, 2), dtype=np.float32)
    boundary_bottom = np.zeros((T, 1, W, 2), dtype=np.float32)
    boundary_top = np.zeros((T, 1, W, 2), dtype=np.float32)
    
    # Create meshgrids for interior domain
    X_interior, Y_interior = np.meshgrid(x_interior, y_interior, indexing='xy')
    
    # Generate interior domain data
    for k in range(T):
        u, v = analytical_solution(X_interior, Y_interior, t[k], nu)
        data[k, :, :, 0] = u
        data[k, :, :, 1] = v
    
    # Generate left boundary (x = 0.0)
    for k in range(T):
        for i in range(H):
            u, v = analytical_solution(0.0, y_interior[i], t[k], nu)
            boundary_left[k, i, 0, 0] = u
            boundary_left[k, i, 0, 1] = v
    
    # Generate right boundary (x = 1.0)
    for k in range(T):
        for i in range(H):
            u, v = analytical_solution(1.0, y_interior[i], t[k], nu)
            boundary_right[k, i, 0, 0] = u
            boundary_right[k, i, 0, 1] = v
    
    # Generate bottom boundary (y = 0.0)
    for k in range(T):
        for j in range(W):
            u, v = analytical_solution(x_interior[j], 0.0, t[k], nu)
            boundary_bottom[k, 0, j, 0] = u
            boundary_bottom[k, 0, j, 1] = v
    
    # Generate top boundary (y = 1.0)
    for k in range(T):
        for j in range(W):
            u, v = analytical_solution(x_interior[j], 1.0, t[k], nu)
            boundary_top[k, 0, j, 0] = u
            boundary_top[k, 0, j, 1] = v
    
    return data, boundary_left, boundary_right, boundary_bottom, boundary_top


def save_to_hdf5(filename, sample_idx, nu, data, boundary_left, boundary_right, 
                 boundary_bottom, boundary_top):
    """
    Save a single sample to HDF5 file.
    
    Parameters:
    -----------
    filename : str
        HDF5 filename
    sample_idx : int
        Sample index (0-99)
    nu : float
        Viscosity coefficient
    data : ndarray
        Interior domain data
    boundary_left : ndarray
        Left boundary data
    boundary_right : ndarray
        Right boundary data
    boundary_bottom : ndarray
        Bottom boundary data
    boundary_top : ndarray
        Top boundary data
    """
    with h5py.File(filename, 'a') as f:
        # Create group for this sample
        grp = f.create_group(str(sample_idx))
        
        # Save viscosity coefficient
        grp.create_dataset('nu', data=nu, dtype=np.float32)
        
        # Scalar field (None for this problem)
        grp.create_dataset('scalar', data=h5py.Empty("f"))
        
        # Create vector group
        vec_grp = grp.create_group('vector')
        
        # Save interior data
        vec_grp.create_dataset('data', data=data, dtype=np.float32)
        
        # Create boundary group
        bnd_grp = vec_grp.create_group('boundary')
        
        # Save all four boundaries
        bnd_grp.create_dataset('left', data=boundary_left, dtype=np.float32)
        bnd_grp.create_dataset('right', data=boundary_right, dtype=np.float32)
        bnd_grp.create_dataset('bottom', data=boundary_bottom, dtype=np.float32)
        bnd_grp.create_dataset('top', data=boundary_top, dtype=np.float32)


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
    OUTPUT_FILE = "burgers2d_nu0.1_0.15_res128_t1000_n100.h5"
    
    print(f"\nConfiguration:")
    print(f"  • Number of samples: {N_SAMPLES}")
    print(f"  • Spatial resolution: {N_X} × {N_Y}")
    print(f"  • Temporal steps: {N_T}")
    print(f"  • Viscosity range: [{NU_MIN}, {NU_MAX}]")
    print(f"  • Random seed: {SEED}")
    print(f"  • Output file: {OUTPUT_FILE}")
    
    # Step 1: Generate viscosity coefficients
    print(f"\n[Step 1/4] Generating viscosity coefficients...")
    np.random.seed(SEED)
    nu_values = np.random.uniform(NU_MIN, NU_MAX, N_SAMPLES)
    nu_values = np.sort(nu_values)  # Sort for organized storage
    
    print(f"  ✓ Generated {N_SAMPLES} viscosity values")
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
    
    print(f"  ✓ Interior x grid: [{x_interior[0]:.8f}, {x_interior[-1]:.8f}]")
    print(f"  ✓ Interior y grid: [{y_interior[0]:.8f}, {y_interior[-1]:.8f}]")
    print(f"  ✓ Time grid: [{t[0]:.8f}, {t[-1]:.8f}]")
    print(f"  ✓ Grid points - Interior: {N_X}×{N_Y}, Time: {N_T}")
    
    # Step 3: Generate samples
    print(f"\n[Step 3/4] Generating {N_SAMPLES} samples...")
    
    # Remove existing file if it exists
    import os
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        print(f"  • Removed existing file: {OUTPUT_FILE}")
    
    start_time = time.time()
    
    for sample_idx in tqdm(range(N_SAMPLES), desc="  Progress"):
        nu = nu_values[sample_idx]
        
        # Generate sample
        data, boundary_left, boundary_right, boundary_bottom, boundary_top = generate_single_sample(
            nu, x_interior, y_interior, t
        )
        
        # Verify sample
        verify_sample(data, boundary_left, boundary_right, boundary_bottom, 
                     boundary_top, nu, sample_idx)
        
        # Save to HDF5
        save_to_hdf5(OUTPUT_FILE, sample_idx, nu, data, 
                     boundary_left, boundary_right, boundary_bottom, boundary_top)
    
    elapsed_time = time.time() - start_time
    
    print(f"\n  ✓ Generated all {N_SAMPLES} samples in {elapsed_time:.2f}s")
    print(f"  ✓ Average time per sample: {elapsed_time/N_SAMPLES:.2f}s")
    
    # Step 4: Final verification and summary
    print(f"\n[Step 4/4] Final verification and summary...")
    
    with h5py.File(OUTPUT_FILE, 'r') as f:
        n_samples_saved = len(f.keys())
        
        # Check first sample structure
        sample_0 = f['0']
        
        print(f"  ✓ Total samples in file: {n_samples_saved}")
        print(f"  ✓ Sample structure:")
        print(f"    - nu: scalar float32")
        print(f"    - scalar: None")
        print(f"    - vector/data: {sample_0['vector']['data'].shape}")
        print(f"    - vector/boundary/right: {sample_0['vector']['boundary']['right'].shape}")
        print(f"    - vector/boundary/top: {sample_0['vector']['boundary']['top'].shape}")
        
        # Calculate file size
        file_size = os.path.getsize(OUTPUT_FILE) / (1024**3)  # GB
        print(f"\n  ✓ File size: {file_size:.2f} GB")
    
    print("\n" + "="*70)
    print("Dataset generation completed successfully!")
    print("="*70)
    print(f"\nOutput file: {OUTPUT_FILE}")
    print(f"\nTo load the dataset:")
    print(f"  import h5py")
    print(f"  with h5py.File('{OUTPUT_FILE}', 'r') as f:")
    print(f"      # Access sample 0")
    print(f"      nu = f['0']['nu'][()]")
    print(f"      data = f['0']['vector']['data'][:]")
    print(f"      boundary_right = f['0']['vector']['boundary']['right'][:]")
    print(f"      boundary_top = f['0']['vector']['boundary']['top'][:]")
    print("\n")


if __name__ == "__main__":
    generate_dataset()