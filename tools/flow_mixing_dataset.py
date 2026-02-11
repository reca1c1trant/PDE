"""
2D Flow Mixing Dataset Generation

Governing equation: ∂u/∂t + a·∂u/∂x + b·∂u/∂y = 0
Analytical solution: u(x,y,t) = -tanh(y/2·cos(ωt) - x/2·sin(ωt))
"""

import numpy as np
import h5py
from tqdm import tqdm
import time
import os


def analytical_solution(x, y, t, vtmax):
    r = np.sqrt(x**2 + y**2)
    r_safe = np.where(r < 1e-10, 1e-10, r)
    v_t = (1.0 / np.cosh(r_safe))**2 * np.tanh(r_safe)
    omega = (v_t / r_safe) / vtmax
    omega = np.where(r < 1e-10, 1.0 / vtmax, omega)
    u = -np.tanh((y / 2.0) * np.cos(omega * t) - (x / 2.0) * np.sin(omega * t))
    return u


def generate_single_sample(vtmax, x_interior, y_interior, t):
    T_len, H, W = len(t), len(y_interior), len(x_interior)
    
    data = np.zeros((T_len, H, W, 1), dtype=np.float32)
    boundary_left = np.zeros((T_len, H, 1, 1), dtype=np.float32)
    boundary_right = np.zeros((T_len, H, 1, 1), dtype=np.float32)
    boundary_bottom = np.zeros((T_len, 1, W, 1), dtype=np.float32)
    boundary_top = np.zeros((T_len, 1, W, 1), dtype=np.float32)
    
    X, Y = np.meshgrid(x_interior, y_interior, indexing='xy')
    
    for k in range(T_len):
        data[k, :, :, 0] = analytical_solution(X, Y, t[k], vtmax)
        boundary_left[k, :, 0, 0] = analytical_solution(0.0, y_interior, t[k], vtmax)
        boundary_right[k, :, 0, 0] = analytical_solution(1.0, y_interior, t[k], vtmax)
        boundary_bottom[k, 0, :, 0] = analytical_solution(x_interior, 0.0, t[k], vtmax)
        boundary_top[k, 0, :, 0] = analytical_solution(x_interior, 1.0, t[k], vtmax)
    
    return data, boundary_left, boundary_right, boundary_bottom, boundary_top


def save_to_hdf5(filename, sample_idx, vtmax, data, boundary_left, boundary_right,
                 boundary_bottom, boundary_top):
    with h5py.File(filename, 'a') as f:
        grp = f.create_group(str(sample_idx))
        grp.create_dataset('vtmax', data=vtmax, dtype=np.float32)
        grp.create_dataset('scalar', data=h5py.Empty("f"))
        
        vec_grp = grp.create_group('vector')
        vec_grp.create_dataset('data', data=data, dtype=np.float32)
        
        bnd_grp = vec_grp.create_group('boundary')
        bnd_grp.create_dataset('left', data=boundary_left, dtype=np.float32)
        bnd_grp.create_dataset('right', data=boundary_right, dtype=np.float32)
        bnd_grp.create_dataset('bottom', data=boundary_bottom, dtype=np.float32)
        bnd_grp.create_dataset('top', data=boundary_top, dtype=np.float32)


def generate_dataset():
    # Configuration
    N_SAMPLES = 150
    N_X, N_Y, N_T = 128, 128, 1000
    VTMAX_MIN, VTMAX_MAX = 0.3, 0.5
    SEED = 42
    OUTPUT_FILE = "flow_mixing_vtmax0.3_0.5_res128_t1000_n150.h5"
    
    print(f"Generating {N_SAMPLES} samples, vtmax ∈ [{VTMAX_MIN}, {VTMAX_MAX}]")
    
    np.random.seed(SEED)
    vtmax_values = np.sort(np.random.uniform(VTMAX_MIN, VTMAX_MAX, N_SAMPLES))
    
    x_interior = np.linspace(1/256, 255/256, N_X)
    y_interior = np.linspace(1/256, 255/256, N_Y)
    t = np.linspace(0, 1, N_T)
    
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
    
    start_time = time.time()
    for i in tqdm(range(N_SAMPLES)):
        data, bl, br, bb, bt = generate_single_sample(vtmax_values[i], x_interior, y_interior, t)
        save_to_hdf5(OUTPUT_FILE, i, vtmax_values[i], data, bl, br, bb, bt)
    
    print(f"Done in {time.time() - start_time:.1f}s, file: {OUTPUT_FILE}")
    print(f"Size: {os.path.getsize(OUTPUT_FILE) / 1024**3:.2f} GB")


if __name__ == "__main__":
    generate_dataset()