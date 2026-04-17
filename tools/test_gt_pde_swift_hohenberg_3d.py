"""
GT PDE Residual Verification for 3D Swift-Hohenberg (APEBench).

PDE: u_t = r*u - (k + Δ)²*u + u² - u³

Expanded: u_t = r*u - k²*u - 2k*Δu - Δ²u + u² - u³
       = (r - k²)*u - 2k*Δu - Δ²u + u² - u³

With r=0.7, k=1.0:
       u_t = -0.3*u - 2*Δu - Δ²u + u² - u³

Where Δ = ∂²/∂x² + ∂²/∂y² + ∂²/∂z² (Laplacian)
      Δ² = biharmonic operator

Domain: [0, 10π]³, periodic BC, N=32, dx=10π/32
dt = 0.1, num_substeps=5 → effective dt per snapshot = 0.1

Usage:
    CUDA_VISIBLE_DEVICES=0 python tools/test_gt_pde_swift_hohenberg_3d.py
"""

import torch
import h5py
import numpy as np
import os

DATA_PATH = "/scratch-share/SONG0304/finetune/swift_hohenberg_3d_apebench.hdf5"

# Physics parameters
r = 0.7
k = 1.0
L = 10 * np.pi  # domain extent
N = 32
dx = L / N  # boundary-exclusive grid: N points on [0, L)
dt = 0.1  # time between snapshots

print("=" * 60)
print("3D Swift-Hohenberg GT PDE Residual")
print("=" * 60)
print(f"  r={r}, k={k}, L={L:.4f}, N={N}, dx={dx:.6f}, dt={dt}")
print(f"  Linear part: (r-k²)u - 2k·Δu - Δ²u = {r-k**2}·u - {2*k}·Δu - Δ²u")
print("=" * 60)

# Load data
with h5py.File(DATA_PATH, 'r') as f:
    scalar = torch.tensor(f['scalar'][:], dtype=torch.float64)  # [30, 201, 32, 32, 32, 1]

u_all = scalar[..., 0]  # [30, 201, 32, 32, 32]
N_samples = u_all.shape[0]
T_total = u_all.shape[1]
print(f"\n  Data: {N_samples} samples, T={T_total}, shape={u_all.shape}")
print(f"  Range: [{u_all.min():.4f}, {u_all.max():.4f}]")

# FD operators (periodic, 3D)
def laplacian_3d(f, dx):
    """2nd-order central Laplacian, periodic."""
    lap = torch.zeros_like(f)
    for dim in [-3, -2, -1]:
        lap += (torch.roll(f, -1, dims=dim) - 2*f + torch.roll(f, 1, dims=dim)) / dx**2
    return lap

def biharmonic_3d(f, dx):
    """Biharmonic Δ² = Laplacian of Laplacian, periodic."""
    return laplacian_3d(laplacian_3d(f, dx), dx)

# Use first N_test samples
N_test = min(10, N_samples)
data = u_all[:N_test]  # [N_test, T, 32, 32, 32]

# Time derivative: 2nd-order central
u_mid = data[:, 1:-1]  # [N, T-2, 32, 32, 32]
du_dt = (data[:, 2:] - data[:, :-2]) / (2 * dt)  # [N, T-2, 32, 32, 32]

# Spatial terms at mid-timesteps
lap_u = laplacian_3d(u_mid, dx)
bilap_u = biharmonic_3d(u_mid, dx)

# PDE residual: du/dt - [(r-k²)*u - 2k*Δu - Δ²u + u² - u³] = 0
# = du/dt - (r-k²)*u + 2k*Δu + Δ²u - u² + u³
linear_coeff = r - k**2  # = -0.3
R = du_dt - linear_coeff * u_mid + 2*k * lap_u + bilap_u - u_mid**2 + u_mid**3

# Wait, sign check:
# PDE: u_t = (r-k²)*u - 2k*Δu - Δ²u + u² - u³
# Residual: du/dt - RHS = 0
# R = du/dt - [(r-k²)*u - 2k*Δu - Δ²u + u² - u³]
# R = du/dt - (r-k²)*u + 2k*Δu + Δ²u - u² + u³
# That's what we have. Good.

# Per-sample RMS
print(f"\nPer-sample RMS ({N_test} samples):")
sample_rms = []
for i in range(N_test):
    rms = torch.sqrt(torch.mean(R[i]**2)).item()
    sample_rms.append(rms)
    print(f"  Sample {i}: {rms:.6e}")

mean_rms = np.mean(sample_rms)

print(f"\n{'='*60}")
print(f"Overall: mean RMS = {mean_rms:.6e}")
print(f"{'='*60}")

# Per-timestep RMS
rms_per_t = torch.sqrt(torch.mean(R**2, dim=(0, 2, 3, 4)))  # [T-2]
print(f"\nPer-timestep RMS (first 5 / last 5):")
for t in range(min(5, len(rms_per_t))):
    print(f"  t={t+1}: {rms_per_t[t]:.6e}")
print("  ...")
for t in range(max(0, len(rms_per_t)-5), len(rms_per_t)):
    print(f"  t={t+1}: {rms_per_t[t]:.6e}")

print(f"\n  eq_scales:")
print(f"    swift_hohenberg: {mean_rms:.4e}")

# Save per-timestep scales
gt_dir = "./data/gt_scales"
os.makedirs(gt_dir, exist_ok=True)
per_t_path = os.path.join(gt_dir, "swift_hohenberg_3d_per_t.pt")
torch.save({'swift_hohenberg': rms_per_t.float()}, per_t_path)
print(f"\n  Saved per-timestep scales to {per_t_path}")
print(f"    shape={rms_per_t.shape}, mean={rms_per_t.mean():.6e}")

print("\nDone!")
