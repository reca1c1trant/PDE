"""
GT PDE Residual Verification for 3D Burgers (APEBench).

PDE (conservative form, as used by APEBench):
    u_t + b₁ * 1/2 * ∇·(u ⊗ u) = Σ_j α_j * (∇²)^(j/2) u

In component form (conservative, 3D):
    u_t + β₁ * 1/2 * [∂(uu)/∂x + ∂(uv)/∂y + ∂(uw)/∂z] = α₂ * ∇²u
    v_t + β₁ * 1/2 * [∂(vu)/∂x + ∂(vv)/∂y + ∂(vw)/∂z] = α₂ * ∇²v
    w_t + β₁ * 1/2 * [∂(wu)/∂x + ∂(wv)/∂y + ∂(ww)/∂z] = α₂ * ∇²w

Coefficient conversion from APEBench difficulty factors:
    γ = (0, 0, 1.5, 0, 0), δ_conv = -1.5
    N = 32, D = 3, M = 1.0, L = 1.0, Δt = 1.0

    Linear (j=2, diffusion):
        α₂ = γ₂ / (N² · 2^(2-1) · D) = 10^1.5 / (32² · 2 · 3) = 31.623 / 6144 ≈ 5.148e-3

    Convection:
        Note: APEBench passes -δ_conv to the stepper (see _convection.py line 36)
        So stepper receives convection_difficulty = -(-1.5) = 1.5
        β₁ = δ / (M · N · D) = 1.5 / (1.0 · 32 · 3) = 1.5 / 96 ≈ 0.015625

    The PDE in normalized space (L=1, Δt=1):
        u^{n+1} = u^n + β₁ * 1/2 * ∇·(u ⊗ u) + α₂ * ∇²u

    In continuous form for residual checking:
        u_t = -β₁ * 1/2 * ∇·(u ⊗ u) + α₂ * ∇²u

    Where ∇ operates on [0,1]³ with dx = 1/N = 1/32.

Usage:
    CUDA_VISIBLE_DEVICES=0 python tools/test_gt_pde_burgers_3d_apebench.py
"""

import torch
import h5py
import numpy as np

DATA_PATH = "/scratch-share/SONG0304/finetune/burgers_3d_apebench.hdf5"

# APEBench parameters
N = 32
D = 3  # spatial dims
M = 1.0  # max absolute
GAMMA_2 = 1.5  # diffusion difficulty (actually 10^1.5? No, gamma=1.5 directly)
DELTA_CONV = -1.5  # convection difficulty (from scenario)

# Wait - the gammas in the scenario are (0, 0, 1.5, 0, 0)
# But gamma is the DIFFICULTY, not 10^gamma.
# From the code: gammas = (0.0, 0.0, self.diffusion_gamma, 0.0, 0.0)
# where diffusion_gamma = 1.5
# So gamma_2 = 1.5 (NOT 10^1.5)

# Linear coefficient (diffusion):
# α₂ = γ₂ / (N² · 2^(2-1) · D) = 1.5 / (32² · 2 · 3) = 1.5 / 6144
alpha_2 = GAMMA_2 / (N**2 * 2**(2-1) * D)

# Convection coefficient:
# APEBench _convection.py line 36: convection_difficulty = -substepped_delta
# So when delta_conv = -1.5, the stepper gets convection_difficulty = 1.5
# β₁ = δ_stepper / (M · N · D) = 1.5 / (1.0 · 32 · 3) = 1.5/96
# BUT the PDE has -β₁ in front (convection moves to RHS)
# The actual equation solved: u_t = -β₁ * 1/2 * ∇·(u⊗u) + α₂ * ∇²u
beta_1 = abs(DELTA_CONV) / (M * N * D)

# Domain
L = 1.0
dx = L / N  # APEBench uses boundary-exclusive grid: N points in [0, L)

print("=" * 60)
print("3D Burgers GT PDE Residual (APEBench)")
print("=" * 60)
print(f"  N={N}, D={D}, L={L}, dx={dx}")
print(f"  γ₂={GAMMA_2}, δ_conv={DELTA_CONV}")
print(f"  α₂ (diffusion) = {alpha_2:.6e}")
print(f"  β₁ (convection) = {beta_1:.6e}")
print(f"  dt = 1.0 (normalized)")
print("=" * 60)

# Load data
with h5py.File(DATA_PATH, 'r') as f:
    vector = torch.tensor(f['vector'][:], dtype=torch.float64)  # [60, T, 32, 32, 32, 3]

print(f"\nvector shape: {vector.shape}")
N_samples = vector.shape[0]
T_total = vector.shape[1]

# Use first 5 train samples for verification
N_test = min(5, N_samples)
data = vector[:N_test]  # [5, T, 32, 32, 32, 3]

u = data[..., 0]  # [5, T, 32, 32, 32]
v = data[..., 1]
w = data[..., 2]

print(f"Using {N_test} samples, T={T_total}")
print(f"  u range: [{u.min():.4f}, {u.max():.4f}]")

# FD operators (periodic, 4th-order central on [0,1) with N points, dx=1/N)
def fd_4th_periodic(f, dim, dx):
    """4th-order central difference, periodic BC via roll."""
    fE = torch.roll(f, -1, dims=dim)
    fW = torch.roll(f, 1, dims=dim)
    fEE = torch.roll(f, -2, dims=dim)
    fWW = torch.roll(f, 2, dims=dim)
    return (-fEE + 8*fE - 8*fW + fWW) / (12 * dx)

def laplacian_3d(f, dx):
    """2nd-order central Laplacian, periodic."""
    lap = torch.zeros_like(f)
    for dim in [-3, -2, -1]:  # x, y, z
        lap += (torch.roll(f, -1, dims=dim) - 2*f + torch.roll(f, 1, dims=dim)) / dx**2
    return lap

# Time derivative: 2nd-order central
# df/dt ≈ (f[t+1] - f[t-1]) / (2*dt), dt=1
dt = 1.0

u_mid = u[:, 1:-1]  # [N, T-2, 32, 32, 32]
v_mid = v[:, 1:-1]
w_mid = w[:, 1:-1]

du_dt = (u[:, 2:] - u[:, :-2]) / (2 * dt)
dv_dt = (v[:, 2:] - v[:, :-2]) / (2 * dt)
dw_dt = (w[:, 2:] - w[:, :-2]) / (2 * dt)

# Conservative convection: 1/2 * ∇·(u ⊗ u)
# For u-equation: 1/2 * [∂(uu)/∂x + ∂(uv)/∂y + ∂(uw)/∂z]
conv_u = 0.5 * (fd_4th_periodic(u_mid * u_mid, -3, dx) +
                fd_4th_periodic(u_mid * v_mid, -2, dx) +
                fd_4th_periodic(u_mid * w_mid, -1, dx))

conv_v = 0.5 * (fd_4th_periodic(v_mid * u_mid, -3, dx) +
                fd_4th_periodic(v_mid * v_mid, -2, dx) +
                fd_4th_periodic(v_mid * w_mid, -1, dx))

conv_w = 0.5 * (fd_4th_periodic(w_mid * u_mid, -3, dx) +
                fd_4th_periodic(w_mid * v_mid, -2, dx) +
                fd_4th_periodic(w_mid * w_mid, -1, dx))

# Diffusion
lap_u = laplacian_3d(u_mid, dx)
lap_v = laplacian_3d(v_mid, dx)
lap_w = laplacian_3d(w_mid, dx)

# Residual: du/dt + β₁ * conv - α₂ * lap = 0
R_u = du_dt + beta_1 * conv_u - alpha_2 * lap_u
R_v = dv_dt + beta_1 * conv_v - alpha_2 * lap_v
R_w = dw_dt + beta_1 * conv_w - alpha_2 * lap_w

# RMS per equation
rms_u = torch.sqrt(torch.mean(R_u**2)).item()
rms_v = torch.sqrt(torch.mean(R_v**2)).item()
rms_w = torch.sqrt(torch.mean(R_w**2)).item()
rms_total = torch.sqrt(torch.mean(R_u**2 + R_v**2 + R_w**2) / 3).item()

print(f"\n{'='*60}")
print(f"PDE Residual (RMS, float64, {N_test} samples)")
print(f"{'='*60}")
print(f"  u-momentum: {rms_u:.6e}")
print(f"  v-momentum: {rms_v:.6e}")
print(f"  w-momentum: {rms_w:.6e}")
print(f"  average:    {rms_total:.6e}")
print(f"{'='*60}")

# Per-sample breakdown
print(f"\nPer-sample breakdown:")
for i in range(N_test):
    r_u = torch.sqrt(torch.mean(R_u[i]**2)).item()
    r_v = torch.sqrt(torch.mean(R_v[i]**2)).item()
    r_w = torch.sqrt(torch.mean(R_w[i]**2)).item()
    print(f"  Sample {i}: u={r_u:.6e}, v={r_v:.6e}, w={r_w:.6e}")

# Per-timestep RMS (for eq_scales_per_t)
print(f"\nPer-timestep RMS (first 10 / last 5):")
for eq_name, R in [('u', R_u), ('v', R_v), ('w', R_w)]:
    rms_per_t = torch.sqrt(torch.mean(R**2, dim=(0, 2, 3, 4)))  # [T-2]
    print(f"  {eq_name}: t0={rms_per_t[0]:.6e}, t5={rms_per_t[5]:.6e}, "
          f"t_last={rms_per_t[-1]:.6e}, mean={rms_per_t.mean():.6e}")

print("\nDone!")
