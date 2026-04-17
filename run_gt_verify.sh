#!/bin/bash
#SBATCH --job-name=gt_verify
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=00:10:00
#SBATCH --output=logs_1d3d/gt_verify_%j.out
#SBATCH --error=logs_1d3d/gt_verify_%j.err

cd /home/msai/song0304/code/PDE

python3 -c "
import h5py, numpy as np

N_grid = 160
D_channels = 2  # 2D has 2 channels

# Physical coefficients from APEBench normalization
# alpha_2 = gamma_2 / (N^2 * 2^(2-1) * D) = 1.5 / (160^2 * 2 * 2) for 2D
# delta_conv_physical = delta_conv / (N * 2^? * D) — need to find exact formula
# From user: delta_physical = -1.5 / (N * 4) = -2.344e-3

# === 2D Burgers ===
print('=== 2D Burgers ===')
with h5py.File('data/finetune/apebench/apebench_2d_burgers.hdf5', 'r') as f:
    vec = f['vector'][:].astype(np.float64)

N, T, H, W, _ = vec.shape
L = 1.0
dx = L / H
dy = L / W
dt = 1.0
alpha_2 = 1.5 / (H**2 * 2 * D_channels)  # diffusion
delta_phys = -1.5 / (H * 2 * D_channels)  # convection normalization

print(f'  alpha_2={alpha_2:.6e}, delta_phys={delta_phys:.6e}')
print(f'  dx={dx:.6f}, dt={dt}')

# PDE: u_t + delta_phys * (u*u_x + v*u_y) = alpha_2 * lap(u)
# Actually, let me use the correct form:
# u_t = alpha_2 * lap(u) + delta_phys * u * u_x + delta_phys * v * u_y
# Which is: u_t - alpha_2*lap(u) - delta_phys*(u*u_x + v*u_y) = 0

XA, YA = 1, 0  # W=x, H=y
umom, vmom = [], []

for n in range(N):
    u, v = vec[n,:,:,:,0], vec[n,:,:,:,1]
    for t in range(1, T-1):
        ut, vt = u[t], v[t]
        du_dt = (u[t+1]-u[t-1])/(2*dt)
        dv_dt = (v[t+1]-v[t-1])/(2*dt)
        
        # 4th order central
        du_dx = (-np.roll(ut,-2,XA)+8*np.roll(ut,-1,XA)-8*np.roll(ut,1,XA)+np.roll(ut,2,XA))/(12*dx)
        du_dy = (-np.roll(ut,-2,YA)+8*np.roll(ut,-1,YA)-8*np.roll(ut,1,YA)+np.roll(ut,2,YA))/(12*dy)
        dv_dx = (-np.roll(vt,-2,XA)+8*np.roll(vt,-1,XA)-8*np.roll(vt,1,XA)+np.roll(vt,2,XA))/(12*dx)
        dv_dy = (-np.roll(vt,-2,YA)+8*np.roll(vt,-1,YA)-8*np.roll(vt,1,YA)+np.roll(vt,2,YA))/(12*dy)
        
        lap_u = (np.roll(ut,-1,XA)-2*ut+np.roll(ut,1,XA))/dx**2 + (np.roll(ut,-1,YA)-2*ut+np.roll(ut,1,YA))/dy**2
        lap_v = (np.roll(vt,-1,XA)-2*vt+np.roll(vt,1,XA))/dx**2 + (np.roll(vt,-1,YA)-2*vt+np.roll(vt,1,YA))/dy**2
        
        R_u = du_dt - alpha_2*lap_u - delta_phys*(ut*du_dx + vt*du_dy)
        R_v = dv_dt - alpha_2*lap_v - delta_phys*(ut*dv_dx + vt*dv_dy)
        umom.append(np.sqrt(np.mean(R_u**2)))
        vmom.append(np.sqrt(np.mean(R_v**2)))

print(f'  u_momentum: mean={np.mean(umom):.6e}, max={np.max(umom):.6e}')
print(f'  v_momentum: mean={np.mean(vmom):.6e}, max={np.max(vmom):.6e}')
print(f'  min(eq_scales) = {min(np.mean(umom), np.mean(vmom)):.6e}')

# === 1D Burgers ===
print()
print('=== 1D Burgers ===')
D_1d = 1
with h5py.File('data/finetune/apebench/apebench_1d_burgers.hdf5', 'r') as f:
    vec = f['vector'][:].astype(np.float64)

N, T, X, _ = vec.shape
dx_1d = L / X
alpha_2_1d = 1.5 / (X**2 * 2 * D_1d)
delta_1d = -1.5 / (X * 2 * D_1d)

print(f'  alpha_2={alpha_2_1d:.6e}, delta_phys={delta_1d:.6e}')

mom_1d = []
for n in range(N):
    u = vec[n,:,:,0]
    for t in range(1, T-1):
        ut = u[t]
        du_dt = (u[t+1]-u[t-1])/(2*dt)
        du_dx = (-np.roll(ut,-2)+8*np.roll(ut,-1)-8*np.roll(ut,1)+np.roll(ut,2))/(12*dx_1d)
        lap_u = (np.roll(ut,-1)-2*ut+np.roll(ut,1))/dx_1d**2
        R = du_dt - alpha_2_1d*lap_u - delta_1d*ut*du_dx
        mom_1d.append(np.sqrt(np.mean(R**2)))

print(f'  momentum: mean={np.mean(mom_1d):.6e}, max={np.max(mom_1d):.6e}')
"
