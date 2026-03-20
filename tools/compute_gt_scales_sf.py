"""
Compute per-timestep GT PDE residual RMS scales for Shear Flow.

Equations (simplified FD, not n-PINN):
  1. Continuity: du/dx + dv/dy = 0
  2. x-Momentum: du/dt + d(u*u)/dx + d(u*v)/dy + dp/dx - nu*lap(u) = 0
  3. y-Momentum: dv/dt + d(v*u)/dx + d(v*v)/dy + dp/dy - nu*lap(v) = 0
  4. Tracer: ds/dt + d(s*u)/dx + d(s*v)/dy - D*lap(s) = 0

All periodic BCs, 4th-order central FD.
Data layout: [N, T, 256(Nx), 512(Ny)]  (H=Nx x-periodic, W=Ny y-periodic)

Output: dict with per-equation RMS tensor of shape [T_interior].
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import h5py
import numpy as np
from pathlib import Path

# ---- Config ----
DATA_PATH = './data/finetune/shear_flow_clean.h5'
SAVE_PATH = './data/gt_scales/shear_flow_per_t.pt'
NX, NY = 256, 512
LX, LY = 1.0, 2.0
DT = 0.1
NU = 1e-4
D_COEFF = 1e-3


def dx_4th(f: torch.Tensor, dx: float) -> torch.Tensor:
    """4th-order central x-derivative (periodic, dim=-2)."""
    return (-torch.roll(f, -2, dims=-2) + 8 * torch.roll(f, -1, dims=-2)
            - 8 * torch.roll(f, 1, dims=-2) + torch.roll(f, 2, dims=-2)) / (12 * dx)


def dy_4th(f: torch.Tensor, dy: float) -> torch.Tensor:
    """4th-order central y-derivative (periodic, dim=-1)."""
    return (-torch.roll(f, -2, dims=-1) + 8 * torch.roll(f, -1, dims=-1)
            - 8 * torch.roll(f, 1, dims=-1) + torch.roll(f, 2, dims=-1)) / (12 * dy)


def d2x_4th(f: torch.Tensor, dx: float) -> torch.Tensor:
    """4th-order central 2nd x-derivative (periodic, dim=-2)."""
    return (-torch.roll(f, -2, dims=-2) + 16 * torch.roll(f, -1, dims=-2)
            - 30 * f + 16 * torch.roll(f, 1, dims=-2)
            - torch.roll(f, 2, dims=-2)) / (12 * dx ** 2)


def d2y_4th(f: torch.Tensor, dy: float) -> torch.Tensor:
    """4th-order central 2nd y-derivative (periodic, dim=-1)."""
    return (-torch.roll(f, -2, dims=-1) + 16 * torch.roll(f, -1, dims=-1)
            - 30 * f + 16 * torch.roll(f, 1, dims=-1)
            - torch.roll(f, 2, dims=-1)) / (12 * dy ** 2)


def laplacian_4th(f: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    return d2x_4th(f, dx) + d2y_4th(f, dy)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dx = LX / NX
    dy = LY / NY

    # Load data
    print(f"Loading data from {DATA_PATH} ...")
    with h5py.File(DATA_PATH, 'r') as f:
        vector = torch.from_numpy(f['vector'][:]).float()  # [N, T, 256, 512, 3]
        scalar = torch.from_numpy(f['scalar'][:]).float()  # [N, T, 256, 512, 2]
        print(f"scalar_indices: {f['scalar_indices'][:]}")

    N, T = vector.shape[0], vector.shape[1]
    print(f"N={N}, T={T}, spatial={vector.shape[2]}x{vector.shape[3]}")

    # Extract fields: [N, T, 256(Nx), 512(Ny)]
    # scalar_indices=[11, 12] → scalar[...,0]=tracer, scalar[...,1]=pressure
    u_all = vector[..., 0]    # [N, T, 256, 512]
    v_all = vector[..., 1]
    s_all = scalar[..., 0]    # tracer
    p_all = scalar[..., 1]    # pressure

    print(f"Field shapes: u={u_all.shape}")

    # Interior timesteps for 2nd-order central time derivative: t=1..T-2
    T_int = T - 2
    n_spatial = NX * NY

    # Process in batches
    batch_size = 2  # shear flow is larger (256x512)

    # Accumulators
    cont_sq_sum = torch.zeros(T_int, device=device)
    xmom_sq_sum = torch.zeros(T_int, device=device)
    ymom_sq_sum = torch.zeros(T_int, device=device)
    trac_sq_sum = torch.zeros(T_int, device=device)
    total_count = 0

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        bs = end - start
        print(f"  Processing samples {start}..{end-1}")

        u = u_all[start:end].to(device)
        v = v_all[start:end].to(device)
        s = s_all[start:end].to(device)
        p = p_all[start:end].to(device)

        u_int = u[:, 1:-1]
        v_int = v[:, 1:-1]
        s_int = s[:, 1:-1]
        p_int = p[:, 1:-1]

        # ---- Continuity: du/dx + dv/dy ----
        du_dx = dx_4th(u_int, dx)
        dv_dy = dy_4th(v_int, dy)
        R_cont = du_dx + dv_dy

        for t in range(T_int):
            cont_sq_sum[t] += (R_cont[:, t] ** 2).sum().item()

        # ---- x-Momentum: du/dt + d(uu)/dx + d(uv)/dy + dp/dx - nu*lap(u) ----
        du_dt = (u[:, 2:] - u[:, :-2]) / (2 * DT)
        d_uu_dx = dx_4th(u_int * u_int, dx)
        d_uv_dy = dy_4th(u_int * v_int, dy)
        dp_dx = dx_4th(p_int, dx)
        lap_u = laplacian_4th(u_int, dx, dy)
        R_xmom = du_dt + d_uu_dx + d_uv_dy + dp_dx - NU * lap_u

        for t in range(T_int):
            xmom_sq_sum[t] += (R_xmom[:, t] ** 2).sum().item()

        # ---- y-Momentum: dv/dt + d(vu)/dx + d(vv)/dy + dp/dy - nu*lap(v) ----
        dv_dt = (v[:, 2:] - v[:, :-2]) / (2 * DT)
        d_vu_dx = dx_4th(v_int * u_int, dx)
        d_vv_dy = dy_4th(v_int * v_int, dy)
        dp_dy = dy_4th(p_int, dy)
        lap_v = laplacian_4th(v_int, dx, dy)
        R_ymom = dv_dt + d_vu_dx + d_vv_dy + dp_dy - NU * lap_v

        for t in range(T_int):
            ymom_sq_sum[t] += (R_ymom[:, t] ** 2).sum().item()

        # ---- Tracer: ds/dt + d(su)/dx + d(sv)/dy - D*lap(s) ----
        ds_dt = (s[:, 2:] - s[:, :-2]) / (2 * DT)
        d_su_dx = dx_4th(s_int * u_int, dx)
        d_sv_dy = dy_4th(s_int * v_int, dy)
        lap_s = laplacian_4th(s_int, dx, dy)
        R_trac = ds_dt + d_su_dx + d_sv_dy - D_COEFF * lap_s

        for t in range(T_int):
            trac_sq_sum[t] += (R_trac[:, t] ** 2).sum().item()

        total_count += bs * n_spatial

        del u, v, s, p, u_int, v_int, s_int, p_int
        del R_cont, R_xmom, R_ymom, R_trac
        torch.cuda.empty_cache()

    # Compute RMS per timestep
    cont_rms = torch.sqrt(cont_sq_sum / total_count)
    xmom_rms = torch.sqrt(xmom_sq_sum / total_count)
    ymom_rms = torch.sqrt(ymom_sq_sum / total_count)
    trac_rms = torch.sqrt(trac_sq_sum / total_count)

    result = {
        'continuity': cont_rms.cpu(),
        'x_momentum': xmom_rms.cpu(),
        'y_momentum': ymom_rms.cpu(),
        'tracer': trac_rms.cpu(),
    }

    save_path = Path(SAVE_PATH)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, str(save_path))

    print(f"\nSaved to {save_path}")
    for name, rms in result.items():
        print(f"{name:>12s}: mean={rms.mean():.6f}, std={rms.std():.6f}, "
              f"min={rms.min():.6f}, max={rms.max():.6f}")
        print(f"{'':>12s}  first 10: {rms[:10].tolist()}")


if __name__ == '__main__':
    main()
