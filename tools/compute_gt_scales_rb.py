"""
Compute per-timestep GT PDE residual RMS scales for Rayleigh-Bénard.

Equations:
  1. Continuity: du/dx + dv/dy = 0
  2. Buoyancy:   db/dt + u*db/dx + v*db/dy - kappa*lap(b) = 0

Data layout (after transpose): [B, T, 512(Nx), 128(Ny)]
  x-direction (dim=-2, 512 pts): periodic → 4th-order central via roll
  y-direction (dim=-1, 128 pts): Dirichlet → 2nd-order central + ghost cell

Output: dict with per-equation RMS tensor of shape [T_interior] saved to .pt file.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import h5py
import numpy as np
from pathlib import Path

# ---- Config ----
DATA_PATH = './data/finetune/rayleigh_benard_pr1.h5'
SAVE_PATH = './data/gt_scales/rayleigh_benard_per_t.pt'
NX, NY = 512, 128
LX, LY = 4.0, 1.0
DT = 0.25
SKIP_BL = 15


def dx_4th(f: torch.Tensor, dx: float) -> torch.Tensor:
    """4th-order central x-derivative (periodic, dim=-2)."""
    return (-torch.roll(f, -2, dims=-2) + 8 * torch.roll(f, -1, dims=-2)
            - 8 * torch.roll(f, 1, dims=-2) + torch.roll(f, 2, dims=-2)) / (12 * dx)


def d2x_4th(f: torch.Tensor, dx: float) -> torch.Tensor:
    """4th-order central 2nd x-derivative (periodic, dim=-2)."""
    return (-torch.roll(f, -2, dims=-2) + 16 * torch.roll(f, -1, dims=-2)
            - 30 * f + 16 * torch.roll(f, 1, dims=-2)
            - torch.roll(f, 2, dims=-2)) / (12 * dx ** 2)


def dy_ghost(f: torch.Tensor, dy: float) -> torch.Tensor:
    """1st y-derivative with ghost cell (Dirichlet, dim=-1)."""
    interior = (f[..., 2:] - f[..., :-2]) / (2 * dy)
    left = (f[..., 1:2] - f[..., 0:1]) / dy
    right = (f[..., -1:] - f[..., -2:-1]) / dy
    return torch.cat([left, interior, right], dim=-1)


def d2y_ghost(f: torch.Tensor, dy: float) -> torch.Tensor:
    """2nd y-derivative with ghost cell (Dirichlet, dim=-1)."""
    dy2 = dy ** 2
    interior = (f[..., 2:] - 2 * f[..., 1:-1] + f[..., :-2]) / dy2
    zeros = torch.zeros_like(f[..., 0:1])
    return torch.cat([zeros, interior, zeros], dim=-1)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dx = LX / NX
    dy = LY / (NY - 1)

    # Load data
    print(f"Loading data from {DATA_PATH} ...")
    with h5py.File(DATA_PATH, 'r') as f:
        vector = torch.from_numpy(f['vector'][:]).float()  # [N, T, 128, 512, 3]
        scalar = torch.from_numpy(f['scalar'][:]).float()  # [N, T, 128, 512, 2]
        nu_arr = torch.from_numpy(f['nu'][:]).float()       # [N]

    N, T = vector.shape[0], vector.shape[1]
    print(f"N={N}, T={T}, spatial={vector.shape[2]}x{vector.shape[3]}")

    # Extract fields: [N, T, 128(y), 512(x)]
    vx = vector[..., 0]  # [N, T, 128, 512]
    vy = vector[..., 1]
    buoy = scalar[..., 0]

    # IMPORTANT: Transpose to match PDE loss layout [B, T, 512(Nx), 128(Ny)]
    vx = vx.transpose(-1, -2)    # [N, T, 512, 128]
    vy = vy.transpose(-1, -2)
    buoy = buoy.transpose(-1, -2)

    print(f"After transpose: vx shape = {vx.shape}")

    # Interior timesteps for 2nd-order central time derivative: t=1..T-2
    T_int = T - 2  # number of interior timesteps

    # Process in batches to avoid OOM
    batch_size = 4

    # Accumulators: sum of squared residuals per timestep
    cont_sq_sum = torch.zeros(T_int, device=device)
    buoy_sq_sum = torch.zeros(T_int, device=device)
    count_cont = 0
    count_buoy = 0

    skip = SKIP_BL
    n_spatial_cont = NX * (NY - 2 * skip) if skip > 0 else NX * NY
    n_spatial_buoy = NX * (NY - 2 * skip) if skip > 0 else NX * NY

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        bs = end - start
        print(f"  Processing samples {start}..{end-1}")

        u = vx[start:end].to(device)     # [bs, T, 512, 128]
        v = vy[start:end].to(device)
        b = buoy[start:end].to(device)
        kappa = nu_arr[start:end].to(device)  # [bs] (Pr=1 → kappa=nu)

        # Interior timesteps
        u_int = u[:, 1:-1]  # [bs, T_int, 512, 128]
        v_int = v[:, 1:-1]
        b_int = b[:, 1:-1]

        # ---- Continuity: du/dx + dv/dy ----
        du_dx = dx_4th(u_int, dx)
        dv_dy = dy_ghost(v_int, dy)
        R_cont = du_dx + dv_dy  # [bs, T_int, 512, 128]
        if skip > 0:
            R_cont = R_cont[..., skip:-skip]

        # Per-timestep: mean over (batch, x, y) of R^2, then accumulate batch sum
        # R_cont: [bs, T_int, Nx, Ny_trimmed]
        for t in range(T_int):
            r = R_cont[:, t]  # [bs, Nx, Ny_trimmed]
            cont_sq_sum[t] += (r ** 2).sum().item()

        count_cont += bs * n_spatial_cont

        # ---- Buoyancy: db/dt + u*db/dx + v*db/dy - kappa*lap(b) ----
        db_dt = (b[:, 2:] - b[:, :-2]) / (2 * DT)
        db_dx = dx_4th(b_int, dx)
        db_dy = dy_ghost(b_int, dy)
        d2b_dx2 = d2x_4th(b_int, dx)
        d2b_dy2 = d2y_ghost(b_int, dy)
        lap_b = d2b_dx2 + d2b_dy2

        adv = u_int * db_dx + v_int * db_dy
        kappa_4d = kappa.view(-1, 1, 1, 1)  # [bs, 1, 1, 1]
        R_buoy = db_dt + adv - kappa_4d * lap_b

        if skip > 0:
            R_buoy = R_buoy[..., skip:-skip]

        for t in range(T_int):
            r = R_buoy[:, t]
            buoy_sq_sum[t] += (r ** 2).sum().item()

        count_buoy += bs * n_spatial_buoy

        del u, v, b, u_int, v_int, b_int, R_cont, R_buoy
        torch.cuda.empty_cache()

    # Compute RMS per timestep
    cont_rms = torch.sqrt(cont_sq_sum / count_cont)
    buoy_rms = torch.sqrt(buoy_sq_sum / count_buoy)

    result = {
        'continuity': cont_rms.cpu(),
        'buoyancy': buoy_rms.cpu(),
    }

    save_path = Path(SAVE_PATH)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, str(save_path))

    print(f"\nSaved to {save_path}")
    print(f"Continuity RMS per timestep (first 10): {cont_rms[:10].tolist()}")
    print(f"Buoyancy   RMS per timestep (first 10): {buoy_rms[:10].tolist()}")
    print(f"Continuity: mean={cont_rms.mean():.6f}, std={cont_rms.std():.6f}, "
          f"min={cont_rms.min():.6f}, max={cont_rms.max():.6f}")
    print(f"Buoyancy:   mean={buoy_rms.mean():.6f}, std={buoy_rms.std():.6f}, "
          f"min={buoy_rms.min():.6f}, max={buoy_rms.max():.6f}")


if __name__ == '__main__':
    main()
