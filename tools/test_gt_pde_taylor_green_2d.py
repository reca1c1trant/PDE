"""GT PDE Residual Verification for Taylor-Green Vortex 2D.

Verifies 3 equations using 4th-order central FD (periodic, torch.roll):
    1. Continuity: du/dx + dv/dy = 0
    2. x-momentum: du/dt + u*du/dx + v*du/dy + dp/dx - nu*lap(u) = 0
    3. y-momentum: dv/dt + u*dv/dx + v*dv/dy + dp/dy - nu*lap(v) = 0

Time derivative: 2nd-order central (f[t+1] - f[t-1]) / (2*dt)
All computations in float64. Processes one sample at a time to avoid OOM.
"""

import torch
import h5py
import numpy as np
import os

DATA_PATH = "./data/finetune/taylor_green_2d.hdf5"
GT_SCALES_DIR = "./data/gt_scales"


def fd_dx_4th_periodic(f: torch.Tensor, dx: float) -> torch.Tensor:
    """4th-order central x-derivative (periodic). f: [..., H, W], x along W (dims=-1)."""
    fE = torch.roll(f, -1, dims=-1)
    fW = torch.roll(f, 1, dims=-1)
    fEE = torch.roll(f, -2, dims=-1)
    fWW = torch.roll(f, 2, dims=-1)
    return (-fEE + 8 * fE - 8 * fW + fWW) / (12 * dx)


def fd_dy_4th_periodic(f: torch.Tensor, dy: float) -> torch.Tensor:
    """4th-order central y-derivative (periodic). f: [..., H, W], y along H (dims=-2)."""
    fN = torch.roll(f, -1, dims=-2)
    fS = torch.roll(f, 1, dims=-2)
    fNN = torch.roll(f, -2, dims=-2)
    fSS = torch.roll(f, 2, dims=-2)
    return (-fNN + 8 * fN - 8 * fS + fSS) / (12 * dy)


def fd_d2x_4th_periodic(f: torch.Tensor, dx: float) -> torch.Tensor:
    """4th-order central 2nd x-derivative (periodic)."""
    fE = torch.roll(f, -1, dims=-1)
    fW = torch.roll(f, 1, dims=-1)
    fEE = torch.roll(f, -2, dims=-1)
    fWW = torch.roll(f, 2, dims=-1)
    return (-fEE + 16 * fE - 30 * f + 16 * fW - fWW) / (12 * dx ** 2)


def fd_d2y_4th_periodic(f: torch.Tensor, dy: float) -> torch.Tensor:
    """4th-order central 2nd y-derivative (periodic)."""
    fN = torch.roll(f, -1, dims=-2)
    fS = torch.roll(f, 1, dims=-2)
    fNN = torch.roll(f, -2, dims=-2)
    fSS = torch.roll(f, 2, dims=-2)
    return (-fNN + 16 * fN - 30 * f + 16 * fS - fSS) / (12 * dy ** 2)


def compute_residuals_single(
    u: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    nu_val: float,
    dx: float,
    dy: float,
    dt: float,
) -> dict[str, torch.Tensor]:
    """Compute PDE residuals for a single sample.

    Args:
        u, v, p: [T, H, W] in float64
        nu_val: viscosity scalar
        dx, dy, dt: grid spacings

    Returns:
        dict of residual tensors [T-2, H, W]
    """
    # Interior timesteps
    u_int = u[1:-1]
    v_int = v[1:-1]
    p_int = p[1:-1]

    # Continuity
    du_dx = fd_dx_4th_periodic(u_int, dx)
    dv_dy = fd_dy_4th_periodic(v_int, dy)
    R_cont = du_dx + dv_dy

    # x-momentum
    du_dt = (u[2:] - u[:-2]) / (2 * dt)
    du_dy = fd_dy_4th_periodic(u_int, dy)
    dp_dx = fd_dx_4th_periodic(p_int, dx)
    d2u_dx2 = fd_d2x_4th_periodic(u_int, dx)
    d2u_dy2 = fd_d2y_4th_periodic(u_int, dy)
    lap_u = d2u_dx2 + d2u_dy2
    R_xmom = du_dt + u_int * du_dx + v_int * du_dy + dp_dx - nu_val * lap_u

    # y-momentum
    dv_dt = (v[2:] - v[:-2]) / (2 * dt)
    dv_dx = fd_dx_4th_periodic(v_int, dx)
    dv_dy_mom = fd_dy_4th_periodic(v_int, dy)
    dp_dy = fd_dy_4th_periodic(p_int, dy)
    d2v_dx2 = fd_d2x_4th_periodic(v_int, dx)
    d2v_dy2 = fd_d2y_4th_periodic(v_int, dy)
    lap_v = d2v_dx2 + d2v_dy2
    R_ymom = dv_dt + u_int * dv_dx + v_int * dv_dy_mom + dp_dy - nu_val * lap_v

    return {
        'continuity': R_cont,
        'x_momentum': R_xmom,
        'y_momentum': R_ymom,
    }


def main():
    print("=" * 60)
    print("GT PDE Residual Verification: Taylor-Green Vortex 2D")
    print("=" * 60)

    # 1. Load metadata
    with h5py.File(DATA_PATH, 'r') as f:
        N, T, H, W, _ = f['vector'].shape
        nu_vals = f['nu'][:]

    print(f"\nData shape: N={N}, T={T}, H={H}, W={W}")
    print(f"nu range: [{nu_vals.min():.4f}, {nu_vals.max():.4f}]")

    # Grid parameters
    L = 2 * np.pi
    dx = L / H
    dy = L / W
    dt = 0.01
    T_int = T - 2

    print(f"dx = {dx:.6f}, dy = {dy:.6f}, dt = {dt}")

    # Accumulators for online RMS computation
    eq_names = ['continuity', 'x_momentum', 'y_momentum']
    sum_sq = {name: torch.zeros(T_int, dtype=torch.float64) for name in eq_names}
    count_per_t = H * W

    # 2. Process sample by sample
    print(f"\nProcessing {N} samples one by one...")
    with h5py.File(DATA_PATH, 'r') as f:
        for i in range(N):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Sample {i + 1}/{N}")

            u = torch.tensor(f['vector'][i, :, :, :, 0], dtype=torch.float64)
            v = torch.tensor(f['vector'][i, :, :, :, 1], dtype=torch.float64)
            p = torch.tensor(f['scalar'][i, :, :, :, 0], dtype=torch.float64)
            nu_val = float(nu_vals[i])

            res = compute_residuals_single(u, v, p, nu_val, dx, dy, dt)

            for name in eq_names:
                R = res[name]  # [T-2, H, W]
                sum_sq[name] += torch.sum(R ** 2, dim=(1, 2))  # [T-2]

    # 3. Compute overall and per-timestep RMS
    total_spatial = N * count_per_t

    print("\n" + "=" * 60)
    print("Per-equation RMS residuals (float64):")
    print("=" * 60)
    eq_scales = {}
    per_t_scales = {}
    for name in eq_names:
        rms_per_t = torch.sqrt(sum_sq[name] / total_spatial)  # [T-2]
        rms_overall = torch.sqrt(torch.sum(sum_sq[name]) / (total_spatial * T_int)).item()
        eq_scales[name] = rms_overall
        per_t_scales[name] = rms_per_t.float()
        print(f"  {name}: RMS = {rms_overall:.6e}")

    # Per-timestep detail
    print("\nPer-timestep RMS (first 5 + last 5):")
    for name in eq_names:
        rms_per_t = per_t_scales[name]
        print(f"  {name}:")
        for i in range(min(5, len(rms_per_t))):
            print(f"    t={i + 1}: {rms_per_t[i]:.6e}")
        if len(rms_per_t) > 10:
            print(f"    ...")
            for i in range(max(5, len(rms_per_t) - 5), len(rms_per_t)):
                print(f"    t={i + 1}: {rms_per_t[i]:.6e}")

    # Save per-timestep scales
    os.makedirs(GT_SCALES_DIR, exist_ok=True)
    save_path = os.path.join(GT_SCALES_DIR, 'taylor_green_2d_per_t.pt')
    torch.save(per_t_scales, save_path)
    print(f"\nSaved per-timestep scales to {save_path}")

    # Print eq_scales for YAML config
    print("\n" + "=" * 60)
    print("eq_scales for YAML config:")
    print("=" * 60)
    for name, rms in eq_scales.items():
        print(f"    {name}: {rms:.6e}")

    print("\nVerification complete!")
    if all(rms < 1e-4 for rms in eq_scales.values()):
        print("All residuals < 1e-4. PASS")
    else:
        print("WARNING: Some residuals >= 1e-4")
        for name, rms in eq_scales.items():
            status = "PASS" if rms < 1e-4 else "FAIL"
            print(f"  {name}: {status} (RMS={rms:.6e})")


if __name__ == "__main__":
    main()
