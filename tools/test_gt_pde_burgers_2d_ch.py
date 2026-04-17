"""
GT PDE Residual Verification for 2D Burgers (Cole-Hopf, periodic).

Verifies:
    1. u-momentum: u_t + u*u_x + v*u_y - nu*(u_xx + u_yy) = 0
    2. v-momentum: v_t + u*v_x + v*v_y - nu*(v_xx + v_yy) = 0
    3. irrotational: v_x - u_y = 0

Uses n-PINN conservative upwind for advection terms.

Usage:
    python tools/test_gt_pde_burgers_2d_ch.py
"""

import torch
import h5py
import numpy as np
import os

DATA_PATH = "/scratch-share/SONG0304/finetune/burgers_2d_ch.hdf5"
L = 2.0 * np.pi
N_GRID = 256
DT = 0.01
dx = L / N_GRID
dy = dx


def face_velocities(u, v):
    """Face-averaged velocities (periodic via roll)."""
    uc_e = 0.5 * (u + torch.roll(u, -1, dims=-1))
    uc_w = 0.5 * (u + torch.roll(u, 1, dims=-1))
    vc_n = 0.5 * (v + torch.roll(v, -1, dims=-2))
    vc_s = 0.5 * (v + torch.roll(v, 1, dims=-2))
    return uc_e, uc_w, vc_n, vc_s


def divergence(uc_e, uc_w, vc_n, vc_s):
    """Face-velocity divergence."""
    return (uc_e - uc_w) / dx + (vc_n - vc_s) / dy


def upwind_convection(f, uc_e, uc_w, vc_n, vc_s):
    """Conservative upwind convection: div(f*u) with 2nd-order upwind."""
    f_ip1 = torch.roll(f, -1, dims=-1)
    f_im1 = torch.roll(f, 1, dims=-1)
    f_ip2 = torch.roll(f, -2, dims=-1)
    f_im2 = torch.roll(f, 2, dims=-1)

    f_jp1 = torch.roll(f, -1, dims=-2)
    f_jm1 = torch.roll(f, 1, dims=-2)
    f_jp2 = torch.roll(f, -2, dims=-2)
    f_jm2 = torch.roll(f, 2, dims=-2)

    Fe_pos = 1.5 * f - 0.5 * f_im1
    Fe_neg = 1.5 * f_ip1 - 0.5 * f_ip2
    Fe = torch.where(uc_e >= 0, Fe_pos, Fe_neg)

    Fw_pos = 1.5 * f_im1 - 0.5 * f_im2
    Fw_neg = 1.5 * f - 0.5 * f_ip1
    Fw = torch.where(uc_w >= 0, Fw_pos, Fw_neg)

    Fn_pos = 1.5 * f - 0.5 * f_jm1
    Fn_neg = 1.5 * f_jp1 - 0.5 * f_jp2
    Fn = torch.where(vc_n >= 0, Fn_pos, Fn_neg)

    Fs_pos = 1.5 * f_jm1 - 0.5 * f_jm2
    Fs_neg = 1.5 * f - 0.5 * f_jp1
    Fs = torch.where(vc_s >= 0, Fs_pos, Fs_neg)

    conv = (uc_e * Fe - uc_w * Fw) / dx + (vc_n * Fn - vc_s * Fs) / dy
    return conv


def laplacian_2d(f):
    """2nd-order central Laplacian with periodic BC."""
    lap_x = (torch.roll(f, -1, dims=-1) - 2 * f + torch.roll(f, 1, dims=-1)) / (dx ** 2)
    lap_y = (torch.roll(f, -1, dims=-2) - 2 * f + torch.roll(f, 1, dims=-2)) / (dy ** 2)
    return lap_x + lap_y


def fd_dx_4th(f):
    """4th-order central difference in x (dim=-1), periodic."""
    fE = torch.roll(f, -1, dims=-1)
    fW = torch.roll(f, 1, dims=-1)
    fEE = torch.roll(f, -2, dims=-1)
    fWW = torch.roll(f, 2, dims=-1)
    return (-fEE + 8 * fE - 8 * fW + fWW) / (12 * dx)


def fd_dy_4th(f):
    """4th-order central difference in y (dim=-2), periodic."""
    fN = torch.roll(f, -1, dims=-2)
    fS = torch.roll(f, 1, dims=-2)
    fNN = torch.roll(f, -2, dims=-2)
    fSS = torch.roll(f, 2, dims=-2)
    return (-fNN + 8 * fN - 8 * fS + fSS) / (12 * dy)


def main():
    print("=" * 60)
    print("GT PDE Residual Verification - 2D Burgers (Cole-Hopf)")
    print("=" * 60)

    # Load data
    with h5py.File(DATA_PATH, 'r') as f:
        vector = torch.tensor(f['vector'][:], dtype=torch.float64)
        nu_vals = f['nu'][:]
        print(f"  vector shape: {vector.shape}")
        print(f"  nu range: [{nu_vals.min():.4f}, {nu_vals.max():.4f}]")

    N_SAMPLES = vector.shape[0]
    u_all = vector[..., 0]  # [N, T, H, W]
    v_all = vector[..., 1]

    # Accumulate per-equation RMS
    all_u_mom_rms = []
    all_v_mom_rms = []
    all_irrot_rms = []

    # Per-timestep accumulators
    T_int = u_all.shape[1] - 2  # interior timesteps (central diff)
    u_mom_per_t = torch.zeros(T_int, dtype=torch.float64)
    v_mom_per_t = torch.zeros(T_int, dtype=torch.float64)
    irrot_per_t = torch.zeros(T_int, dtype=torch.float64)

    print(f"\nComputing PDE residuals for {N_SAMPLES} samples...")
    for i in range(N_SAMPLES):
        nu = float(nu_vals[i])
        u = u_all[i:i+1]  # [1, T, H, W]
        v = v_all[i:i+1]

        # Interior timesteps
        u_int = u[:, 1:-1]
        v_int = v[:, 1:-1]

        # Time derivatives (2nd-order central)
        du_dt = (u[:, 2:] - u[:, :-2]) / (2 * DT)
        dv_dt = (v[:, 2:] - v[:, :-2]) / (2 * DT)

        # Face velocities
        uc_e, uc_w, vc_n, vc_s = face_velocities(u_int, v_int)

        # Divergence
        div_uv = divergence(uc_e, uc_w, vc_n, vc_s)

        # u-momentum: du/dt + div(u*u) - u*div - nu*lap_u = 0
        conv_u = upwind_convection(u_int, uc_e, uc_w, vc_n, vc_s)
        lap_u = laplacian_2d(u_int)
        R_u = du_dt + conv_u - u_int * div_uv - nu * lap_u

        # v-momentum: dv/dt + div(v*u) - v*div - nu*lap_v = 0
        conv_v = upwind_convection(v_int, uc_e, uc_w, vc_n, vc_s)
        lap_v = laplacian_2d(v_int)
        R_v = dv_dt + conv_v - v_int * div_uv - nu * lap_v

        # Irrotational constraint: v_x - u_y = 0
        vx = fd_dx_4th(v_int)
        uy = fd_dy_4th(u_int)
        R_irrot = vx - uy

        # RMS per sample
        rms_u = torch.sqrt(torch.mean(R_u ** 2)).item()
        rms_v = torch.sqrt(torch.mean(R_v ** 2)).item()
        rms_irrot = torch.sqrt(torch.mean(R_irrot ** 2)).item()

        all_u_mom_rms.append(rms_u)
        all_v_mom_rms.append(rms_v)
        all_irrot_rms.append(rms_irrot)

        # Per-timestep RMS (mean over space and batch dim)
        u_mom_per_t += torch.sqrt(torch.mean(R_u ** 2, dim=(0, 2, 3)))
        v_mom_per_t += torch.sqrt(torch.mean(R_v ** 2, dim=(0, 2, 3)))
        irrot_per_t += torch.sqrt(torch.mean(R_irrot ** 2, dim=(0, 2, 3)))

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Sample {i+1}/{N_SAMPLES}: u_mom={rms_u:.6e}, v_mom={rms_v:.6e}, irrot={rms_irrot:.6e}")

    # Average per-timestep scales
    u_mom_per_t /= N_SAMPLES
    v_mom_per_t /= N_SAMPLES
    irrot_per_t /= N_SAMPLES

    # Summary
    u_mom_arr = np.array(all_u_mom_rms)
    v_mom_arr = np.array(all_v_mom_rms)
    irrot_arr = np.array(all_irrot_rms)

    print(f"\n{'=' * 60}")
    print(f"Results (N={N_SAMPLES})")
    print(f"{'=' * 60}")
    print(f"  u_momentum: mean_RMS = {u_mom_arr.mean():.6e}, max = {u_mom_arr.max():.6e}")
    print(f"  v_momentum: mean_RMS = {v_mom_arr.mean():.6e}, max = {v_mom_arr.max():.6e}")
    print(f"  irrotational: mean_RMS = {irrot_arr.mean():.6e}, max = {irrot_arr.max():.6e}")

    # eq_scales for config
    print(f"\n  --- eq_scales for config (mean RMS) ---")
    print(f"  physics:")
    print(f"    eq_scales:")
    print(f"      u_momentum: {u_mom_arr.mean():.4e}")
    print(f"      v_momentum: {v_mom_arr.mean():.4e}")
    print(f"      irrotational: {irrot_arr.mean():.4e}")

    # Save per-timestep scales
    gt_scales_dir = "./data/gt_scales"
    os.makedirs(gt_scales_dir, exist_ok=True)
    per_t_path = os.path.join(gt_scales_dir, "burgers_2d_ch_per_t.pt")
    torch.save({
        'u_momentum': u_mom_per_t.float(),
        'v_momentum': v_mom_per_t.float(),
        'irrotational': irrot_per_t.float(),
    }, per_t_path)
    print(f"\n  Saved per-timestep scales to {per_t_path}")
    print(f"    u_momentum: shape={u_mom_per_t.shape}, mean={u_mom_per_t.mean():.6e}")
    print(f"    v_momentum: shape={v_mom_per_t.shape}, mean={v_mom_per_t.mean():.6e}")
    print(f"    irrotational: shape={irrot_per_t.shape}, mean={irrot_per_t.mean():.6e}")

    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
