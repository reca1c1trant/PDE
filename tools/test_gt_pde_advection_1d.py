"""
GT PDE Residual Verification for 1D Advection (periodic).

Verifies:
    u_t + a * u_x = 0

Three schemes:
  (A) 4th-order central u_x, 2nd-order central u_t
  (B) 1st-order upwind u_x (1st-order, O(dx))
  (C) n-PINN 2nd-order upwind conservative flux: a*(Fe-Fw)/dx, O(dx²)
      [this is what Advection1DPDELoss in pde_loss_verified.py uses]

Usage:
    python tools/test_gt_pde_advection_1d.py
"""

import torch
import h5py
import numpy as np
import os

DATA_PATH = "/scratch-share/SONG0304/finetune/advection_1d.hdf5"
L = 2.0 * np.pi
N_GRID = 256       # matches generate_advection_1d.py and physics.nx in config
DT = 0.01
dx = L / N_GRID    # 2*pi/256 = 0.024544


def fd_dx_4th(f: torch.Tensor) -> torch.Tensor:
    """4th-order central d/dx in x (dim=-1), periodic."""
    fE = torch.roll(f, -1, dims=-1)
    fW = torch.roll(f, 1, dims=-1)
    fEE = torch.roll(f, -2, dims=-1)
    fWW = torch.roll(f, 2, dims=-1)
    return (-fEE + 8 * fE - 8 * fW + fWW) / (12 * dx)


def adv_upwind_2nd(f: torch.Tensor, a: float) -> torch.Tensor:
    """n-PINN 2nd-order upwind conservative advection: a*(Fe - Fw)/dx.

    Face reconstruction matching Advection1DPDELoss._adv_upwind_2nd:
        Fe_pos = 1.5*f_i - 0.5*f_{i-1}       (a >= 0)
        Fe_neg = 1.5*f_{i+1} - 0.5*f_{i+2}   (a <  0)
    Truncation error O(dx²).
    """
    f_ip1 = torch.roll(f, -1, dims=-1)
    f_im1 = torch.roll(f, 1, dims=-1)
    f_ip2 = torch.roll(f, -2, dims=-1)
    f_im2 = torch.roll(f, 2, dims=-1)

    if a >= 0:
        Fe = 1.5 * f - 0.5 * f_im1
        Fw = 1.5 * f_im1 - 0.5 * f_im2
    else:
        Fe = 1.5 * f_ip1 - 0.5 * f_ip2
        Fw = 1.5 * f - 0.5 * f_ip1

    return a * (Fe - Fw) / dx


def compute_residual_central(u: torch.Tensor, a: float) -> torch.Tensor:
    """Compute residual R = u_t + a * u_x using 4th-order central."""
    du_dt = (u[:, 2:] - u[:, :-2]) / (2 * DT)
    u_int = u[:, 1:-1]
    u_x = fd_dx_4th(u_int)
    return du_dt + a * u_x


def compute_residual_upwind_1st(u: torch.Tensor, a: float) -> torch.Tensor:
    """Compute residual R = u_t + a * u_x using 1st-order upwind.

    When a > 0: backward difference u_x = (u_i - u_{i-1}) / dx
    When a < 0: forward difference  u_x = (u_{i+1} - u_i) / dx
    u_t: 2nd-order central.
    """
    du_dt = (u[:, 2:] - u[:, :-2]) / (2 * DT)
    u_int = u[:, 1:-1]
    u_bwd = (u_int - torch.roll(u_int, 1, dims=-1)) / dx   # backward
    u_fwd = (torch.roll(u_int, -1, dims=-1) - u_int) / dx  # forward
    u_x_upwind = u_bwd if a >= 0 else u_fwd
    return du_dt + a * u_x_upwind


def compute_residual_upwind_2nd(u: torch.Tensor, a: float) -> torch.Tensor:
    """Compute residual using n-PINN 2nd-order upwind (matches training loss)."""
    du_dt = (u[:, 2:] - u[:, :-2]) / (2 * DT)
    u_int = u[:, 1:-1]
    adv_term = adv_upwind_2nd(u_int, a)
    return du_dt + adv_term


def main():
    print("=" * 60)
    print("GT PDE Residual Verification - 1D Advection")
    print("=" * 60)

    # Load data
    with h5py.File(DATA_PATH, 'r') as f:
        scalar = torch.tensor(f['scalar'][:], dtype=torch.float64)  # [N, T, X, 1]
        a_vals = f['nu'][:]  # advection speed stored in 'nu' field
        print(f"  scalar shape: {scalar.shape}")
        print(f"  a range: [{a_vals.min():.4f}, {a_vals.max():.4f}]")

    N_SAMPLES = scalar.shape[0]
    u_all = scalar[..., 0]  # [N, T, X]

    T_int = u_all.shape[1] - 2  # interior timesteps

    rms_central_list = []
    rms_upwind1_list = []
    rms_upwind2_list = []
    per_t_central  = torch.zeros(T_int, dtype=torch.float64)
    per_t_upwind1  = torch.zeros(T_int, dtype=torch.float64)
    per_t_upwind2  = torch.zeros(T_int, dtype=torch.float64)

    print(f"\nComputing PDE residuals for {N_SAMPLES} samples...")
    print(f"  (A) 4th-order central u_x, 2nd-order central u_t")
    print(f"  (B) 1st-order upwind u_x (a-signed), 2nd-order central u_t")
    print(f"  (C) n-PINN 2nd-order upwind a*(Fe-Fw)/dx  [matches training loss]")
    print()

    for i in range(N_SAMPLES):
        a = float(a_vals[i])
        u = u_all[i:i+1]  # [1, T, X]

        # Scheme A: central
        R_c = compute_residual_central(u, a)
        rms_c = torch.sqrt(torch.mean(R_c ** 2)).item()
        rms_central_list.append(rms_c)
        per_t_central += torch.sqrt(torch.mean(R_c ** 2, dim=(0, 2)))

        # Scheme B: 1st-order upwind
        R_u1 = compute_residual_upwind_1st(u, a)
        rms_u1 = torch.sqrt(torch.mean(R_u1 ** 2)).item()
        rms_upwind1_list.append(rms_u1)
        per_t_upwind1 += torch.sqrt(torch.mean(R_u1 ** 2, dim=(0, 2)))

        # Scheme C: n-PINN 2nd-order upwind
        R_u2 = compute_residual_upwind_2nd(u, a)
        rms_u2 = torch.sqrt(torch.mean(R_u2 ** 2)).item()
        rms_upwind2_list.append(rms_u2)
        per_t_upwind2 += torch.sqrt(torch.mean(R_u2 ** 2, dim=(0, 2)))

        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Sample {i+1}/{N_SAMPLES}: central={rms_c:.4e}  upwind1={rms_u1:.4e}  upwind2={rms_u2:.4e}  a={a:.3f}")

    per_t_central /= N_SAMPLES
    per_t_upwind1 /= N_SAMPLES
    per_t_upwind2 /= N_SAMPLES

    central_arr  = np.array(rms_central_list)
    upwind1_arr  = np.array(rms_upwind1_list)
    upwind2_arr  = np.array(rms_upwind2_list)

    print(f"\n{'=' * 60}")
    print(f"Results (N={N_SAMPLES})")
    print(f"{'=' * 60}")
    print(f"  (A) 4th-order central:          mean_RMS = {central_arr.mean():.6e}, max = {central_arr.max():.6e}")
    print(f"  (B) 1st-order upwind:           mean_RMS = {upwind1_arr.mean():.6e}, max = {upwind1_arr.max():.6e}")
    print(f"  (C) n-PINN 2nd-order upwind:    mean_RMS = {upwind2_arr.mean():.6e}, max = {upwind2_arr.max():.6e}")

    # Use n-PINN 2nd-order upwind as eq_scale (matches training loss)
    winner_rms   = upwind2_arr
    winner_per_t = per_t_upwind2
    print(f"\n  >>> Using n-PINN 2nd-order upwind RMS as eq_scales (matches Advection1DPDELoss)")

    # eq_scales for config
    print(f"\n  --- eq_scales for config (mean RMS) ---")
    print(f"  physics:")
    print(f"    eq_scales:")
    print(f"      advection: {winner_rms.mean():.4e}")

    # Save per-timestep scales
    gt_scales_dir = "./data/gt_scales"
    os.makedirs(gt_scales_dir, exist_ok=True)
    per_t_path = os.path.join(gt_scales_dir, "advection_1d_per_t.pt")
    torch.save({
        'advection': winner_per_t.float(),
        'scheme': 'upwind_2nd_npinn',
        'mean_rms_central':  float(central_arr.mean()),
        'mean_rms_upwind1':  float(upwind1_arr.mean()),
        'mean_rms_upwind2':  float(upwind2_arr.mean()),
    }, per_t_path)
    print(f"\n  Saved per-timestep scales to {per_t_path}")
    print(f"    advection (upwind2): shape={winner_per_t.shape}, mean={winner_per_t.mean():.6e}")

    print(f"\n{'=' * 60}")


if __name__ == '__main__':
    main()
