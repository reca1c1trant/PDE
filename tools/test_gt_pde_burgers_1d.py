"""
GT PDE Residual Verification for 1D Burgers (Cole-Hopf, periodic).

Verifies:
    u_t + u * u_x - nu * u_xx = 0

Compares two schemes for the advection term u*u_x:
    (A) 4th-order central difference for u_x
    (B) n-PINN conservative upwind: div(u*u) - u*div(u) with 2nd-order upwind

Uses 4th-order central for u_xx, 2nd-order central for u_t.

Usage:
    python tools/test_gt_pde_burgers_1d.py
"""

import torch
import h5py
import numpy as np
import os

DATA_PATH = "/scratch-share/SONG0304/finetune/burgers_1d.hdf5"
L = 2.0 * np.pi
N_GRID = 256       # matches generate_burgers_1d.py and physics.nx in config
DT = 0.01
dx = L / N_GRID    # 2*pi/256 = 0.024544


# =====================================================================
# Scheme A: 4th-order central for both advection and diffusion
# =====================================================================

def fd_dx_4th(f: torch.Tensor) -> torch.Tensor:
    """4th-order central d/dx in x (dim=-1), periodic."""
    fE = torch.roll(f, -1, dims=-1)
    fW = torch.roll(f, 1, dims=-1)
    fEE = torch.roll(f, -2, dims=-1)
    fWW = torch.roll(f, 2, dims=-1)
    return (-fEE + 8 * fE - 8 * fW + fWW) / (12 * dx)


def fd_d2x_4th(f: torch.Tensor) -> torch.Tensor:
    """4th-order central d^2/dx^2 in x (dim=-1), periodic."""
    fE = torch.roll(f, -1, dims=-1)
    fW = torch.roll(f, 1, dims=-1)
    fEE = torch.roll(f, -2, dims=-1)
    fWW = torch.roll(f, 2, dims=-1)
    return (-fEE + 16 * fE - 30 * f + 16 * fW - fWW) / (12 * dx**2)


def compute_residual_central(u: torch.Tensor, nu: float) -> torch.Tensor:
    """Compute residual using 4th-order central for advection and diffusion.

    Args:
        u: [1, T, X]
    Returns:
        R: [1, T_int, X] residual
    """
    u_int = u[:, 1:-1]

    # Time derivative (2nd-order central)
    du_dt = (u[:, 2:] - u[:, :-2]) / (2 * DT)

    # Advection: u * u_x
    u_x = fd_dx_4th(u_int)
    advection = u_int * u_x

    # Diffusion: nu * u_xx
    u_xx = fd_d2x_4th(u_int)
    diffusion = nu * u_xx

    # Residual: u_t + u*u_x - nu*u_xx = 0
    R = du_dt + advection - diffusion
    return R


# =====================================================================
# Scheme B: n-PINN conservative upwind for advection
# =====================================================================

def face_velocities_1d(u: torch.Tensor):
    """Face-averaged velocities (periodic via roll) in x direction."""
    uc_e = 0.5 * (u + torch.roll(u, -1, dims=-1))  # east face
    uc_w = 0.5 * (u + torch.roll(u, 1, dims=-1))   # west face
    return uc_e, uc_w


def divergence_1d(uc_e: torch.Tensor, uc_w: torch.Tensor) -> torch.Tensor:
    """Face-velocity divergence in 1D."""
    return (uc_e - uc_w) / dx


def upwind_convection_1d(
    f: torch.Tensor,
    uc_e: torch.Tensor,
    uc_w: torch.Tensor,
) -> torch.Tensor:
    """Conservative upwind convection in 1D: div(f * u_face) with 2nd-order upwind."""
    f_ip1 = torch.roll(f, -1, dims=-1)
    f_im1 = torch.roll(f, 1, dims=-1)
    f_ip2 = torch.roll(f, -2, dims=-1)
    f_im2 = torch.roll(f, 2, dims=-1)

    # East face
    Fe_pos = 1.5 * f - 0.5 * f_im1
    Fe_neg = 1.5 * f_ip1 - 0.5 * f_ip2
    Fe = torch.where(uc_e >= 0, Fe_pos, Fe_neg)

    # West face
    Fw_pos = 1.5 * f_im1 - 0.5 * f_im2
    Fw_neg = 1.5 * f - 0.5 * f_ip1
    Fw = torch.where(uc_w >= 0, Fw_pos, Fw_neg)

    conv = (uc_e * Fe - uc_w * Fw) / dx
    return conv


def compute_residual_upwind(u: torch.Tensor, nu: float) -> torch.Tensor:
    """Compute residual using n-PINN conservative upwind for advection.

    Args:
        u: [1, T, X]
    Returns:
        R: [1, T_int, X] residual
    """
    u_int = u[:, 1:-1]

    # Time derivative (2nd-order central)
    du_dt = (u[:, 2:] - u[:, :-2]) / (2 * DT)

    # Face velocities
    uc_e, uc_w = face_velocities_1d(u_int)

    # Divergence correction
    div_u = divergence_1d(uc_e, uc_w)

    # Upwind convection: div(u * u_face) - u * div(u_face)
    conv_u = upwind_convection_1d(u_int, uc_e, uc_w)
    advection = conv_u - u_int * div_u

    # Diffusion: nu * u_xx (4th-order central)
    u_xx = fd_d2x_4th(u_int)
    diffusion = nu * u_xx

    # Residual: u_t + advection - nu*u_xx = 0
    R = du_dt + advection - diffusion
    return R


def main():
    print("=" * 60)
    print("GT PDE Residual Verification - 1D Burgers (Cole-Hopf)")
    print("=" * 60)

    # Load data — burgers_1d stores u in vector[..., 0], scalar is empty
    with h5py.File(DATA_PATH, 'r') as f:
        vector = torch.tensor(f['vector'][:], dtype=torch.float64)  # [N, T, X, 3]
        nu_vals = f['nu'][:]
        print(f"  vector shape: {vector.shape}")
        print(f"  nu range: [{nu_vals.min():.4f}, {nu_vals.max():.4f}]")

    N_SAMPLES = vector.shape[0]
    u_all = vector[..., 0]  # [N, T, X]  — u in vector channel 0

    # Accumulators for both schemes
    rms_central_list = []
    rms_upwind_list = []

    T_int = u_all.shape[1] - 2  # interior timesteps

    # Per-timestep accumulators (for the winning scheme)
    central_per_t = torch.zeros(T_int, dtype=torch.float64)
    upwind_per_t = torch.zeros(T_int, dtype=torch.float64)

    print(f"\nComputing PDE residuals for {N_SAMPLES} samples...")
    print(f"  Comparing: (A) 4th-order central vs (B) n-PINN upwind")
    print()

    for i in range(N_SAMPLES):
        nu = float(nu_vals[i])
        u = u_all[i:i+1]  # [1, T, X]

        # Scheme A: central
        R_central = compute_residual_central(u, nu)
        rms_c = torch.sqrt(torch.mean(R_central ** 2)).item()
        rms_central_list.append(rms_c)
        central_per_t += torch.sqrt(torch.mean(R_central ** 2, dim=(0, 2)))

        # Scheme B: upwind
        R_upwind = compute_residual_upwind(u, nu)
        rms_u = torch.sqrt(torch.mean(R_upwind ** 2)).item()
        rms_upwind_list.append(rms_u)
        upwind_per_t += torch.sqrt(torch.mean(R_upwind ** 2, dim=(0, 2)))

        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Sample {i+1}/{N_SAMPLES}: central={rms_c:.6e}, upwind={rms_u:.6e}")

    # Average per-timestep
    central_per_t /= N_SAMPLES
    upwind_per_t /= N_SAMPLES

    # Summary
    central_arr = np.array(rms_central_list)
    upwind_arr = np.array(rms_upwind_list)

    print(f"\n{'=' * 60}")
    print(f"Results (N={N_SAMPLES})")
    print(f"{'=' * 60}")
    print(f"  (A) 4th-order central:  mean_RMS = {central_arr.mean():.6e}, max = {central_arr.max():.6e}")
    print(f"  (B) n-PINN upwind:      mean_RMS = {upwind_arr.mean():.6e}, max = {upwind_arr.max():.6e}")

    # Pick winner
    if central_arr.mean() <= upwind_arr.mean():
        winner = "central"
        winner_rms = central_arr
        winner_per_t = central_per_t
        print(f"\n  >>> Winner: (A) 4th-order central (lower mean RMS)")
    else:
        winner = "upwind"
        winner_rms = upwind_arr
        winner_per_t = upwind_per_t
        print(f"\n  >>> Winner: (B) n-PINN conservative upwind (lower mean RMS)")

    # eq_scales for config
    print(f"\n  --- eq_scales for config (mean RMS) ---")
    print(f"  physics:")
    print(f"    eq_scales:")
    print(f"      momentum: {winner_rms.mean():.4e}")

    # Save per-timestep scales
    gt_scales_dir = "./data/gt_scales"
    os.makedirs(gt_scales_dir, exist_ok=True)
    per_t_path = os.path.join(gt_scales_dir, "burgers_1d_per_t.pt")
    torch.save({
        'momentum': winner_per_t.float(),
        'scheme': winner,
        'central_mean_rms': float(central_arr.mean()),
        'upwind_mean_rms': float(upwind_arr.mean()),
    }, per_t_path)
    print(f"\n  Saved per-timestep scales to {per_t_path}")
    print(f"    momentum: shape={winner_per_t.shape}, mean={winner_per_t.mean():.6e}")

    print(f"\n{'=' * 60}")


if __name__ == '__main__':
    main()
