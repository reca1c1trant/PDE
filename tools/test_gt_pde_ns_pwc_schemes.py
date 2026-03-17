"""
Compare FD schemes for NS-PwC GT PDE residual.

Tests: 2nd-order central, 4th-order central, n-PINN conservative upwind.
Equations: divergence, vorticity transport, tracer transport.
All periodic BCs, 128x128 grid.

Usage:
    CUDA_VISIBLE_DEVICES=0 python tools/test_gt_pde_ns_pwc_schemes.py
"""

import torch
import h5py
import numpy as np

DATA_PATH = "/scratch-share/SONG0304/finetune/ns_pwc_1k.hdf5"
NX, NY = 128, 128
LX, LY = 1.0, 1.0
DT = 0.05
NU = 4e-4
KAPPA = 4e-4
N_SAMPLES = 50  # test on first 50 samples


def load_data():
    with h5py.File(DATA_PATH, "r") as f:
        vec = torch.tensor(f["vector"][:N_SAMPLES], dtype=torch.float64)
        sca = torch.tensor(f["scalar"][:N_SAMPLES], dtype=torch.float64)
    u = vec[..., 0]   # [N, T, H, W]
    v = vec[..., 1]
    s = sca[..., 0]   # tracer
    return u, v, s


# ============================================================
# 2nd-order central (periodic via roll)
# ============================================================

def dx_2nd(f: torch.Tensor, dx: float) -> torch.Tensor:
    return (torch.roll(f, -1, dims=-1) - torch.roll(f, 1, dims=-1)) / (2 * dx)

def dy_2nd(f: torch.Tensor, dy: float) -> torch.Tensor:
    return (torch.roll(f, -1, dims=-2) - torch.roll(f, 1, dims=-2)) / (2 * dy)

def lap_2nd(f: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    lx = (torch.roll(f, -1, dims=-1) - 2*f + torch.roll(f, 1, dims=-1)) / dx**2
    ly = (torch.roll(f, -1, dims=-2) - 2*f + torch.roll(f, 1, dims=-2)) / dy**2
    return lx + ly


# ============================================================
# 4th-order central (periodic via roll)
# ============================================================

def dx_4th(f: torch.Tensor, dx: float) -> torch.Tensor:
    return (-torch.roll(f, -2, dims=-1) + 8*torch.roll(f, -1, dims=-1)
            - 8*torch.roll(f, 1, dims=-1) + torch.roll(f, 2, dims=-1)) / (12 * dx)

def dy_4th(f: torch.Tensor, dy: float) -> torch.Tensor:
    return (-torch.roll(f, -2, dims=-2) + 8*torch.roll(f, -1, dims=-2)
            - 8*torch.roll(f, 1, dims=-2) + torch.roll(f, 2, dims=-2)) / (12 * dy)

def d2x_4th(f: torch.Tensor, dx: float) -> torch.Tensor:
    return (-torch.roll(f, -2, dims=-1) + 16*torch.roll(f, -1, dims=-1)
            - 30*f + 16*torch.roll(f, 1, dims=-1)
            - torch.roll(f, 2, dims=-1)) / (12 * dx**2)

def d2y_4th(f: torch.Tensor, dy: float) -> torch.Tensor:
    return (-torch.roll(f, -2, dims=-2) + 16*torch.roll(f, -1, dims=-2)
            - 30*f + 16*torch.roll(f, 1, dims=-2)
            - torch.roll(f, 2, dims=-2)) / (12 * dy**2)

def lap_4th(f: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    return d2x_4th(f, dx) + d2y_4th(f, dy)


# ============================================================
# n-PINN conservative upwind (periodic)
# ============================================================

def npinn_face_velocities(u, v):
    uc_e = 0.5 * (u + torch.roll(u, -1, dims=-1))
    uc_w = 0.5 * (u + torch.roll(u, 1, dims=-1))
    vc_n = 0.5 * (v + torch.roll(v, -1, dims=-2))
    vc_s = 0.5 * (v + torch.roll(v, 1, dims=-2))
    return uc_e, uc_w, vc_n, vc_s

def npinn_divergence(uc_e, uc_w, vc_n, vc_s, dx, dy):
    return (uc_e - uc_w) / dx + (vc_n - vc_s) / dy

def npinn_upwind_convection(f, uc_e, uc_w, vc_n, vc_s, dx, dy):
    f_ip1 = torch.roll(f, -1, dims=-1)
    f_im1 = torch.roll(f, 1, dims=-1)
    f_ip2 = torch.roll(f, -2, dims=-1)
    f_im2 = torch.roll(f, 2, dims=-1)
    f_jp1 = torch.roll(f, -1, dims=-2)
    f_jm1 = torch.roll(f, 1, dims=-2)
    f_jp2 = torch.roll(f, -2, dims=-2)
    f_jm2 = torch.roll(f, 2, dims=-2)

    Fe = torch.where(uc_e >= 0, 1.5*f - 0.5*f_im1, 1.5*f_ip1 - 0.5*f_ip2)
    Fw = torch.where(uc_w >= 0, 1.5*f_im1 - 0.5*f_im2, 1.5*f - 0.5*f_ip1)
    Fn = torch.where(vc_n >= 0, 1.5*f - 0.5*f_jm1, 1.5*f_jp1 - 0.5*f_jp2)
    Fs = torch.where(vc_s >= 0, 1.5*f_jm1 - 0.5*f_jm2, 1.5*f - 0.5*f_jp1)

    return (uc_e*Fe - uc_w*Fw) / dx + (vc_n*Fn - vc_s*Fs) / dy


# ============================================================
# Compute residuals for each scheme
# ============================================================

def compute_residuals(u, v, s, scheme: str):
    """Compute per-equation MSE residuals for given scheme."""
    dx = LX / NX
    dy = LY / NY

    # Interior time (for 2nd-order central time derivative)
    u_int = u[:, 1:-1]
    v_int = v[:, 1:-1]
    s_int = s[:, 1:-1]

    results = {}

    if scheme == "2nd":
        # Divergence
        div = dx_2nd(u_int, dx) + dy_2nd(v_int, dy)
        results['divergence'] = torch.mean(div**2).item()

        # Vorticity: ω = dv/dx - du/dy
        omega_int = dx_2nd(v_int, dx) - dy_2nd(u_int, dy)
        domega_dt = (dx_2nd(v[:, 2:], dx) - dy_2nd(u[:, 2:], dy)
                     - dx_2nd(v[:, :-2], dx) + dy_2nd(u[:, :-2], dy)) / (2*DT)
        adv_omega = u_int * dx_2nd(omega_int, dx) + v_int * dy_2nd(omega_int, dy)
        lap_omega = lap_2nd(omega_int, dx, dy)
        R_vort = domega_dt + adv_omega - NU * lap_omega
        results['vorticity'] = torch.mean(R_vort**2).item()

        # Tracer
        ds_dt = (s[:, 2:] - s[:, :-2]) / (2*DT)
        adv_s = u_int * dx_2nd(s_int, dx) + v_int * dy_2nd(s_int, dy)
        lap_s = lap_2nd(s_int, dx, dy)
        R_tracer = ds_dt + adv_s - KAPPA * lap_s
        results['tracer'] = torch.mean(R_tracer**2).item()

    elif scheme == "4th":
        div = dx_4th(u_int, dx) + dy_4th(v_int, dy)
        results['divergence'] = torch.mean(div**2).item()

        omega_int = dx_4th(v_int, dx) - dy_4th(u_int, dy)
        domega_dt = (dx_4th(v[:, 2:], dx) - dy_4th(u[:, 2:], dy)
                     - dx_4th(v[:, :-2], dx) + dy_4th(u[:, :-2], dy)) / (2*DT)
        adv_omega = u_int * dx_4th(omega_int, dx) + v_int * dy_4th(omega_int, dy)
        lap_omega = lap_4th(omega_int, dx, dy)
        R_vort = domega_dt + adv_omega - NU * lap_omega
        results['vorticity'] = torch.mean(R_vort**2).item()

        ds_dt = (s[:, 2:] - s[:, :-2]) / (2*DT)
        adv_s = u_int * dx_4th(s_int, dx) + v_int * dy_4th(s_int, dy)
        lap_s = lap_4th(s_int, dx, dy)
        R_tracer = ds_dt + adv_s - KAPPA * lap_s
        results['tracer'] = torch.mean(R_tracer**2).item()

    elif scheme == "npinn":
        uc_e, uc_w, vc_n, vc_s = npinn_face_velocities(u_int, v_int)
        div = npinn_divergence(uc_e, uc_w, vc_n, vc_s, dx, dy)
        results['divergence'] = torch.mean(div**2).item()

        # Vorticity: compute omega, then n-PINN convection
        omega_int = dx_4th(v_int, dx) - dy_4th(u_int, dy)  # use 4th for curl
        domega_dt = (dx_4th(v[:, 2:], dx) - dy_4th(u[:, 2:], dy)
                     - dx_4th(v[:, :-2], dx) + dy_4th(u[:, :-2], dy)) / (2*DT)
        conv_omega = npinn_upwind_convection(omega_int, uc_e, uc_w, vc_n, vc_s, dx, dy)
        lap_omega = lap_2nd(omega_int, dx, dy)
        R_vort = domega_dt + conv_omega - NU * lap_omega
        if True:  # div correction
            R_vort = R_vort - omega_int * div
        results['vorticity'] = torch.mean(R_vort**2).item()

        ds_dt = (s[:, 2:] - s[:, :-2]) / (2*DT)
        conv_s = npinn_upwind_convection(s_int, uc_e, uc_w, vc_n, vc_s, dx, dy)
        lap_s = lap_2nd(s_int, dx, dy)
        R_tracer = ds_dt + conv_s - KAPPA * lap_s
        if True:  # div correction
            R_tracer = R_tracer - s_int * div
        results['tracer'] = torch.mean(R_tracer**2).item()

    # RMS = sqrt(MSE)
    rms = {k + '_rms': np.sqrt(v) for k, v in results.items()}
    results.update(rms)

    return results


def main():
    print("Loading data...")
    u, v, s = load_data()
    print(f"  u: {u.shape}, v: {v.shape}, s: {s.shape}")
    print(f"  Grid: {NX}x{NY}, dt={DT}, nu={NU}, kappa={KAPPA}")

    print(f"\n{'='*70}")
    print(f"{'Equation':<20} {'2nd-order RMS':>15} {'4th-order RMS':>15} {'n-PINN RMS':>15} {'Best':>10}")
    print(f"{'='*70}")

    for scheme in ["2nd", "4th", "npinn"]:
        results = compute_residuals(u, v, s, scheme)
        if scheme == "2nd":
            res_2nd = results
        elif scheme == "4th":
            res_4th = results
        else:
            res_npinn = results

    for eq in ['divergence', 'vorticity', 'tracer']:
        r2 = res_2nd[eq + '_rms']
        r4 = res_4th[eq + '_rms']
        rn = res_npinn[eq + '_rms']
        best = min(r2, r4, rn)
        best_name = {r2: '2nd', r4: '4th', rn: 'npinn'}[best]
        print(f"{eq:<20} {r2:>15.6e} {r4:>15.6e} {rn:>15.6e} {best_name:>10}")

    print(f"{'='*70}")

    print("\n=== Recommended eq_scales (using best scheme per equation) ===")
    for eq in ['divergence', 'vorticity', 'tracer']:
        best_rms = min(res_2nd[eq+'_rms'], res_4th[eq+'_rms'], res_npinn[eq+'_rms'])
        print(f"  {eq}: {best_rms:.4e}")


if __name__ == "__main__":
    main()
