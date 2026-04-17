"""
GT PDE Residual Verification for 1D KdV (Korteweg-de Vries) from APEBench.

APEBench KdV uses the general nonlinear form (domain [0,1), periodic):
    u_t = beta_1 * u * u_x + alpha_3 * u_xxx + alpha_4 * u_xxxx

where (from difficulty-based coefficients):
    gammas = (0, 0, 0, -14, -9)  -> dispersion + hyper-diffusion
    deltas = (0, -2, 0)          -> single-channel convection

Coefficient conversion (N=160, D=1, M=1.0):
    alpha_j = gamma_j / (N^j * 2^(j-1) * D)
    beta_1  = delta_1 / (M * N * D)

Residual form:
    R = u_t - beta_1 * u * u_x - alpha_3 * u_xxx - alpha_4 * u_xxxx = 0

Uses 4th-order central FD (spatial) + 2nd-order central (temporal).
Periodic BC via torch.roll.

Usage:
    python tools/test_gt_pde_kdv_1d.py
"""

import torch
import h5py
import numpy as np
import os

DATA_PATH = "/scratch-share/SONG0304/finetune/apebench_1d_kdv.hdf5"

# APEBench KdV parameters
N_GRID = 160
Lx = 1.0
dx = Lx / N_GRID  # 0.00625
DT = 1.0

# Difficulty-based coefficients
GAMMAS = (0.0, 0.0, 0.0, -14.0, -9.0)
DELTAS = (0.0, -2.0, 0.0)
D = 1  # num_spatial_dims
M = 1.0  # maximum_absolute

# Convert to normalized (physical) coefficients
# alpha_j = gamma_j / (N^j * 2^(j-1) * D)
ALPHA_3 = GAMMAS[3] / (N_GRID**3 * 2**(3-1) * D)  # -8.5449e-7
ALPHA_4 = GAMMAS[4] / (N_GRID**4 * 2**(4-1) * D)  # -1.7166e-9

# beta_1 = delta_1 / (M * N * D)
# The PDE form is u_t = beta_1 * 1/2 * (u^2)_x = beta_1 * u * u_x
BETA_1 = DELTAS[1] / (M * N_GRID * D)  # -0.0125


def fd_dx_4th(f: torch.Tensor) -> torch.Tensor:
    """4th-order central d/dx in x (dim=-1), periodic."""
    fE = torch.roll(f, -1, dims=-1)
    fW = torch.roll(f, 1, dims=-1)
    fEE = torch.roll(f, -2, dims=-1)
    fWW = torch.roll(f, 2, dims=-1)
    return (-fEE + 8 * fE - 8 * fW + fWW) / (12 * dx)


def fd_d3x_4th(f: torch.Tensor) -> torch.Tensor:
    """4th-order central d^3/dx^3 in x (dim=-1), periodic.

    Stencil: (-1/2, 1, -1, 0, 1, -1, 1/2) / dx^3
    """
    fp1 = torch.roll(f, -1, dims=-1)
    fm1 = torch.roll(f, 1, dims=-1)
    fp2 = torch.roll(f, -2, dims=-1)
    fm2 = torch.roll(f, 2, dims=-1)
    fp3 = torch.roll(f, -3, dims=-1)
    fm3 = torch.roll(f, 3, dims=-1)
    return (-fp3/8 + fp2 - 13*fp1/8 + 13*fm1/8 - fm2 + fm3/8) / (dx**3)


def fd_d4x_4th(f: torch.Tensor) -> torch.Tensor:
    """4th-order central d^4/dx^4 in x (dim=-1), periodic.

    Stencil: (-1/6, 2, -13/2, 28/3, -13/2, 2, -1/6) / dx^4
    """
    fp1 = torch.roll(f, -1, dims=-1)
    fm1 = torch.roll(f, 1, dims=-1)
    fp2 = torch.roll(f, -2, dims=-1)
    fm2 = torch.roll(f, 2, dims=-1)
    fp3 = torch.roll(f, -3, dims=-1)
    fm3 = torch.roll(f, 3, dims=-1)
    return (-fp3/6 + 2*fp2 - 13*fp1/2 + 28*f/3 - 13*fm1/2 + 2*fm2 - fm3/6) / (dx**4)


def compute_residual(u: torch.Tensor) -> torch.Tensor:
    """Compute KdV PDE residual.

    R = u_t - beta_1 * u * u_x - alpha_3 * u_xxx - alpha_4 * u_xxxx

    Args:
        u: [1, T, X]
    Returns:
        R: [1, T_int, X] residual at interior timesteps
    """
    # Time derivative (2nd-order central)
    du_dt = (u[:, 2:] - u[:, :-2]) / (2 * DT)

    # Spatial derivatives at interior timesteps
    u_int = u[:, 1:-1]

    u_x = fd_dx_4th(u_int)
    u_xxx = fd_d3x_4th(u_int)
    u_xxxx = fd_d4x_4th(u_int)

    # Residual: u_t - beta_1*u*u_x - alpha_3*u_xxx - alpha_4*u_xxxx = 0
    R = du_dt - BETA_1 * u_int * u_x - ALPHA_3 * u_xxx - ALPHA_4 * u_xxxx
    return R


def compute_residual_fft(u: torch.Tensor) -> torch.Tensor:
    """Compute KdV PDE residual using FFT spectral derivatives.

    Args:
        u: [1, T, X]
    Returns:
        R: [1, T_int, X] residual at interior timesteps
    """
    # Wavenumbers for domain [0, Lx) with N points
    k = torch.fft.fftfreq(N_GRID, d=dx, device=u.device, dtype=u.dtype) * 2 * np.pi

    # Time derivative (2nd-order central)
    du_dt = (u[:, 2:] - u[:, :-2]) / (2 * DT)

    u_int = u[:, 1:-1]

    # FFT spatial derivatives
    u_hat = torch.fft.fft(u_int, dim=-1)
    u_x = torch.fft.ifft(1j * k * u_hat, dim=-1).real
    u_xxx = torch.fft.ifft((1j * k)**3 * u_hat, dim=-1).real
    u_xxxx = torch.fft.ifft(k**4 * u_hat, dim=-1).real

    R = du_dt - BETA_1 * u_int * u_x - ALPHA_3 * u_xxx - ALPHA_4 * u_xxxx
    return R


def main():
    print("=" * 60)
    print("GT PDE Residual Verification - 1D KdV (APEBench)")
    print("=" * 60)
    print(f"  N={N_GRID}, Lx={Lx}, dx={dx}, dt={DT}")
    print(f"  gammas={GAMMAS}")
    print(f"  deltas={DELTAS}")
    print(f"  alpha_3={ALPHA_3:.6e} (dispersion)")
    print(f"  alpha_4={ALPHA_4:.6e} (hyper-diffusion)")
    print(f"  beta_1 ={BETA_1:.6e} (convection)")

    # Load data
    with h5py.File(DATA_PATH, 'r') as f:
        scalar = torch.tensor(f['scalar'][:], dtype=torch.float64)  # [N, T, X, 1]
        print(f"\n  scalar shape: {scalar.shape}")

    N_SAMPLES = scalar.shape[0]
    u_all = scalar[..., 0]  # [N, T, X]

    T_int = u_all.shape[1] - 2

    # Accumulators
    rms_fd_list = []
    rms_fft_list = []
    fd_per_t = torch.zeros(T_int, dtype=torch.float64)
    fft_per_t = torch.zeros(T_int, dtype=torch.float64)

    print(f"\nComputing PDE residuals for {N_SAMPLES} samples...")
    print(f"  Comparing: (A) 4th-order FD vs (B) FFT spectral")
    print()

    for i in range(N_SAMPLES):
        u = u_all[i:i+1]  # [1, T, X]

        # FD scheme
        R_fd = compute_residual(u)
        rms_fd = torch.sqrt(torch.mean(R_fd ** 2)).item()
        rms_fd_list.append(rms_fd)
        fd_per_t += torch.sqrt(torch.mean(R_fd ** 2, dim=(0, 2)))

        # FFT scheme
        R_fft = compute_residual_fft(u)
        rms_fft = torch.sqrt(torch.mean(R_fft ** 2)).item()
        rms_fft_list.append(rms_fft)
        fft_per_t += torch.sqrt(torch.mean(R_fft ** 2, dim=(0, 2)))

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Sample {i+1}/{N_SAMPLES}: FD={rms_fd:.6e}, FFT={rms_fft:.6e}")

    # Average per-timestep
    fd_per_t /= N_SAMPLES
    fft_per_t /= N_SAMPLES

    fd_arr = np.array(rms_fd_list)
    fft_arr = np.array(rms_fft_list)

    print(f"\n{'=' * 60}")
    print(f"Results (N={N_SAMPLES})")
    print(f"{'=' * 60}")
    print(f"  (A) 4th-order FD:   mean_RMS = {fd_arr.mean():.6e}, max = {fd_arr.max():.6e}")
    print(f"  (B) FFT spectral:   mean_RMS = {fft_arr.mean():.6e}, max = {fft_arr.max():.6e}")

    # Pick winner
    if fft_arr.mean() <= fd_arr.mean():
        winner = "fft"
        winner_rms = fft_arr
        winner_per_t = fft_per_t
        print(f"\n  >>> Winner: (B) FFT spectral (lower mean RMS)")
    else:
        winner = "fd"
        winner_rms = fd_arr
        winner_per_t = fd_per_t
        print(f"\n  >>> Winner: (A) 4th-order FD (lower mean RMS)")

    # eq_scales for config
    print(f"\n  --- eq_scales for config (mean RMS) ---")
    print(f"  physics:")
    print(f"    eq_scales:")
    print(f"      kdv: {winner_rms.mean():.4e}")

    # Save per-timestep scales
    gt_scales_dir = "./data/gt_scales"
    os.makedirs(gt_scales_dir, exist_ok=True)
    per_t_path = os.path.join(gt_scales_dir, "kdv_1d_per_t.pt")
    torch.save({
        'kdv': winner_per_t.float(),
        'scheme': winner,
        'fd_mean_rms': float(fd_arr.mean()),
        'fft_mean_rms': float(fft_arr.mean()),
        'alpha_3': ALPHA_3,
        'alpha_4': ALPHA_4,
        'beta_1': BETA_1,
    }, per_t_path)
    print(f"\n  Saved per-timestep scales to {per_t_path}")
    print(f"    kdv: shape={winner_per_t.shape}, mean={winner_per_t.mean():.6e}")

    print(f"\n{'=' * 60}")


if __name__ == '__main__':
    main()
