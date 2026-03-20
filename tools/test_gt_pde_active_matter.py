"""
GT PDE residual verification for Active Matter dataset.

Computes per-equation RMS residuals to determine eq_scales
for normalized PDE loss.

Active Matter equations (n-PINN conservative upwind):
    1. Continuity: du/dx + dv/dy = 0
    2. Concentration: dc/dt + div(c*u) = d_T * Lap(c)   (+ div correction)
    3. D_xx advection: dD_xx/dt + div(D_xx*u) = d_T * Lap(D_xx)  (+ div correction)

Domain: [0, 10] x [0, 10], periodic BC, 256x256
Parameters: d_T=0.05, dt=0.25

Usage:
    CUDA_VISIBLE_DEVICES=0 python tools/test_gt_pde_active_matter.py
"""

import sys
sys.path.insert(0, '.')

import h5py
import numpy as np
import torch

from finetune.pde_loss_verified import ActiveMatterNPINNPDELoss


def rms(x: np.ndarray) -> float:
    """Root Mean Square."""
    return float(np.sqrt(np.mean(x ** 2)))


def main() -> None:
    data_path = '/home/msai/song0304/code/PDE/data/finetune/active_matter_all.hdf5'
    nx = ny = 256
    Lx = Ly = 10.0
    dt = 0.25
    d_T = 0.05

    print("=" * 78)
    print("  Active Matter — GT PDE Residual Verification")
    print(f"  File: {data_path}")
    print(f"  nx={nx}, ny={ny}, Lx={Lx}, Ly={Ly}, dt={dt}, d_T={d_T}")
    print(f"  dx={Lx/nx:.6f}, dy={Ly/ny:.6f}")
    print("=" * 78)

    # Build PDE loss (default eq_scales=1.0 for all)
    pde_loss = ActiveMatterNPINNPDELoss(
        nx=nx, ny=ny, Lx=Lx, Ly=Ly, dt=dt, d_T=d_T,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pde_loss = pde_loss.to(device)
    print(f"  Device: {device}")

    eq_names = ['continuity', 'concentration', 'Dxx']
    all_losses = {k: [] for k in eq_names}
    all_total = []

    with h5py.File(data_path, 'r') as f:
        vector = f['vector']   # [N, T, H, W, 3]
        scalar = f['scalar']   # [N, T, H, W, 4]
        n_samples = vector.shape[0]
        n_time = vector.shape[1]
        print(f"  Samples={n_samples}, T={n_time}, Grid={nx}x{ny}")
        print(f"  vector shape: {vector.shape}, scalar shape: {scalar.shape}")
        print()

        for s_idx in range(n_samples):
            # Extract fields as float64
            vec = np.array(vector[s_idx], dtype=np.float64)  # [T, H, W, 3]
            scl = np.array(scalar[s_idx], dtype=np.float64)  # [T, H, W, 1]

            vx = vec[:, :, :, 0]   # [T, H, W]
            vy = vec[:, :, :, 1]   # [T, H, W]
            conc = scl[:, :, :, 0]  # [T, H, W] — concentration

            # Reshape to [1, T, H, W] for PDE loss
            u_t = torch.from_numpy(vx).unsqueeze(0).to(device)    # [1, T, H, W]
            v_t = torch.from_numpy(vy).unsqueeze(0).to(device)    # [1, T, H, W]
            c_t = torch.from_numpy(conc).unsqueeze(0).to(device)  # [1, T, H, W]
            Dxx_t = torch.zeros_like(c_t)                          # dummy (weight=0)

            with torch.no_grad():
                total_loss, losses_dict = pde_loss(u_t, v_t, c_t, Dxx_t)

            total_val = total_loss.item()
            all_total.append(total_val)

            per_eq = {}
            for eq in eq_names:
                val = losses_dict[eq].item()
                all_losses[eq].append(val)
                per_eq[eq] = val

            # MSE losses -> RMS of residual = sqrt(MSE)
            print(f"  Sample {s_idx:3d}: "
                  f"total={total_val:.6e}, "
                  f"cont={per_eq['continuity']:.6e}, "
                  f"conc={per_eq['concentration']:.6e}, "
                  f"Dxx={per_eq['Dxx']:.6e}")

    # ── Summary ──
    print()
    print("=" * 78)
    print(f"  Summary over {n_samples} samples")
    print("=" * 78)

    # Per-equation statistics (losses are MSE, so sqrt gives RMS of residual)
    print(f"\n  {'Equation':<18} | {'Mean MSE':>12} | {'Mean RMS':>12} | {'Std RMS':>12}")
    print(f"  {'-'*62}")
    for eq in eq_names:
        arr = np.array(all_losses[eq])
        rms_arr = np.sqrt(arr)  # sqrt(MSE) = RMS of residual
        print(f"  {eq:<18} | {arr.mean():>12.6e} | {rms_arr.mean():>12.6e} | {rms_arr.std():>12.6e}")

    total_arr = np.array(all_total)
    print(f"  {'total':<18} | {total_arr.mean():>12.6e} |")

    # ── eq_scales (RMS of residual per equation, averaged over samples) ──
    print()
    print("=" * 78)
    print("  RMS per equation (use as eq_scales for normalization)")
    print("=" * 78)
    print()
    print("physics:")
    print("  eq_scales:")
    for eq in eq_names:
        arr = np.array(all_losses[eq])
        rms_val = np.sqrt(arr).mean()  # mean of per-sample RMS
        print(f"    {eq}: {rms_val:.4e}")

    # Also compute overall RMS (sqrt of mean MSE across all samples)
    print()
    print("  # Alternative: sqrt(mean_MSE) across all samples")
    print("  # eq_scales:")
    for eq in eq_names:
        arr = np.array(all_losses[eq])
        overall_rms = np.sqrt(arr.mean())
        print(f"  #   {eq}: {overall_rms:.4e}")


if __name__ == '__main__':
    main()
