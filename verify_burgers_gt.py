"""
Verify Burgers GT dataset with PDE loss.

This script:
1. Loads GT data (new format with boundaries)
2. Computes PDE residual (should be ~0 for analytical solution)
3. Computes boundary RMSE (should be 0 for GT vs GT)

If PDE residual is near 0, the pde_loss implementation is correct.
"""

import torch
import h5py
import numpy as np
from pathlib import Path

from pde_loss import burgers_pde_loss_upwind as burgers_pde_loss


def load_sample_new_format(file_path: str, sample_idx: int, start_t: int, end_t: int):
    """Load a sample from new format HDF5."""
    with h5py.File(file_path, 'r') as f:
        # Load nu
        nu = float(f['nu'][sample_idx])

        # Load interior data [T, H, W, 3] -> take first 2 channels
        vector_data = np.array(f['vector'][sample_idx, start_t:end_t, ..., :2], dtype=np.float32)

        # Load boundaries [T, H, 1, 2] or [T, 1, W, 2]
        boundary_left = np.array(f['boundary_left'][sample_idx, start_t:end_t], dtype=np.float32)
        boundary_right = np.array(f['boundary_right'][sample_idx, start_t:end_t], dtype=np.float32)
        boundary_bottom = np.array(f['boundary_bottom'][sample_idx, start_t:end_t], dtype=np.float32)
        boundary_top = np.array(f['boundary_top'][sample_idx, start_t:end_t], dtype=np.float32)

    return {
        'nu': nu,
        'data': torch.from_numpy(vector_data).double(),  # [T, H, W, 2] fp64
        'boundary_left': torch.from_numpy(boundary_left).double(),
        'boundary_right': torch.from_numpy(boundary_right).double(),
        'boundary_bottom': torch.from_numpy(boundary_bottom).double(),
        'boundary_top': torch.from_numpy(boundary_top).double(),
    }


def compute_boundary_rmse(pred, target):
    """Compute boundary RMSE on 4 edges."""
    # pred, target: [B, T, H, W, 2]

    # Left edge (x=1/256)
    left_pred = pred[:, :, :, 0, :]
    left_target = target[:, :, :, 0, :]

    # Right edge (x=255/256)
    right_pred = pred[:, :, :, -1, :]
    right_target = target[:, :, :, -1, :]

    # Bottom edge (y=1/256)
    bottom_pred = pred[:, :, 0, :, :]
    bottom_target = target[:, :, 0, :, :]

    # Top edge (y=255/256)
    top_pred = pred[:, :, -1, :, :]
    top_target = target[:, :, -1, :, :]

    all_pred = torch.cat([
        left_pred.reshape(-1),
        right_pred.reshape(-1),
        bottom_pred.reshape(-1),
        top_pred.reshape(-1),
    ])
    all_target = torch.cat([
        left_target.reshape(-1),
        right_target.reshape(-1),
        bottom_target.reshape(-1),
        top_target.reshape(-1),
    ])

    mse = torch.mean((all_pred - all_target) ** 2)
    return torch.sqrt(mse)


def main():
    # Path to new format dataset
    dataset_path = "/scratch-share/SONG0304/finetune/burgers2d_nu0.1_0.15_res128_t1000_n100.h5"

    if not Path(dataset_path).exists():
        print(f"Dataset not found: {dataset_path}")
        return

    print("=" * 60)
    print("Verifying Burgers GT Dataset with PDE Loss")
    print("=" * 60)

    # Check dataset info
    with h5py.File(dataset_path, 'r') as f:
        print(f"\nDataset info:")
        print(f"  vector shape: {f['vector'].shape}")
        print(f"  boundary_left shape: {f['boundary_left'].shape}")
        print(f"  boundary_right shape: {f['boundary_right'].shape}")
        print(f"  boundary_bottom shape: {f['boundary_bottom'].shape}")
        print(f"  boundary_top shape: {f['boundary_top'].shape}")
        print(f"  nu shape: {f['nu'].shape}")
        n_samples = f['vector'].shape[0]

    # Physics parameters
    dt = 1 / 999  # 1000 timesteps over [0, 1]
    Lx = 1.0
    Ly = 1.0
    H = 128
    W = 128
    dx = Lx / H
    dy = Ly / W

    print(f"\nPhysics parameters:")
    print(f"  dt = {dt:.6f}")
    print(f"  dx = dy = {dx:.6f}")
    print(f"  Lx = Ly = {Lx}")

    # Test multiple samples
    samples_to_test = [0, 10, 50, min(90, n_samples - 1)]
    temporal_length = 17  # Match training

    all_pde_losses = []
    all_bc_rmses = []

    for sample_idx in samples_to_test:
        print(f"\n{'='*40}")
        print(f"Sample {sample_idx}")
        print(f"{'='*40}")

        # Load sample
        sample = load_sample_new_format(dataset_path, sample_idx, 0, temporal_length)
        nu = sample['nu']
        data = sample['data'].unsqueeze(0)  # [1, T, H, W, 2]

        print(f"  nu = {nu:.6f}")
        print(f"  data shape: {data.shape}")

        # Compute PDE loss on GT
        total_loss, loss_u, loss_v, res_u, res_v = burgers_pde_loss(
            pred=data,
            boundary_left=sample['boundary_left'].unsqueeze(0),
            boundary_right=sample['boundary_right'].unsqueeze(0),
            boundary_bottom=sample['boundary_bottom'].unsqueeze(0),
            boundary_top=sample['boundary_top'].unsqueeze(0),
            nu=nu,
            dt=dt,
            Lx=Lx,
            Ly=Ly,
        )

        print(f"\n  PDE Residual (should be ~0 for GT):")
        print(f"    Total Loss: {total_loss.item():.2e}")
        print(f"    Loss U:     {loss_u.item():.2e}")
        print(f"    Loss V:     {loss_v.item():.2e}")
        print(f"    Residual U: max={res_u.abs().max().item():.2e}, mean={res_u.abs().mean().item():.2e}")
        print(f"    Residual V: max={res_v.abs().max().item():.2e}, mean={res_v.abs().mean().item():.2e}")

        all_pde_losses.append(total_loss.item())

        # Compute boundary RMSE (GT vs GT should be 0)
        bc_rmse = compute_boundary_rmse(data, data)
        print(f"\n  Boundary RMSE (GT vs GT, should be 0):")
        print(f"    BC RMSE: {bc_rmse.item():.2e}")

        all_bc_rmses.append(bc_rmse.item())


    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Average PDE Loss on GT: {np.mean(all_pde_losses):.2e}")
    print(f"Average BC RMSE (GT vs GT): {np.mean(all_bc_rmses):.2e}")

    if np.mean(all_pde_losses) < 1e-4:
        print("\n[OK] PDE loss is small for GT data!")
    else:
        print(f"\n[WARNING] PDE loss is {np.mean(all_pde_losses):.2e}")
        print("  This might be due to:")
        print("  1. Numerical precision (dt, dx discretization error)")
        print("  2. Boundary condition handling")
        print("  3. Time derivative approximation")

    # Expected discretization error
    print("\n" + "=" * 60)
    print("Expected Discretization Error Analysis")
    print("=" * 60)
    print(f"  O(dt) ~ {dt:.2e}")
    print(f"  O(dx^2) ~ {dx**2:.2e}")
    print(f"  O(dt) + O(dx^2) ~ {dt + dx**2:.2e}")
    print("  PDE loss should be around this order for GT data")


if __name__ == "__main__":
    main()
