"""
Verify Burgers GT dataset with PDE loss.

For boundary-inclusive grid (128 points from 0 to 1):
- Uses ghost cell extrapolation to compute PDE on [1:127, 1:127] (126x126 points)
- Ghost cell: ghost[-1] = 2*data[0] - data[1]
- Expected PDE residual: ~1e-8 for GT data (discretization error)
"""

import torch
import h5py
import numpy as np
from pathlib import Path

from finetune.pde_loss import burgers_pde_loss


def load_sample(file_path: str, sample_idx: int, start_t: int, end_t: int):
    """Load a sample from HDF5."""
    with h5py.File(file_path, 'r') as f:
        nu = float(f['nu'][sample_idx])
        # [T, H, W, 3] -> take first 2 channels (u, v)
        vector_data = np.array(f['vector'][sample_idx, start_t:end_t, ..., :2], dtype=np.float32)

    return {
        'nu': nu,
        'data': torch.from_numpy(vector_data),  # [T, H, W, 2]
    }


def main():
    # Path to dataset
    dataset_path = "/scratch-share/SONG0304/finetune/burgers2d_nu0.1_0.15_n500.h5"

    if not Path(dataset_path).exists():
        print(f"Dataset not found: {dataset_path}")
        print("Please run: python generate_burgers2d_dataset.py")
        return

    print("=" * 60)
    print("Verifying Burgers GT (Boundary-Inclusive Grid)")
    print("=" * 60)

    # Check dataset info
    with h5py.File(dataset_path, 'r') as f:
        print(f"\nDataset info:")
        print(f"  vector shape: {f['vector'].shape}")
        print(f"  nu shape: {f['nu'].shape}")
        if 'dx' in f.attrs:
            print(f"  dx: {f.attrs['dx']}")
        n_samples = f['vector'].shape[0]
        H = f['vector'].shape[2]

    # Physics parameters
    dx = 1.0 / (H - 1)  # = 1/127 for H=128
    dt = 1 / 999
    temporal_length = 17

    print(f"\nPhysics parameters:")
    print(f"  dt = {dt:.6f}")
    print(f"  dx = dy = {dx:.6f}")
    print(f"  Grid: {H}x{H} (boundary-inclusive)")
    print(f"  PDE region: [1:{H-1}, 1:{H-1}] = {H-2}x{H-2} interior points (with ghost cell)")

    # Test samples (randomly select 10 for quick verification)
    all_pde_losses = []
    test_count = min(10, n_samples)
    np.random.seed(42)
    test_indices = np.random.choice(n_samples, test_count, replace=False)
    test_indices = np.sort(test_indices)

    for sample_idx in test_indices:
        print(f"\n{'='*40}")
        print(f"Sample {sample_idx}")
        print(f"{'='*40}")

        # Load sample
        sample = load_sample(dataset_path, sample_idx, 0, temporal_length)
        nu = sample['nu']
        data = sample['data'].unsqueeze(0)  # [1, T, H, W, 2]

        print(f"  nu = {nu:.6f}")
        print(f"  data shape: {data.shape}")

        # Compute PDE loss on GT
        total_loss, loss_u, loss_v, res_u, res_v = burgers_pde_loss(
            pred=data,
            nu=nu,
            dt=dt,
            dx=dx,
            dy=dx,
        )

        print(f"\n  PDE Loss (ghost cell, [1:127]):")
        print(f"    Total: {total_loss.item():.2e}, U: {loss_u.item():.2e}, V: {loss_v.item():.2e}")
        print(f"    Residual shape: {res_u.shape} (expected: [1, T-1, 126, 126])")

        all_pde_losses.append(total_loss.item())

    # Summary
    print("\n" + "=" * 60)
    print(f"Summary (tested {test_count}/{n_samples} samples, indices: {test_indices.tolist()})")
    print("=" * 60)
    avg_loss = np.mean(all_pde_losses)
    print(f"Average PDE Loss: {avg_loss:.2e}")

    if avg_loss < 1e-6:
        print("\n[OK] PDE loss is small for GT data!")
    else:
        print(f"\n[WARNING] PDE loss is {avg_loss:.2e}, expected ~1e-8")

    # Expected discretization error
    print(f"\nExpected discretization error:")
    print(f"  O(dt) = {dt:.2e}")
    print(f"  O(dx²) = {dx**2:.2e}")
    print(f"  Combined ~ {dt + dx**2:.2e}")


if __name__ == "__main__":
    main()
