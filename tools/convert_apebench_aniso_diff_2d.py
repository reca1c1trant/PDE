"""
Convert APEBench 2D Anisotropic Diffusion .npy data to FinetuneDataset HDF5.

APEBench shape: (N_samples, T, C, H, W)
  - train: (50, 51, 1, 160, 160)
  - test:  (30, 201, 1, 160, 160)

Our shape: scalar (N, T, H, W, C_s), vector (N, T, H, W, 3)

This is a single scalar channel PDE (u only), no velocity fields.

PDE: u_t = div(A * grad(u)) with A = [[0.001, 0.0005], [0.0005, 0.002]]
     domain_extent=1.0, dt=0.1, num_points=160, periodic BC

Usage:
    # Step 1: Download
    hf download thuerey-group/apebench-scraped \\
        "data/2d_phy_aniso_diff__num_spatial_dims=2/2d_phy_aniso_diff__num_spatial_dims=2_train.npy" \\
        "data/2d_phy_aniso_diff__num_spatial_dims=2/2d_phy_aniso_diff__num_spatial_dims=2_test.npy" \\
        "data/2d_phy_aniso_diff__num_spatial_dims=2/2d_phy_aniso_diff__num_spatial_dims=2.json" \\
        --local-dir /tmp/apebench_raw_2d_aniso_diff --repo-type dataset

    # Step 2: Convert
    python tools/convert_apebench_aniso_diff_2d.py \\
        --input_dir /tmp/apebench_raw_2d_aniso_diff/data/2d_phy_aniso_diff__num_spatial_dims=2 \\
        --output /scratch-share/SONG0304/finetune/apebench_2d_aniso_diff.hdf5
"""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Convert APEBench 2D Anisotropic Diffusion to HDF5")
    parser.add_argument(
        "--input_dir", type=str,
        default="/tmp/apebench_raw_2d_aniso_diff/data/2d_phy_aniso_diff__num_spatial_dims=2",
        help="Directory containing train/test .npy and .json",
    )
    parser.add_argument(
        "--output", type=str,
        default="/scratch-share/SONG0304/finetune/apebench_2d_aniso_diff.hdf5",
        help="Output HDF5 path",
    )
    parser.add_argument("--merge", action="store_true", default=True,
                        help="Merge train + test[:51] samples")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    prefix = "2d_phy_aniso_diff__num_spatial_dims=2"

    train_path = input_dir / f"{prefix}_train.npy"
    test_path = input_dir / f"{prefix}_test.npy"

    # Load
    arrays = []
    if train_path.exists():
        train_data = np.load(str(train_path))
        print(f"Train: {train_data.shape}, dtype={train_data.dtype}")
        arrays.append(("train", train_data))
    else:
        print(f"WARNING: train file not found: {train_path}")

    if test_path.exists():
        test_data = np.load(str(test_path))
        print(f"Test:  {test_data.shape}, dtype={test_data.dtype}")
        arrays.append(("test", test_data))
    else:
        print(f"WARNING: test file not found: {test_path}")

    if not arrays:
        raise FileNotFoundError(f"No data files found in {input_dir}")

    # Use test data (full 201 timesteps) as primary
    if len(arrays) == 2 and args.merge:
        # train: [50, 51, 1, 160, 160], test: [30, 201, 1, 160, 160]
        # Use test (full trajectory) + train trimmed to test length? No.
        # Actually: test has 201 steps, train has 51 steps.
        # We want max temporal coverage. Use test as-is (201 steps),
        # and concatenate train padded to 201 (but train only has 51).
        # Better: just use test data which has 201 timesteps.
        # Or: merge train[:51] + test[:51] for more samples with short trajectories,
        #     and keep test with 201 for long-horizon evaluation.
        # Following the pattern in convert_apebench_to_hdf5.py: use full test data.
        data = test_data  # [30, 201, 1, 160, 160]
        print(f"Using test data (full 201 timesteps): {data.shape}")
    elif len(arrays) == 1:
        data = arrays[0][1]
    else:
        data = test_data

    # data shape: [N, T, C, H, W] → scalar [N, T, H, W, C]
    N, T, C, H, W = data.shape
    print(f"\nFinal data: N={N}, T={T}, C={C}, H={H}, W={W}")

    # Transpose: (N, T, C, H, W) → (N, T, H, W, C)
    data_transposed = np.transpose(data, (0, 1, 3, 4, 2))  # [N, T, H, W, C]
    print(f"Transposed: {data_transposed.shape}")

    # Print data range
    print(f"Data range: [{data_transposed.min():.6f}, {data_transposed.max():.6f}]")
    print(f"Data mean: {data_transposed.mean():.6f}, std: {data_transposed.std():.6f}")

    # Save HDF5
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(output_path), "w") as f:
        # Vector: zeros (no velocity field for this PDE)
        vec = np.zeros((N, T, H, W, 3), dtype=np.float32)
        f.create_dataset("vector", data=vec, dtype=np.float32)

        # Scalar: the u field
        f.create_dataset("scalar", data=data_transposed.astype(np.float32), dtype=np.float32)

        # scalar_indices: channel 0 is the scalar u field
        # For diffusion-type scalar, use index 0 (generic scalar)
        f.create_dataset("scalar_indices", data=np.array([0], dtype=np.int64))

        # nu: no per-sample variation, all same PDE params
        f.create_dataset("nu", data=np.ones(N, dtype=np.float32))

    size_mb = os.path.getsize(str(output_path)) / 1e6
    print(f"\nSaved: {output_path} ({size_mb:.1f} MB)")
    print(f"  scalar:         [{N}, {T}, {H}, {W}, {C}]")
    print(f"  vector:         [{N}, {T}, {H}, {W}, 3] (zeros)")
    print(f"  scalar_indices: [0]")

    # Also print the JSON metadata if available
    json_path = input_dir / f"{prefix}.json"
    if json_path.exists():
        import json
        with open(json_path, "r") as jf:
            meta = json.load(jf)
        print(f"\nJSON metadata:")
        for k, v in meta.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
