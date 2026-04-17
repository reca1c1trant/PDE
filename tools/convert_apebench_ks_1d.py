"""
Convert APEBench 1D Kuramoto-Sivashinsky (KS) .npy data to HDF5.

APEBench format:
    train: [50, 51, 1, 160]
    test:  [30, 201, 1, 160]

Output HDF5:
    scalar: [80, 201, 160, 1]
    vector: [80, 201, 160, 3] (all zeros)
    scalar_indices: [0]
    nu: [80] (all ones; KS has no per-sample parameter variation)

Strategy:
    - Pad train (51 timesteps) to 201 by repeating the last frame
    - Combine train + test = 80 samples
    - All samples share the same PDE coefficients

Usage:
    python tools/convert_apebench_ks_1d.py \
        --input_dir /tmp/apebench_raw_ks/data \
        --output /scratch-share/SONG0304/finetune/apebench_1d_ks.hdf5
"""

import argparse
import numpy as np
import h5py
import os


def main():
    parser = argparse.ArgumentParser(description="Convert APEBench KS 1D npy -> HDF5")
    parser.add_argument('--input_dir', type=str,
                        default='/tmp/apebench_raw_ks/data')
    parser.add_argument('--output', type=str,
                        default='/scratch-share/SONG0304/finetune/apebench_1d_ks.hdf5')
    args = parser.parse_args()

    train_path = os.path.join(args.input_dir, '1d_diff_ks_train.npy')
    test_path = os.path.join(args.input_dir, '1d_diff_ks_test.npy')

    print("Loading data...")
    train_data = np.load(train_path)  # [50, 51, 1, 160]
    test_data = np.load(test_path)    # [30, 201, 1, 160]
    print(f"  train: {train_data.shape}, dtype={train_data.dtype}")
    print(f"  test:  {test_data.shape}, dtype={test_data.dtype}")
    print(f"  train range: [{train_data.min():.4f}, {train_data.max():.4f}]")
    print(f"  test  range: [{test_data.min():.4f}, {test_data.max():.4f}]")

    N_train, T_train, C, X = train_data.shape
    N_test, T_test = test_data.shape[0], test_data.shape[1]
    T_target = T_test  # 201

    # Pad train from 51 -> 201 by repeating last frame
    if T_train < T_target:
        pad_len = T_target - T_train
        last_frame = train_data[:, -1:, :, :]  # [50, 1, 1, 160]
        padding = np.repeat(last_frame, pad_len, axis=1)  # [50, 150, 1, 160]
        train_padded = np.concatenate([train_data, padding], axis=1)
        print(f"  Padded train: {train_data.shape} -> {train_padded.shape}")
    else:
        train_padded = train_data

    # Merge train + test
    data = np.concatenate([train_padded, test_data], axis=0)  # [80, 201, 1, 160]
    print(f"  Merged: {data.shape}")

    # Rearrange [N, T, C, X] -> [N, T, X, C]
    data = np.transpose(data, (0, 1, 3, 2))  # [80, 201, 160, 1]
    print(f"  Transposed: {data.shape}")

    N_total = data.shape[0]

    # Save HDF5
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with h5py.File(args.output, 'w') as f:
        # Scalar field
        f.create_dataset('scalar', data=data.astype(np.float32))
        f.create_dataset('scalar_indices', data=np.array([0], dtype=np.int64))

        # Empty vector field (1D scalar PDE, no velocity components)
        vec = np.zeros((N_total, T_target, X, 3), dtype=np.float32)
        f.create_dataset('vector', data=vec)

        # nu: all ones (no per-sample parameter variation)
        f.create_dataset('nu', data=np.ones(N_total, dtype=np.float32))

    size_mb = os.path.getsize(args.output) / 1e6
    print(f"\nSaved: {args.output} ({size_mb:.1f} MB)")
    print(f"  scalar: {data.shape}")
    print(f"  scalar_indices: [0]")
    print(f"  nu: all ones ({N_total} samples)")

    # Verify
    with h5py.File(args.output, 'r') as f:
        print(f"\nVerification:")
        for key in f.keys():
            ds = f[key]
            print(f"  {key}: shape={ds.shape}, dtype={ds.dtype}")


if __name__ == '__main__':
    main()
