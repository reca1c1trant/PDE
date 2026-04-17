"""
Convert APEBench .npy data to FinetuneDataset HDF5 format.

APEBench shape: (N_samples, T, C, H, W) for 2D, (N_samples, T, C, X) for 1D
Our shape:      vector (N, T, H, W, 3), scalar (N, T, H, W, C_s)

For 1D: we treat it as (N, T, X, C) → no vector, scalar only
For 2D single-channel: scalar only (e.g., adv, diff, adv_diff, fisher)
For 2D multi-channel (burgers): C=2 → vector (u, v, 0)

Usage:
    python tools/convert_apebench_to_hdf5.py \
        --input_dir /scratch-share/SONG0304/finetune/apebench_raw/data \
        --output_dir /scratch-share/SONG0304/finetune \
        --datasets 2d_adv 2d_diff 2d_adv_diff 2d_burgers 2d_burgers_sc 2d_fisher 2d_disp
"""

import argparse
import numpy as np
import h5py
import os
from pathlib import Path


# APEBench scenario configs: name -> (npy_prefix, ndim, n_channels, channel_type, scalar_indices)
SCENARIOS = {
    # 1D scenarios
    '1d_adv':      ('1d_diff_adv',      1, 1, 'scalar', [1]),  # concentration_rho
    '1d_diff':     ('1d_diff_diff',     1, 1, 'scalar', [1]),  # concentration_rho
    '1d_adv_diff': ('1d_diff_adv_diff', 1, 1, 'scalar', [1]),  # concentration_rho
    '1d_burgers':  ('1d_diff_burgers',  1, 1, 'vector', []),   # 1 channel but it's velocity
    '1d_kdv':      ('1d_diff_kdv',      1, 1, 'scalar', [0]),
    '1d_ks':       ('1d_diff_ks',       1, 1, 'scalar', [0]),
    '1d_fisher':   ('1d_diff_fisher',   1, 1, 'scalar', [0]),
    # 2D scenarios
    '2d_adv':      ('2d_diff_adv__num_spatial_dims=2',      2, 1, 'scalar', [1]),  # concentration_rho
    '2d_diff':     ('2d_diff_diff__num_spatial_dims=2',     2, 1, 'scalar', [1]),  # concentration_rho
    '2d_adv_diff': ('2d_diff_adv_diff__num_spatial_dims=2', 2, 1, 'scalar', [1]),  # concentration_rho
    '2d_burgers':  ('2d_diff_burgers__num_spatial_dims=2',  2, 2, 'vector', []),  # 2ch = (u, v)
    '2d_burgers_sc': ('2d_diff_burgers_sc__num_spatial_dims=2', 2, 1, 'scalar', [0]),
    '2d_disp':     ('2d_diff_disp__num_spatial_dims=2',     2, 1, 'scalar', [0]),
    '2d_fisher':   ('2d_diff_fisher__num_spatial_dims=2',   2, 1, 'scalar', [0]),
}


def convert_scenario(
    name: str,
    input_dir: Path,
    output_dir: Path,
    merge_train_test: bool = True,
):
    """Convert one APEBench scenario from npy to HDF5."""
    cfg = SCENARIOS[name]
    npy_prefix, ndim, n_ch, ch_type, scalar_indices = cfg

    # Load npy files
    train_path = input_dir / f"{npy_prefix}_train.npy"
    test_path = input_dir / f"{npy_prefix}_test.npy"

    has_train = train_path.exists()
    has_test = test_path.exists()

    if not has_train and not has_test:
        print(f"  SKIP {name}: no data files found")
        return

    arrays = []
    if has_train:
        train_data = np.load(str(train_path))
        arrays.append(train_data)
        print(f"  train: {train_data.shape}")
    if has_test:
        test_data = np.load(str(test_path))
        arrays.append(test_data)
        print(f"  test: {test_data.shape}")

    if merge_train_test and len(arrays) == 2:
        # Use first 51 timesteps from test (to match train temporal length)
        test_trimmed = test_data[:, :train_data.shape[1]]
        data = np.concatenate([train_data, test_trimmed], axis=0)
        print(f"  merged (train + test[:51]): {data.shape}")
    elif has_test:
        data = test_data  # keep full 201 timesteps
        print(f"  test only (full): {data.shape}")
    else:
        data = train_data

    # APEBench shape: (N, T, C, ..spatial..)
    # Our shape: vector (N, T, ..spatial.., 3), scalar (N, T, ..spatial.., C_s)
    N, T, C = data.shape[0], data.shape[1], data.shape[2]

    output_path = output_dir / f"apebench_{name}.hdf5"

    with h5py.File(str(output_path), 'w') as f:
        if ndim == 1:
            # (N, T, C, X) → scalar (N, T, X, C)
            X = data.shape[3]
            data_transposed = np.transpose(data, (0, 1, 3, 2))  # (N, T, X, C)

            if ch_type == 'vector':
                # Burgers 1D: single velocity channel → vector (N, T, X, 3)
                vec = np.zeros((N, T, X, 3), dtype=np.float32)
                vec[..., 0] = data_transposed[..., 0]
                f.create_dataset('vector', data=vec, dtype=np.float32)
                f.create_dataset('scalar', data=np.zeros((N, T, X, 0), dtype=np.float32))
                f.create_dataset('scalar_indices', data=np.array([], dtype=np.int64))
            else:
                # Pure scalar
                vec = np.zeros((N, T, X, 3), dtype=np.float32)
                f.create_dataset('vector', data=vec, dtype=np.float32)
                f.create_dataset('scalar', data=data_transposed.astype(np.float32))
                f.create_dataset('scalar_indices', data=np.array(scalar_indices, dtype=np.int64))

        elif ndim == 2:
            # (N, T, C, H, W) → our format
            H, W = data.shape[3], data.shape[4]

            if ch_type == 'vector' and C == 2:
                # Burgers 2D: 2 channels = (u, v) → vector (N, T, H, W, 3)
                vec = np.zeros((N, T, H, W, 3), dtype=np.float32)
                vec[..., 0] = data[:, :, 0]  # u
                vec[..., 1] = data[:, :, 1]  # v
                f.create_dataset('vector', data=vec, dtype=np.float32)
                f.create_dataset('scalar', data=np.zeros((N, T, H, W, 0), dtype=np.float32))
                f.create_dataset('scalar_indices', data=np.array([], dtype=np.int64))
            else:
                # Scalar: (N, T, C, H, W) → scalar (N, T, H, W, C)
                data_transposed = np.transpose(data, (0, 1, 3, 4, 2))  # (N, T, H, W, C)
                vec = np.zeros((N, T, H, W, 3), dtype=np.float32)
                f.create_dataset('vector', data=vec, dtype=np.float32)
                f.create_dataset('scalar', data=data_transposed.astype(np.float32))
                f.create_dataset('scalar_indices', data=np.array(scalar_indices, dtype=np.int64))

        # No per-sample parameter variation in APEBench (all same PDE params)
        f.create_dataset('nu', data=np.ones(N, dtype=np.float32))

    size_mb = os.path.getsize(str(output_path)) / 1e6
    print(f"  Saved: {output_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Convert APEBench npy → HDF5")
    parser.add_argument('--input_dir', type=str,
                        default='/scratch-share/SONG0304/finetune/apebench_raw/data')
    parser.add_argument('--output_dir', type=str,
                        default='/scratch-share/SONG0304/finetune')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Which scenarios to convert (default: all available)')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = args.datasets or list(SCENARIOS.keys())

    print(f"Converting APEBench data from {input_dir}")
    print(f"Output to {output_dir}")
    print(f"Datasets: {datasets}\n")

    for name in datasets:
        if name not in SCENARIOS:
            print(f"Unknown scenario: {name}, skipping")
            continue
        print(f"=== {name} ===")
        convert_scenario(name, input_dir, output_dir)
        print()


if __name__ == "__main__":
    main()
