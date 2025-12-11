"""
Compute global sigma (std) for each channel across the entire training set.
Used for nRMSE loss normalization.

Usage:
    python compute_sigma.py --data_dir /path/to/data --temporal_length 16

Output:
    Prints sigma values to copy into config.yaml
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm


def compute_sigma(data_dir: str, temporal_length: int = 16, train_ratio: float = 0.9, seed: int = 42):
    """
    Compute global sigma for each channel.

    Uses Welford's online algorithm for numerical stability with large datasets.
    """
    data_dir = Path(data_dir)
    h5_files = sorted(data_dir.glob("*.hdf5"))

    if not h5_files:
        raise ValueError(f"No .hdf5 files found in {data_dir}")

    print(f"Found {len(h5_files)} hdf5 files")

    # Collect all sample keys
    all_samples = []
    for file_path in h5_files:
        with h5py.File(file_path, 'r') as f:
            for sample_key in f.keys():
                vector_shape = f[sample_key]['vector'].shape
                total_timesteps = vector_shape[0]
                if total_timesteps >= temporal_length + 1:
                    all_samples.append((file_path, sample_key))

    print(f"Total samples: {len(all_samples)}")

    # Train/val split (same as dataset.py)
    np.random.seed(seed)
    indices = np.random.permutation(len(all_samples))
    split_idx = int(len(all_samples) * train_ratio)
    train_indices = indices[:split_idx]

    train_samples = [all_samples[i] for i in train_indices]
    print(f"Training samples: {len(train_samples)}")

    # Welford's online algorithm for mean and variance
    # For each channel: track count, mean, M2
    n_channels = 6  # [vx, vy, vz, p, rho, T]
    count = np.zeros(n_channels)
    mean = np.zeros(n_channels)
    M2 = np.zeros(n_channels)

    # Track which channels have real data
    channel_has_data = np.zeros(n_channels, dtype=bool)

    for file_path, sample_key in tqdm(train_samples, desc="Computing sigma"):
        with h5py.File(file_path, 'r') as f:
            vector = np.array(f[sample_key]['vector'], dtype=np.float32)  # [T, H, W, C_v]
            scalar = np.array(f[sample_key]['scalar'], dtype=np.float32)  # [T, H, W, C_s]

        n_vector = vector.shape[-1]
        n_scalar = scalar.shape[-1]

        # Mark channels with real data
        channel_has_data[:n_vector] = True
        channel_has_data[3:3+n_scalar] = True

        # Process vector channels
        for c in range(n_vector):
            data = vector[..., c].flatten()
            for x in data:
                count[c] += 1
                delta = x - mean[c]
                mean[c] += delta / count[c]
                delta2 = x - mean[c]
                M2[c] += delta * delta2

        # Process scalar channels
        for c in range(n_scalar):
            ch_idx = 3 + c
            data = scalar[..., c].flatten()
            for x in data:
                count[ch_idx] += 1
                delta = x - mean[ch_idx]
                mean[ch_idx] += delta / count[ch_idx]
                delta2 = x - mean[ch_idx]
                M2[ch_idx] += delta * delta2

    # Compute final sigma
    sigma = np.zeros(n_channels)
    for c in range(n_channels):
        if count[c] > 1:
            variance = M2[c] / (count[c] - 1)
            sigma[c] = np.sqrt(variance)

    # Print results
    channel_names = ['vx', 'vy', 'vz', 'p', 'rho', 'T']

    print("\n" + "="*50)
    print("Global Sigma (std) per Channel")
    print("="*50)

    for i, name in enumerate(channel_names):
        status = "valid" if channel_has_data[i] else "padded"
        print(f"  {name:>4}: {sigma[i]:.6f}  ({status})")

    print("\n" + "="*50)
    print("Config format (copy to yaml):")
    print("="*50)

    # Format for config
    sigma_list = [f"{s:.6f}" for s in sigma]
    print(f"  nrmse_sigma: [{', '.join(sigma_list)}]")

    # Also print with channel mask
    print("\n  # Channel validity: ", end="")
    print([1 if channel_has_data[i] else 0 for i in range(n_channels)])

    return sigma, channel_has_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute global sigma for nRMSE")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--temporal_length', type=int, default=16, help='Temporal length')
    parser.add_argument('--train_ratio', type=float, default=0.9, help='Train ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()
    compute_sigma(args.data_dir, args.temporal_length, args.train_ratio, args.seed)
