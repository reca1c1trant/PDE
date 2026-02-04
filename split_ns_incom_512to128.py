"""
Split ns_incom 512x512 dataset into 128x128 patches.

Input: ns_incom_inhom_2d_512_merged.hdf5
    vector: [N=8, T=1000, H=512, W=512, 3]
    scalar: [N=8, T=1000, H=512, W=512, 1]
    force:  [N=8, H=512, W=512, 2]
    scalar_indices: [1]

Output: ns_incom_inhom_2d_128_split.hdf5
    vector: [N=128, T=1000, H=128, W=128, 3]  # 8 samples × 16 patches = 128
    scalar: [N=128, T=1000, H=128, W=128, 1]
    force:  [N=128, H=128, W=128, 2]
    scalar_indices: [1]

Patch layout (4×4 grid):
    Sample 0 → patches 0-15
    Sample 1 → patches 16-31
    ...
    Sample 7 → patches 112-127

Each patch covers a 128×128 region:
    patch_idx = row * 4 + col
    h_start = row * 128
    w_start = col * 128

Chunking: (1, 16, 128, 128, C) for efficient single-sample temporal clip access.

Usage:
    python split_ns_incom_512to128.py --input /path/to/ns_incom_inhom_2d_512_merged.hdf5

    Output saved to same directory as ns_incom_inhom_2d_128_split.hdf5
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm


def split_512_to_128(input_path: str):
    """
    Split 512x512 ns_incom dataset into 128x128 patches.

    Args:
        input_path: Path to ns_incom_inhom_2d_512_merged.hdf5
    """
    input_path = Path(input_path)
    output_path = input_path.parent / 'ns_incom_inhom_2d_128_split.hdf5'

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    # Read input file info
    with h5py.File(input_path, 'r') as f_in:
        vector_shape = f_in['vector'].shape  # [N, T, H, W, 3]
        scalar_shape = f_in['scalar'].shape  # [N, T, H, W, 1]
        force_shape = f_in['force'].shape    # [N, H, W, 2]
        scalar_indices = f_in['scalar_indices'][:]

        N, T, H, W, C_vec = vector_shape
        _, _, _, _, C_sca = scalar_shape

        print(f"\nInput shapes:")
        print(f"  vector: {vector_shape}")
        print(f"  scalar: {scalar_shape}")
        print(f"  force:  {force_shape}")
        print(f"  scalar_indices: {scalar_indices}")

    # Verify dimensions
    assert H == 512 and W == 512, f"Expected 512x512, got {H}x{W}"

    # Output dimensions
    patch_size = 128
    patches_per_dim = H // patch_size  # 4
    patches_per_sample = patches_per_dim * patches_per_dim  # 16
    N_out = N * patches_per_sample  # 8 * 16 = 128

    print(f"\nSplit configuration:")
    print(f"  Original samples: {N}")
    print(f"  Patches per sample: {patches_per_sample} ({patches_per_dim}×{patches_per_dim})")
    print(f"  Total output samples: {N_out}")
    print(f"  Patch size: {patch_size}×{patch_size}")

    # Output shapes
    out_vector_shape = (N_out, T, patch_size, patch_size, C_vec)
    out_scalar_shape = (N_out, T, patch_size, patch_size, C_sca)
    out_force_shape = (N_out, patch_size, patch_size, 2)

    print(f"\nOutput shapes:")
    print(f"  vector: {out_vector_shape}")
    print(f"  scalar: {out_scalar_shape}")
    print(f"  force:  {out_force_shape}")

    # Chunking for efficient temporal clip access
    # Chunk: (1, 16, 128, 128, C) - one sample, 16 timesteps (one clip)
    chunk_T = 16
    vector_chunks = (1, chunk_T, patch_size, patch_size, C_vec)
    scalar_chunks = (1, chunk_T, patch_size, patch_size, C_sca)
    force_chunks = (1, patch_size, patch_size, 2)

    print(f"\nChunking:")
    print(f"  vector: {vector_chunks}")
    print(f"  scalar: {scalar_chunks}")
    print(f"  force:  {force_chunks}")

    print(f"\n{'='*60}")
    print("Starting split...")
    print(f"{'='*60}\n")

    with h5py.File(input_path, 'r') as f_in, \
         h5py.File(output_path, 'w') as f_out:

        # Create output datasets with chunking
        vector_ds = f_out.create_dataset(
            'vector',
            shape=out_vector_shape,
            dtype=np.float32,
            chunks=vector_chunks,
            compression='gzip',
            compression_opts=4
        )

        scalar_ds = f_out.create_dataset(
            'scalar',
            shape=out_scalar_shape,
            dtype=np.float32,
            chunks=scalar_chunks,
            compression='gzip',
            compression_opts=4
        )

        force_ds = f_out.create_dataset(
            'force',
            shape=out_force_shape,
            dtype=np.float32,
            chunks=force_chunks,
            compression='gzip',
            compression_opts=4
        )

        # Copy scalar_indices
        f_out.create_dataset('scalar_indices', data=scalar_indices)

        # Process each original sample
        for sample_idx in tqdm(range(N), desc="Processing samples"):
            # Read entire sample (unavoidable for splitting)
            vector_data = f_in['vector'][sample_idx]  # [T, 512, 512, 3]
            scalar_data = f_in['scalar'][sample_idx]  # [T, 512, 512, 1]
            force_data = f_in['force'][sample_idx]    # [512, 512, 2]

            # Split into 16 patches (4×4 grid)
            for row in range(patches_per_dim):
                for col in range(patches_per_dim):
                    patch_idx = row * patches_per_dim + col
                    out_idx = sample_idx * patches_per_sample + patch_idx

                    h_start = row * patch_size
                    h_end = h_start + patch_size
                    w_start = col * patch_size
                    w_end = w_start + patch_size

                    # Extract patches
                    vector_patch = vector_data[:, h_start:h_end, w_start:w_end, :]
                    scalar_patch = scalar_data[:, h_start:h_end, w_start:w_end, :]
                    force_patch = force_data[h_start:h_end, w_start:w_end, :]

                    # Write to output
                    vector_ds[out_idx] = vector_patch
                    scalar_ds[out_idx] = scalar_patch
                    force_ds[out_idx] = force_patch

            # Progress info
            if (sample_idx + 1) % 2 == 0:
                print(f"  Completed {sample_idx + 1}/{N} samples "
                      f"(patches {(sample_idx + 1) * patches_per_sample}/{N_out})")

    print(f"\n{'='*60}")
    print("Split complete!")
    print(f"{'='*60}")

    # Verify output
    print("\nVerifying output...")
    with h5py.File(output_path, 'r') as f_out:
        print(f"\nOutput structure:")
        for key in f_out.keys():
            ds = f_out[key]
            if hasattr(ds, 'chunks'):
                print(f"  {key}: shape={ds.shape}, dtype={ds.dtype}, chunks={ds.chunks}")
            else:
                print(f"  {key}: shape={ds.shape}, dtype={ds.dtype}")

        # Data integrity check
        vector = f_out['vector']
        scalar = f_out['scalar']
        force = f_out['force']

        print(f"\nData check:")
        print(f"  Sample 0 (original sample 0, patch 0):")
        print(f"    vector range: [{vector[0].min():.4f}, {vector[0].max():.4f}]")
        print(f"    scalar range: [{scalar[0].min():.4f}, {scalar[0].max():.4f}]")
        print(f"  Sample 16 (original sample 1, patch 0):")
        print(f"    vector range: [{vector[16].min():.4f}, {vector[16].max():.4f}]")
        print(f"    scalar range: [{scalar[16].min():.4f}, {scalar[16].max():.4f}]")

    # File size comparison
    input_size = input_path.stat().st_size / (1024**3)
    output_size = output_path.stat().st_size / (1024**3)
    print(f"\nFile sizes:")
    print(f"  Input:  {input_size:.2f} GB")
    print(f"  Output: {output_size:.2f} GB")

    print(f"\n{'='*60}")
    print(f"Done! Output: {output_path}")
    print(f"{'='*60}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Split ns_incom 512x512 dataset into 128x128 patches"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to ns_incom_inhom_2d_512_merged.hdf5'
    )
    args = parser.parse_args()

    split_512_to_128(args.input)


if __name__ == "__main__":
    main()
