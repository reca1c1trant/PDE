"""
Preprocess diffusion-reaction dataset to new unified format.

Old format (per-sample groups):
    sample_N/
        vector: [T, 128, 128, 2]  # u, v (incorrectly placed)
        scalar: [T, 128, 128, 0]  # empty

New format (flat arrays):
    vector: [N, T, H, W, 3]          # NOT created (no velocity in diffusion-reaction)
    scalar: [N, T, H, W, 2]          # only actual channels (u, v)
    scalar_indices: [2]              # [2, 3] -> concentration_u, concentration_v

Usage:
    python preprocess_diffusion_reaction.py --input /path/to/old.hdf5

    Output saved to pretrained/ folder, original file deleted after conversion.
"""

import argparse
import h5py
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm


# Scalar channel indices (from scalar_channel.csv)
SCALAR_INDICES = {
    'buoyancy': 0,
    'concentration_rho': 1,
    'concentration_u': 2,
    'concentration_v': 3,
    'density': 4,
    'electron_fraction': 5,
    'energy': 6,
    'entropy': 7,
    'geometry': 8,
    'gravitational_potential': 9,
    'height': 10,
    'passive_tracer': 11,
    'pressure': 12,
    'speed_of_sound': 13,
    'temperature': 14,
}


def convert_diffusion_reaction(input_path: str):
    """
    Convert diffusion-reaction dataset to new unified format.

    Output saved to pretrained/ folder, original file deleted after conversion.

    Args:
        input_path: Path to original HDF5 file
    """
    input_path = Path(input_path)

    # Output to pretrained/ folder with same filename
    output_dir = input_path.parent / 'pretrained'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / input_path.name

    # Scalar indices for diffusion-reaction: concentration_u (2), concentration_v (3)
    scalar_indices = np.array([
        SCALAR_INDICES['concentration_u'],
        SCALAR_INDICES['concentration_v']
    ], dtype=np.int32)

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Scalar indices: {scalar_indices.tolist()}")
    print(f"  channel 0 -> concentration_u (global idx {SCALAR_INDICES['concentration_u']})")
    print(f"  channel 1 -> concentration_v (global idx {SCALAR_INDICES['concentration_v']})")
    print(f"  No vector dataset (diffusion-reaction has no velocity)")
    print()

    with h5py.File(input_path, 'r') as f_in:
        sample_keys = sorted(f_in.keys())
        n_samples = len(sample_keys)
        print(f"Found {n_samples} samples")

        # Check first sample to understand structure
        first_sample = f_in[sample_keys[0]]
        print(f"\nOriginal structure (sample {sample_keys[0]}):")
        for key in first_sample.keys():
            data = first_sample[key]
            print(f"  {key}: shape={data.shape}, dtype={data.dtype}")

        # Get dimensions from first sample
        old_vector = first_sample['vector'][:]
        T, H, W, C_old = old_vector.shape
        print(f"\nDimensions: T={T}, H={H}, W={W}, old_vector_channels={C_old}")

        if C_old != 2:
            print(f"WARNING: Expected 2 channels in vector (u, v), got {C_old}")

        # Determine output shape
        n_scalar_channels = len(scalar_indices)  # 2
        scalar_shape = (n_samples, T, H, W, n_scalar_channels)

        print(f"\nOutput shape:")
        print(f"  scalar: {scalar_shape}")
        print(f"  scalar_indices: {scalar_indices.shape}")

        print(f"\n{'='*60}")
        print("Starting conversion...")
        print(f"{'='*60}\n")

        with h5py.File(output_path, 'w') as f_out:
            # Create datasets with chunking for efficient random access
            # Chunk by single sample: (1, T, H, W, C)
            scalar_ds = f_out.create_dataset(
                'scalar',
                shape=scalar_shape,
                dtype=np.float32,
                chunks=(1, T, H, W, n_scalar_channels),
                compression='gzip',
                compression_opts=4
            )

            # Store scalar indices (small, no compression needed)
            f_out.create_dataset('scalar_indices', data=scalar_indices)

            # No vector dataset for diffusion-reaction (no velocity)

            # Convert each sample
            for i, sample_key in enumerate(tqdm(sample_keys, desc="Converting")):
                old_vector = f_in[sample_key]['vector'][:]  # [T, H, W, 2]

                # Extract u and v, stack as [T, H, W, 2]
                u = old_vector[..., 0:1]  # [T, H, W, 1]
                v = old_vector[..., 1:2]  # [T, H, W, 1]
                scalar_data = np.concatenate([u, v], axis=-1)  # [T, H, W, 2]

                # Write to dataset
                scalar_ds[i] = scalar_data

    print(f"\n{'='*60}")
    print("Conversion complete!")
    print(f"{'='*60}")

    # Verify output
    print("\nVerifying output...")
    with h5py.File(output_path, 'r') as f_out:
        print(f"\nNew structure:")
        for key in f_out.keys():
            data = f_out[key]
            print(f"  {key}: shape={data.shape}, dtype={data.dtype}")

        # Check data integrity
        scalar = f_out['scalar'][0]  # First sample
        indices = f_out['scalar_indices'][:]

        print(f"\nData check (sample 0):")
        print(f"  scalar shape: {scalar.shape}")
        print(f"  scalar[..., 0] (u): min={scalar[..., 0].min():.4f}, max={scalar[..., 0].max():.4f}")
        print(f"  scalar[..., 1] (v): min={scalar[..., 1].min():.4f}, max={scalar[..., 1].max():.4f}")
        print(f"  scalar_indices: {indices.tolist()}")

        # Verify no vector dataset
        has_vector = 'vector' in f_out
        print(f"  vector dataset exists: {has_vector} (expected: False)")

    # Delete original file after successful conversion
    print(f"\nDeleting original file: {input_path}")
    os.remove(input_path)
    print("Original file deleted.")

    print(f"\n{'='*60}")
    print(f"Done! Output: {output_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Convert diffusion-reaction dataset to unified format")
    parser.add_argument('--input', type=str, required=True, help='Input HDF5 file path')
    args = parser.parse_args()

    convert_diffusion_reaction(args.input)


if __name__ == "__main__":
    main()
