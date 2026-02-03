"""
Preprocess diffusion-reaction dataset.

Converts from old format (u,v in vector) to new unified format (u,v in scalar[2:3]).

Old format:
    sample_N/
        vector: [T, 128, 128, 2]  # u, v (incorrectly placed)
        scalar: [T, 128, 128, 0]  # empty

New format:
    sample_N/
        vector: [T, 128, 128, 3]       # all zeros (no velocity)
        scalar: [T, 128, 128, 15]      # idx 2: concentration_u, idx 3: concentration_v
        scalar_mask: [15]              # [0,0,1,1,0,0,0,0,0,0,0,0,0,0,0]

Usage:
    python preprocess_diffusion_reaction.py --input /path/to/old.hdf5

    Output will be saved to pretrained/ folder, original file will be deleted.
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

NUM_SCALAR_CHANNELS = 15
NUM_VECTOR_CHANNELS = 3


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

    # Scalar mask for diffusion-reaction: only concentration_u (idx 2) and concentration_v (idx 3)
    scalar_mask = np.zeros(NUM_SCALAR_CHANNELS, dtype=np.float32)
    scalar_mask[SCALAR_INDICES['concentration_u']] = 1.0
    scalar_mask[SCALAR_INDICES['concentration_v']] = 1.0

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Scalar mask: {scalar_mask.astype(int).tolist()}")
    print(f"  concentration_u -> idx {SCALAR_INDICES['concentration_u']}")
    print(f"  concentration_v -> idx {SCALAR_INDICES['concentration_v']}")
    print()

    with h5py.File(input_path, 'r') as f_in:
        sample_keys = sorted(f_in.keys())
        print(f"Found {len(sample_keys)} samples")

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

        print(f"\n{'='*60}")
        print("Starting conversion...")
        print(f"{'='*60}\n")

        with h5py.File(output_path, 'w') as f_out:
            for sample_key in tqdm(sample_keys, desc="Converting"):
                old_vector = f_in[sample_key]['vector'][:]  # [T, H, W, 2]

                # Extract u and v from old vector
                u = old_vector[..., 0]  # [T, H, W]
                v = old_vector[..., 1]  # [T, H, W]

                # Create new vector (all zeros, no velocity in diffusion-reaction)
                new_vector = np.zeros((T, H, W, NUM_VECTOR_CHANNELS), dtype=np.float32)

                # Create new scalar
                new_scalar = np.zeros((T, H, W, NUM_SCALAR_CHANNELS), dtype=np.float32)
                new_scalar[..., SCALAR_INDICES['concentration_u']] = u
                new_scalar[..., SCALAR_INDICES['concentration_v']] = v

                # Write to output
                grp = f_out.create_group(sample_key)
                grp.create_dataset('vector', data=new_vector, compression='gzip', compression_opts=4)
                grp.create_dataset('scalar', data=new_scalar, compression='gzip', compression_opts=4)
                grp.create_dataset('scalar_mask', data=scalar_mask)

    print(f"\n{'='*60}")
    print("Conversion complete!")
    print(f"{'='*60}")

    # Verify output
    print("\nVerifying output...")
    with h5py.File(output_path, 'r') as f_out:
        sample_keys = sorted(f_out.keys())
        first_sample = f_out[sample_keys[0]]
        print(f"\nNew structure (sample {sample_keys[0]}):")
        for key in first_sample.keys():
            data = first_sample[key]
            print(f"  {key}: shape={data.shape}, dtype={data.dtype}")

        # Check data integrity
        vector = first_sample['vector'][:]
        scalar = first_sample['scalar'][:]
        mask = first_sample['scalar_mask'][:]

        print(f"\nData check:")
        print(f"  vector: all zeros = {np.allclose(vector, 0)}")
        print(f"  scalar[..., 2] (u): min={scalar[..., 2].min():.4f}, max={scalar[..., 2].max():.4f}")
        print(f"  scalar[..., 3] (v): min={scalar[..., 3].min():.4f}, max={scalar[..., 3].max():.4f}")
        print(f"  scalar_mask: {mask.astype(int).tolist()}")

        # Check other scalar channels are zero
        other_channels = [i for i in range(NUM_SCALAR_CHANNELS) if i not in [2, 3]]
        other_zero = all(np.allclose(scalar[..., i], 0) for i in other_channels)
        print(f"  Other scalar channels all zero: {other_zero}")

    # Delete original file after successful conversion
    print(f"\nDeleting original file: {input_path}")
    os.remove(input_path)
    print("Original file deleted.")


def main():
    parser = argparse.ArgumentParser(description="Convert diffusion-reaction dataset to unified format")
    parser.add_argument('--input', type=str, required=True, help='Input HDF5 file path')
    args = parser.parse_args()

    convert_diffusion_reaction(args.input)


if __name__ == "__main__":
    main()
