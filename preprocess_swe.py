"""
Preprocess SWE (Shallow Water Equation) dataset to unified format.

Original format (per-sample groups):
    0000/
        data: [101, 128, 128, 1]  # height only
        grid/t: [101]
        grid/x: [128]
        grid/y: [128]

New unified format (flat arrays):
    vector: NOT created (no velocity)
    scalar: [N, T, H, W, 1]          # height only
    scalar_indices: [10]             # height

Usage:
    python preprocess_swe.py --input /path/to/2D_rdb_NA_NA.h5

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


def convert_swe(input_path: str):
    """
    Convert SWE dataset to unified format.

    Output saved to pretrained/ folder, original file deleted after conversion.

    Args:
        input_path: Path to original HDF5 file
    """
    input_path = Path(input_path)

    # Output to pretrained/ folder with same filename
    output_dir = input_path.parent / 'pretrained'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / input_path.name

    # Scalar indices for SWE: height (10)
    scalar_indices = np.array([
        SCALAR_INDICES['height']
    ], dtype=np.int32)

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Scalar indices: {scalar_indices.tolist()}")
    print(f"  channel 0 -> height (global idx {SCALAR_INDICES['height']})")
    print(f"  No vector dataset (SWE has no velocity in this dataset)")
    print()

    with h5py.File(input_path, 'r') as f_in:
        sample_keys = sorted(f_in.keys())
        n_samples = len(sample_keys)
        print(f"Found {n_samples} samples")

        # Check first sample to understand structure
        first_sample = f_in[sample_keys[0]]
        print(f"\nOriginal structure (sample {sample_keys[0]}):")
        for key in first_sample.keys():
            item = first_sample[key]
            if isinstance(item, h5py.Dataset):
                print(f"  {key}: shape={item.shape}, dtype={item.dtype}")
            elif isinstance(item, h5py.Group):
                print(f"  {key}/ (Group)")
                for subkey in item.keys():
                    subitem = item[subkey]
                    print(f"    {subkey}: shape={subitem.shape}, dtype={subitem.dtype}")

        # Get dimensions from first sample
        data_shape = first_sample['data'].shape  # [T, H, W, 1]
        T, H, W, C = data_shape
        print(f"\nDimensions: T={T}, H={H}, W={W}, C={C}")

        if C != 1:
            print(f"WARNING: Expected 1 channel (height), got {C}")

        # Output shapes
        n_scalar_channels = len(scalar_indices)  # 1
        scalar_shape = (n_samples, T, H, W, n_scalar_channels)

        print(f"\nOutput shape:")
        print(f"  scalar: {scalar_shape}")
        print(f"  scalar_indices: {scalar_indices.shape}")

        print(f"\n{'='*60}")
        print("Starting conversion...")
        print(f"{'='*60}\n")

        with h5py.File(output_path, 'w') as f_out:
            # Create dataset with chunking for efficient random access
            scalar_ds = f_out.create_dataset(
                'scalar',
                shape=scalar_shape,
                dtype=np.float32,
                chunks=(1, T, H, W, n_scalar_channels),
                compression='gzip',
                compression_opts=4
            )

            # Store scalar indices
            f_out.create_dataset('scalar_indices', data=scalar_indices)

            # No vector dataset for SWE

            # Convert each sample
            for i, sample_key in enumerate(tqdm(sample_keys, desc="Converting")):
                data = f_in[sample_key]['data'][:]  # [T, H, W, 1]

                # Write to dataset
                scalar_ds[i] = data

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

        # Check data integrity (first sample)
        scalar = f_out['scalar'][0]
        indices = f_out['scalar_indices'][:]

        print(f"\nData check (sample 0):")
        print(f"  scalar shape: {scalar.shape}")
        print(f"  scalar[..., 0] (height): min={scalar[..., 0].min():.4f}, max={scalar[..., 0].max():.4f}")
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
    parser = argparse.ArgumentParser(description="Convert SWE dataset to unified format")
    parser.add_argument('--input', type=str, required=True, help='Input HDF5 file path')
    args = parser.parse_args()

    convert_swe(args.input)


if __name__ == "__main__":
    main()
