"""
Preprocess 2D CFD dataset to unified format.

Original format (4 files, each with flat arrays):
    Vx: [1000, 21, 128, 128]
    Vy: [1000, 21, 128, 128]
    density: [1000, 21, 128, 128]
    pressure: [1000, 21, 128, 128]

New unified format (single merged file):
    vector: [N, T, H, W, 3]          # [Vx, Vy, 0]
    scalar: [N, T, H, W, 2]          # [density, pressure]
    scalar_indices: [2]              # [4, 12] -> density, pressure

Usage:
    python preprocess_2d_cfd.py --input_dir /path/to/2D_CFD_128

    Output saved to pretrained/ folder, original files deleted after conversion.
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

NUM_VECTOR_CHANNELS = 3


def convert_2d_cfd(input_dir: str):
    """
    Convert 2D CFD dataset to unified format.

    Merges all files in input_dir into a single output file.
    Output saved to pretrained/ folder, original files deleted after conversion.

    Args:
        input_dir: Directory containing 2D CFD HDF5 files
    """
    input_dir = Path(input_dir)

    # Find all HDF5 files
    h5_files = sorted(list(input_dir.glob("*.hdf5")) + list(input_dir.glob("*.h5")))
    if not h5_files:
        raise ValueError(f"No HDF5 files found in {input_dir}")

    print(f"Found {len(h5_files)} files:")
    for f in h5_files:
        print(f"  {f.name}")

    # Output to pretrained/ folder
    output_dir = input_dir.parent / 'pretrained'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / '2D_CFD_128_merged.hdf5'

    # Scalar indices for 2D CFD: density (4), pressure (12)
    scalar_indices = np.array([
        SCALAR_INDICES['density'],
        SCALAR_INDICES['pressure']
    ], dtype=np.int32)

    print(f"\nOutput: {output_path}")
    print(f"Scalar indices: {scalar_indices.tolist()}")
    print(f"  channel 0 -> density (global idx {SCALAR_INDICES['density']})")
    print(f"  channel 1 -> pressure (global idx {SCALAR_INDICES['pressure']})")
    print(f"Vector: [Vx, Vy, 0]")

    # First pass: count total samples and verify dimensions
    total_samples = 0
    T, H, W = None, None, None

    print(f"\nScanning files...")
    for h5_file in h5_files:
        with h5py.File(h5_file, 'r') as f:
            vx_shape = f['Vx'].shape  # [N, T, H, W]
            n, t, h, w = vx_shape

            if T is None:
                T, H, W = t, h, w
            else:
                assert (t, h, w) == (T, H, W), f"Shape mismatch in {h5_file}"

            total_samples += n
            print(f"  {h5_file.name}: {n} samples, shape [T={t}, H={h}, W={w}]")

    print(f"\nTotal: {total_samples} samples")
    print(f"Dimensions: T={T}, H={H}, W={W}")

    # Output shapes
    n_scalar_channels = len(scalar_indices)  # 2
    vector_shape = (total_samples, T, H, W, NUM_VECTOR_CHANNELS)
    scalar_shape = (total_samples, T, H, W, n_scalar_channels)

    print(f"\nOutput shapes:")
    print(f"  vector: {vector_shape}")
    print(f"  scalar: {scalar_shape}")

    print(f"\n{'='*60}")
    print("Starting conversion...")
    print(f"{'='*60}\n")

    with h5py.File(output_path, 'w') as f_out:
        # Create datasets with chunking for efficient random access
        vector_ds = f_out.create_dataset(
            'vector',
            shape=vector_shape,
            dtype=np.float32,
            chunks=(1, T, H, W, NUM_VECTOR_CHANNELS),
            compression='gzip',
            compression_opts=4
        )

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

        # Convert each file
        sample_offset = 0
        for h5_file in h5_files:
            print(f"Processing {h5_file.name}...")

            with h5py.File(h5_file, 'r') as f_in:
                n_samples = f_in['Vx'].shape[0]

                # Process in batches to manage memory
                batch_size = 100
                for start in tqdm(range(0, n_samples, batch_size), desc=f"  {h5_file.name}"):
                    end = min(start + batch_size, n_samples)

                    # Read batch
                    vx = f_in['Vx'][start:end]        # [batch, T, H, W]
                    vy = f_in['Vy'][start:end]        # [batch, T, H, W]
                    density = f_in['density'][start:end]    # [batch, T, H, W]
                    pressure = f_in['pressure'][start:end]  # [batch, T, H, W]

                    # Create vector: [batch, T, H, W, 3]
                    vz = np.zeros_like(vx)
                    vector_batch = np.stack([vx, vy, vz], axis=-1)

                    # Create scalar: [batch, T, H, W, 2]
                    scalar_batch = np.stack([density, pressure], axis=-1)

                    # Write to output
                    out_start = sample_offset + start
                    out_end = sample_offset + end
                    vector_ds[out_start:out_end] = vector_batch
                    scalar_ds[out_start:out_end] = scalar_batch

                sample_offset += n_samples

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
        vector = f_out['vector'][0]
        scalar = f_out['scalar'][0]
        indices = f_out['scalar_indices'][:]

        print(f"\nData check (sample 0):")
        print(f"  vector[..., 0] (Vx): min={vector[..., 0].min():.4f}, max={vector[..., 0].max():.4f}")
        print(f"  vector[..., 1] (Vy): min={vector[..., 1].min():.4f}, max={vector[..., 1].max():.4f}")
        print(f"  vector[..., 2] (Vz): all zeros = {np.allclose(vector[..., 2], 0)}")
        print(f"  scalar[..., 0] (density): min={scalar[..., 0].min():.4f}, max={scalar[..., 0].max():.4f}")
        print(f"  scalar[..., 1] (pressure): min={scalar[..., 1].min():.4f}, max={scalar[..., 1].max():.4f}")
        print(f"  scalar_indices: {indices.tolist()}")

    # Delete original files after successful conversion
    print(f"\nDeleting original files...")
    for h5_file in h5_files:
        print(f"  Deleting {h5_file.name}")
        os.remove(h5_file)
    print("Original files deleted.")

    print(f"\n{'='*60}")
    print(f"Done! Output: {output_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Convert 2D CFD dataset to unified format")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing 2D CFD HDF5 files')
    args = parser.parse_args()

    convert_2d_cfd(args.input_dir)


if __name__ == "__main__":
    main()
