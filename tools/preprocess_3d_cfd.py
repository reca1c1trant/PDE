"""
Preprocess 3D CFD dataset to unified format.

Original format (single file):
    Vx: [40, 21, 128, 128, 128]
    Vy: [40, 21, 128, 128, 128]
    Vz: [40, 21, 128, 128, 128]
    density: [40, 21, 128, 128, 128]
    pressure: [40, 21, 128, 128, 128]

New unified format:
    vector: [N, T, D, H, W, 3]      # [Vx, Vy, Vz]
    scalar: [N, T, D, H, W, 2]      # [density, pressure]
    scalar_indices: [2]              # [4, 12] -> density, pressure

Usage:
    python preprocess_3d_cfd.py --input /path/to/3D_CFD_sub40.hdf5

    Output saved next to input file (same directory), original file NOT deleted.
"""

import argparse
import h5py
import numpy as np
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


def convert_3d_cfd(input_path: str):
    """
    Convert 3D CFD dataset to unified format.

    Args:
        input_path: Path to the 3D CFD HDF5 file
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Output to same directory
    output_path = input_path.parent / '3D_CFD_unified.hdf5'

    # Scalar indices for 3D CFD: density (4), pressure (12)
    scalar_indices = np.array([
        SCALAR_INDICES['density'],
        SCALAR_INDICES['pressure'],
    ], dtype=np.int32)

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Scalar indices: {scalar_indices.tolist()}")
    print(f"  channel 0 -> density (global idx {SCALAR_INDICES['density']})")
    print(f"  channel 1 -> pressure (global idx {SCALAR_INDICES['pressure']})")
    print(f"Vector: [Vx, Vy, Vz] (3D velocity)")

    # Read metadata
    with h5py.File(input_path, 'r') as f_in:
        print(f"\nOriginal structure:")
        for key in f_in.keys():
            data = f_in[key]
            print(f"  {key}: shape={data.shape}, dtype={data.dtype}")

        vx_shape = f_in['Vx'].shape  # [N, T, D, H, W]
        N, T, D, H, W = vx_shape

    print(f"\nDimensions: N={N}, T={T}, D={D}, H={H}, W={W}")

    # Output shapes
    n_scalar_channels = len(scalar_indices)  # 2
    vector_shape = (N, T, D, H, W, NUM_VECTOR_CHANNELS)
    scalar_shape = (N, T, D, H, W, n_scalar_channels)

    print(f"\nOutput shapes:")
    print(f"  vector: {vector_shape}")
    print(f"  scalar: {scalar_shape}")

    # Estimate size
    total_bytes = (np.prod(vector_shape) + np.prod(scalar_shape)) * 4
    print(f"  Uncompressed size: {total_bytes / 1e9:.1f} GB")

    print(f"\n{'='*60}")
    print("Starting conversion...")
    print(f"{'='*60}\n")

    with h5py.File(output_path, 'w') as f_out:
        # Create datasets with chunking (1 sample per chunk for random access)
        vector_ds = f_out.create_dataset(
            'vector',
            shape=vector_shape,
            dtype=np.float32,
            chunks=(1, T, D, H, W, NUM_VECTOR_CHANNELS),
            compression='gzip',
            compression_opts=4,
        )

        scalar_ds = f_out.create_dataset(
            'scalar',
            shape=scalar_shape,
            dtype=np.float32,
            chunks=(1, T, D, H, W, n_scalar_channels),
            compression='gzip',
            compression_opts=4,
        )

        # Store scalar indices
        f_out.create_dataset('scalar_indices', data=scalar_indices)

        # Convert one sample at a time (128^3 volumes are large)
        with h5py.File(input_path, 'r') as f_in:
            for i in tqdm(range(N), desc="Converting"):
                # Read one sample: [T, D, H, W]
                vx = f_in['Vx'][i]
                vy = f_in['Vy'][i]
                vz = f_in['Vz'][i]
                density = f_in['density'][i]
                pressure = f_in['pressure'][i]

                # Create vector: [T, D, H, W, 3]
                vector_sample = np.stack([vx, vy, vz], axis=-1)

                # Create scalar: [T, D, H, W, 2]
                scalar_sample = np.stack([density, pressure], axis=-1)

                # Write
                vector_ds[i] = vector_sample
                scalar_ds[i] = scalar_sample

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
        print(f"  vector[..., 2] (Vz): min={vector[..., 2].min():.4f}, max={vector[..., 2].max():.4f}")
        print(f"  scalar[..., 0] (density): min={scalar[..., 0].min():.4f}, max={scalar[..., 0].max():.4f}")
        print(f"  scalar[..., 1] (pressure): min={scalar[..., 1].min():.4f}, max={scalar[..., 1].max():.4f}")
        print(f"  scalar_indices: {indices.tolist()}")

    print(f"\nOriginal file preserved: {input_path}")
    print(f"\n{'='*60}")
    print(f"Done! Output: {output_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Convert 3D CFD dataset to unified format")
    parser.add_argument('--input', type=str, required=True, help='Path to 3D CFD HDF5 file')
    args = parser.parse_args()

    convert_3d_cfd(args.input)


if __name__ == "__main__":
    main()
