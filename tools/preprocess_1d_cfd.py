"""
Preprocess 1D CFD dataset to unified format.

Original format (single file, raw fields):
    Vx: [10000, 101, 1024]
    density: [10000, 101, 1024]
    pressure: [10000, 101, 1024]
    t-coordinate: [102]
    x-coordinate: [1024]

New unified format:
    vector: [N, T, X, 3]          # [Vx, 0, 0]
    scalar: [N, T, X, 2]          # [density, pressure]
    scalar_indices: [2]            # [4, 12] -> density, pressure

Usage:
    python preprocess_1d_cfd.py --input /scratch-share/SONG0304/pretrained/1D_CFD_Rand_Eta0.1_Zeta0.1_periodic_Train.hdf5

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


def convert_1d_cfd(input_path: str):
    """
    Convert 1D CFD dataset to unified format.

    Args:
        input_path: Path to the 1D CFD HDF5 file
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Output to same directory
    output_path = input_path.parent / '1D_CFD_unified.hdf5'

    # Scalar indices for 1D CFD: density (4), pressure (12)
    scalar_indices = np.array([
        SCALAR_INDICES['density'],
        SCALAR_INDICES['pressure'],
    ], dtype=np.int32)

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Scalar indices: {scalar_indices.tolist()}")
    print(f"  channel 0 -> density (global idx {SCALAR_INDICES['density']})")
    print(f"  channel 1 -> pressure (global idx {SCALAR_INDICES['pressure']})")
    print(f"Vector: [Vx, 0, 0] (1D velocity)")

    # Read metadata
    with h5py.File(input_path, 'r') as f_in:
        print(f"\nOriginal structure:")
        for key in f_in.keys():
            data = f_in[key]
            print(f"  {key}: shape={data.shape}, dtype={data.dtype}")

        vx_shape = f_in['Vx'].shape  # [N, T, X]
        N, T, X = vx_shape

    print(f"\nDimensions: N={N}, T={T}, X={X}")

    # Output shapes
    n_scalar_channels = len(scalar_indices)  # 2
    vector_shape = (N, T, X, NUM_VECTOR_CHANNELS)
    scalar_shape = (N, T, X, n_scalar_channels)

    print(f"\nOutput shapes:")
    print(f"  vector: {vector_shape}")
    print(f"  scalar: {scalar_shape}")

    print(f"\n{'='*60}")
    print("Starting conversion...")
    print(f"{'='*60}\n")

    with h5py.File(output_path, 'w') as f_out:
        # Create datasets with chunking
        vector_ds = f_out.create_dataset(
            'vector',
            shape=vector_shape,
            dtype=np.float32,
            chunks=(1, T, X, NUM_VECTOR_CHANNELS),
            compression='gzip',
            compression_opts=4,
        )

        scalar_ds = f_out.create_dataset(
            'scalar',
            shape=scalar_shape,
            dtype=np.float32,
            chunks=(1, T, X, n_scalar_channels),
            compression='gzip',
            compression_opts=4,
        )

        # Store scalar indices
        f_out.create_dataset('scalar_indices', data=scalar_indices)

        # Convert in batches
        batch_size = 200
        with h5py.File(input_path, 'r') as f_in:
            for start in tqdm(range(0, N, batch_size), desc="Converting"):
                end = min(start + batch_size, N)

                # Read batch: [batch, T, X]
                vx = f_in['Vx'][start:end]
                density = f_in['density'][start:end]
                pressure = f_in['pressure'][start:end]

                # Create vector: [batch, T, X, 3] = [Vx, 0, 0]
                vy = np.zeros_like(vx)
                vz = np.zeros_like(vx)
                vector_batch = np.stack([vx, vy, vz], axis=-1)

                # Create scalar: [batch, T, X, 2] = [density, pressure]
                scalar_batch = np.stack([density, pressure], axis=-1)

                # Write
                vector_ds[start:end] = vector_batch
                scalar_ds[start:end] = scalar_batch

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
        print(f"  vector[..., 1] (Vy): all zeros = {np.allclose(vector[..., 1], 0)}")
        print(f"  vector[..., 2] (Vz): all zeros = {np.allclose(vector[..., 2], 0)}")
        print(f"  scalar[..., 0] (density): min={scalar[..., 0].min():.4f}, max={scalar[..., 0].max():.4f}")
        print(f"  scalar[..., 1] (pressure): min={scalar[..., 1].min():.4f}, max={scalar[..., 1].max():.4f}")
        print(f"  scalar_indices: {indices.tolist()}")

    print(f"\nOriginal file preserved: {input_path}")
    print(f"\n{'='*60}")
    print(f"Done! Output: {output_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Convert 1D CFD dataset to unified format")
    parser.add_argument('--input', type=str, required=True, help='Path to 1D CFD HDF5 file')
    args = parser.parse_args()

    convert_1d_cfd(args.input)


if __name__ == "__main__":
    main()
