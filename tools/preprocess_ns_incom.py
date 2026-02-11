"""
Preprocess NS_incom (Incompressible Navier-Stokes) dataset to unified format.

Original format (multiple files, flat arrays):
    force: [4, 512, 512, 2]           # external force (time-invariant)
    particles: [4, 1000, 512, 512, 1] # passive tracer
    velocity: [4, 1000, 512, 512, 2]  # [vx, vy]
    t: [4, 1000]                      # time

New unified format (single merged file):
    vector: [N, T, H, W, 3]           # [vx, vy, 0]
    scalar: [N, T, H, W, 1]           # particles (passive_tracer)
    scalar_indices: [11]              # passive_tracer
    force: [N, H, W, 2]               # external force (time-invariant, stored separately)

Note: Spatial resolution kept at 512x512. Downsampling handled in dataset loader.

Usage:
    python preprocess_ns_incom.py --input_dir /path/to/NS_incom

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


def convert_ns_incom(input_dir: str, output_dir: str = None):
    """
    Convert NS_incom dataset to unified format.

    Merges all files in input_dir into a single output file.

    Args:
        input_dir: Directory containing NS_incom HDF5 files
        output_dir: Output directory (default: same as input_dir)
    """
    input_dir = Path(input_dir)

    # Find all HDF5 files matching ns_incom_inhom_2d_512-*.h5
    h5_files = sorted(
        list(input_dir.glob("ns_incom_inhom_2d_512-*.h5")) +
        list(input_dir.glob("ns_incom_inhom_2d_512-*.hdf5")),
        key=lambda x: int(x.stem.split('-')[-1])  # Sort by number suffix
    )
    if not h5_files:
        raise ValueError(f"No NS_incom HDF5 files found in {input_dir}")

    print(f"Found {len(h5_files)} files:")
    for f in h5_files:
        print(f"  {f.name}")

    # Output directory
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'ns_incom_inhom_2d_512.hdf5'

    # Scalar indices for NS_incom: passive_tracer (11)
    scalar_indices = np.array([
        SCALAR_INDICES['passive_tracer']
    ], dtype=np.int32)

    print(f"\nOutput: {output_path}")
    print(f"Scalar indices: {scalar_indices.tolist()}")
    print(f"  channel 0 -> passive_tracer (global idx {SCALAR_INDICES['passive_tracer']})")
    print(f"Vector: [vx, vy, 0]")
    print(f"Force: stored separately (time-invariant)")

    # First pass: count total samples and verify dimensions
    total_samples = 0
    T, H, W = None, None, None

    print(f"\nScanning files...")
    for h5_file in h5_files:
        with h5py.File(h5_file, 'r') as f:
            vel_shape = f['velocity'].shape  # [N, T, H, W, 2]
            n, t, h, w, c = vel_shape

            if T is None:
                T, H, W = t, h, w
            else:
                assert (t, h, w) == (T, H, W), f"Shape mismatch in {h5_file}"

            total_samples += n
            print(f"  {h5_file.name}: {n} samples, shape [T={t}, H={h}, W={w}]")

    print(f"\nTotal: {total_samples} samples")
    print(f"Dimensions: T={T}, H={H}, W={W}")

    # Output shapes
    n_scalar_channels = len(scalar_indices)  # 1
    vector_shape = (total_samples, T, H, W, NUM_VECTOR_CHANNELS)
    scalar_shape = (total_samples, T, H, W, n_scalar_channels)
    force_shape = (total_samples, H, W, 2)  # time-invariant

    print(f"\nOutput shapes:")
    print(f"  vector: {vector_shape}")
    print(f"  scalar: {scalar_shape}")
    print(f"  force: {force_shape}")

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

        # Force: time-invariant, chunk by single sample
        force_ds = f_out.create_dataset(
            'force',
            shape=force_shape,
            dtype=np.float32,
            chunks=(1, H, W, 2),
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
                n_samples = f_in['velocity'].shape[0]

                # Process each sample in the file
                for i in tqdm(range(n_samples), desc=f"  {h5_file.name}"):
                    # Read data for this sample
                    velocity = f_in['velocity'][i]    # [T, H, W, 2]
                    particles = f_in['particles'][i]  # [T, H, W, 1]
                    force = f_in['force'][i]          # [H, W, 2]

                    # Create vector: [T, H, W, 3]
                    vx = velocity[..., 0:1]  # [T, H, W, 1]
                    vy = velocity[..., 1:2]  # [T, H, W, 1]
                    vz = np.zeros_like(vx)   # [T, H, W, 1]
                    vector_data = np.concatenate([vx, vy, vz], axis=-1)  # [T, H, W, 3]

                    # Scalar is already [T, H, W, 1]
                    scalar_data = particles

                    # Write to output
                    out_idx = sample_offset + i
                    vector_ds[out_idx] = vector_data
                    scalar_ds[out_idx] = scalar_data
                    force_ds[out_idx] = force

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
        force = f_out['force'][0]
        indices = f_out['scalar_indices'][:]

        print(f"\nData check (sample 0):")
        print(f"  vector[..., 0] (vx): min={vector[..., 0].min():.4f}, max={vector[..., 0].max():.4f}")
        print(f"  vector[..., 1] (vy): min={vector[..., 1].min():.4f}, max={vector[..., 1].max():.4f}")
        print(f"  vector[..., 2] (vz): all zeros = {np.allclose(vector[..., 2], 0)}")
        print(f"  scalar[..., 0] (particles): min={scalar[..., 0].min():.4f}, max={scalar[..., 0].max():.4f}")
        print(f"  force: min={force.min():.4f}, max={force.max():.4f}")
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
    parser = argparse.ArgumentParser(description="Convert NS_incom dataset to unified format")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing NS_incom HDF5 files')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (default: same as input_dir)')
    args = parser.parse_args()

    convert_ns_incom(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
