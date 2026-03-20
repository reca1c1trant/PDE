"""
Preprocess Active Matter HDF5 data to our finetune HDF5 format.

Input: active_matter_L_10.0_zeta_*.hdf5 files from polymathic-ai/active_matter
Output: HDF5 with vector (N, T, H, W, 3) and scalar (N, T, H, W, 4)

Channel mapping:
  vector[..., 0] = velocity_x, vector[..., 1] = velocity_y, vector[..., 2] = 0 (pad)
  scalar[..., 0] = concentration → scalar_indices[0] = 0 → channel 3
  scalar[..., 1] = D_xx          → scalar_indices[1] = 1 → channel 4
  scalar[..., 2] = D_xy          → scalar_indices[2] = 2 → channel 5
  scalar[..., 3] = D_yy          → scalar_indices[3] = 3 → channel 6

D is symmetric (D_yx = D_xy), so only 3 independent components stored.

Usage:
    python tools/preprocess_active_matter.py \
        --input_dir /scratch-share/SONG0304/finetune/active_matter/data/train \
        --output /scratch-share/SONG0304/finetune/active_matter_zeta1.hdf5
"""

import argparse
import h5py
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Preprocess Active Matter → HDF5")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing active_matter_*.hdf5 files")
    parser.add_argument("--output", type=str, required=True,
                        help="Output HDF5 path")
    parser.add_argument("--pattern", type=str, default="active_matter_L_10.0_zeta_1.0_alpha_*.hdf5",
                        help="Glob pattern for input files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    hdf5_files = sorted(input_dir.glob(args.pattern))
    if not hdf5_files:
        raise FileNotFoundError(f"No files matching {args.pattern} in {input_dir}")
    print(f"Processing {len(hdf5_files)} files from {input_dir}")

    # Count total samples and get dimensions
    sample_counts = []
    for fpath in hdf5_files:
        with h5py.File(str(fpath), "r") as f:
            n_samples = f["t1_fields/velocity"].shape[0]
            sample_counts.append(n_samples)
            print(f"  {fpath.name}: {n_samples} samples")

    total_samples = sum(sample_counts)

    # Get dims from first file
    with h5py.File(str(hdf5_files[0]), "r") as f:
        _, n_time, nx, ny, _ = f["t1_fields/velocity"].shape
        print(f"  Grid: {nx}x{ny}, T={n_time}")

    print(f"  Total samples: {total_samples}")

    # Create output HDF5
    with h5py.File(args.output, "w") as out:
        vec_ds = out.create_dataset(
            "vector", shape=(total_samples, n_time, nx, ny, 3),
            dtype=np.float32, chunks=(1, n_time, nx, ny, 3),
        )
        # scalar: concentration only (no D tensor)
        sca_ds = out.create_dataset(
            "scalar", shape=(total_samples, n_time, nx, ny, 1),
            dtype=np.float32, chunks=(1, n_time, nx, ny, 1),
        )
        out.create_dataset("scalar_indices", data=np.array([0], dtype=np.int64))

        # Store per-sample parameters: zeta*alpha (in nu field for PDE loss)
        nu_ds = out.create_dataset("nu", shape=(total_samples,), dtype=np.float32)

        offset = 0
        for fpath in hdf5_files:
            print(f"  Loading {fpath.name}...")
            with h5py.File(str(fpath), "r") as f:
                n_s = f["t1_fields/velocity"].shape[0]
                alpha_val = float(f["scalars/alpha"][()])
                zeta_val = float(f["scalars/zeta"][()])

                vel = f["t1_fields/velocity"][:].astype(np.float32)  # (N, T, H, W, 2)
                conc = f["t0_fields/concentration"][:].astype(np.float32)  # (N, T, H, W)

                # vector: (N, T, H, W, 3)
                vec_data = np.zeros((n_s, n_time, nx, ny, 3), dtype=np.float32)
                vec_data[..., 0] = vel[..., 0]  # vx
                vec_data[..., 1] = vel[..., 1]  # vy

                # scalar: (N, T, H, W, 1) - concentration only
                sca_data = conc[..., np.newaxis]  # (N, T, H, W, 1)

                vec_ds[offset:offset + n_s] = vec_data
                sca_ds[offset:offset + n_s] = sca_data
                nu_ds[offset:offset + n_s] = zeta_val * alpha_val

                print(f"    alpha={alpha_val}, zeta={zeta_val}")
                print(f"    vx: [{vel[...,0].min():.4f}, {vel[...,0].max():.4f}]")
                print(f"    conc: [{conc.min():.4f}, {conc.max():.4f}]")

                offset += n_s

    print(f"\nSaved: {args.output}")
    print(f"  vector: ({total_samples}, {n_time}, {nx}, {ny}, 3)")
    print(f"  scalar: ({total_samples}, {n_time}, {nx}, {ny}, 1)")
    print(f"  scalar_indices: [0]")
    print(f"  nu: ({total_samples},) [zeta*alpha values]")


if __name__ == "__main__":
    main()
