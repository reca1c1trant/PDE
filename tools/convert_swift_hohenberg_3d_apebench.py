"""
Convert raw APEBench 3D Swift-Hohenberg .npy to our HDF5 finetune format.

Input:  /scratch-share/SONG0304/finetune/swift_hohenberg_3d_raw/test.npy
        APEBench shape: [30, 201, 1, 32, 32, 32]

Output: /scratch-share/SONG0304/finetune/swift_hohenberg_3d_apebench.hdf5
        scalar: [30, 201, 32, 32, 32, 1]
        vector: [30, 201, 32, 32, 32, 3] (zeros)
        scalar_indices: [0]

Usage (in icml conda environment):
    conda activate icml
    python tools/convert_swift_hohenberg_3d_apebench.py
"""

import numpy as np
import h5py
import os
import shutil

RAW_DIR = "/scratch-share/SONG0304/finetune/swift_hohenberg_3d_raw"
OUTPUT_PATH = "/scratch-share/SONG0304/finetune/swift_hohenberg_3d_apebench.hdf5"

print("=" * 60)
print("Convert 3D Swift-Hohenberg APEBench → HDF5")
print("=" * 60)

# Load raw data
test = np.load(os.path.join(RAW_DIR, "test.npy"))  # [30, 201, 1, 32, 32, 32]
print(f"  test: {test.shape}, range=[{test.min():.4f}, {test.max():.4f}]")

# Rearrange: [N, T, C, X, Y, Z] → scalar [N, T, X, Y, Z, C]
scalar = np.transpose(test, (0, 1, 3, 4, 5, 2))  # [30, 201, 32, 32, 32, 1]
vector = np.zeros((*scalar.shape[:4], scalar.shape[4], 3), dtype=np.float32)  # [30, 201, 32, 32, 32, 3]

print(f"  scalar: {scalar.shape}")
print(f"  vector: {vector.shape}")

# nu: placeholder
nu = np.zeros(len(scalar), dtype=np.float32)

# Save HDF5
with h5py.File(OUTPUT_PATH, 'w') as f:
    f.create_dataset('scalar', data=scalar.astype(np.float32))
    f.create_dataset('vector', data=vector)
    f.create_dataset('scalar_indices', data=np.array([0], dtype=np.int64))
    f.create_dataset('nu', data=nu)

file_size = os.path.getsize(OUTPUT_PATH) / 1e6
print(f"\nSaved: {OUTPUT_PATH}")
print(f"  scalar: {scalar.shape}, dtype=float32")
print(f"  File size: {file_size:.1f} MB")

# Cleanup raw
shutil.rmtree(RAW_DIR)
print(f"Cleaned up {RAW_DIR}")
print("Done!")
