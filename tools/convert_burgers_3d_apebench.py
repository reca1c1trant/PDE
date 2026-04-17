"""
Convert raw APEBench 3D Burgers .npy files to our HDF5 finetune format.

Input:  /scratch-share/SONG0304/finetune/burgers_3d_raw/{train,test}.npy
        APEBench shape: [N, T, C, X, Y, Z] (C=3 channels: u, v, w)

Output: /scratch-share/SONG0304/finetune/burgers_3d_apebench.hdf5
        vector: [N_total, T, X, Y, Z, 3]  (our format)
        nu: [N_total] (0=train, 1=test, for splitting)

Usage (in icml conda environment):
    conda activate icml
    python tools/convert_burgers_3d_apebench.py
"""

import numpy as np
import h5py
import os

RAW_DIR = "/scratch-share/SONG0304/finetune/burgers_3d_raw"
OUTPUT_PATH = "/scratch-share/SONG0304/finetune/burgers_3d_apebench.hdf5"

print("=" * 60)
print("Convert 3D Burgers APEBench → HDF5")
print("=" * 60)

# Load raw data
train = np.load(os.path.join(RAW_DIR, "train.npy"))  # [50, 51, 3, 32, 32, 32]
test = np.load(os.path.join(RAW_DIR, "test.npy"))     # [10, 109, 3, 32, 32, 32]

print(f"  train: {train.shape}, range=[{train.min():.4f}, {train.max():.4f}]")
print(f"  test:  {test.shape}, range=[{test.min():.4f}, {test.max():.4f}]")

# Rearrange: [N, T, C, X, Y, Z] → [N, T, X, Y, Z, C]
train_vec = np.transpose(train, (0, 1, 3, 4, 5, 2))  # [50, 51, 32, 32, 32, 3]
test_vec = np.transpose(test, (0, 1, 3, 4, 5, 2))     # [10, 109, 32, 32, 32, 3]

print(f"\n  train_vec: {train_vec.shape}")
print(f"  test_vec:  {test_vec.shape}")

# Pad train to 109 timesteps (fill with NaN to mark padding)
# Actually: don't pad. Store train and test separately in HDF5.
# Our FinetuneDataset uses train_ratio to split by sample index, not by file.
# So store all 60 samples. But train has 51 steps, test has 109 steps — different T!
#
# Solution: pad train from 51 → 109 timesteps by repeating last frame
# (the model only uses t_input=8 window anyway, so padding at end is fine)
T_max = test_vec.shape[1]  # 109
T_train = train_vec.shape[1]  # 51

if T_train < T_max:
    pad_frames = T_max - T_train
    # Repeat last frame
    last_frame = train_vec[:, -1:, ...]  # [50, 1, 32, 32, 32, 3]
    padding = np.repeat(last_frame, pad_frames, axis=1)
    train_vec_padded = np.concatenate([train_vec, padding], axis=1)  # [50, 109, 32, 32, 32, 3]
    print(f"  Padded train: {train_vec.shape} → {train_vec_padded.shape}")
else:
    train_vec_padded = train_vec

# Combine
all_vec = np.concatenate([train_vec_padded, test_vec], axis=0)  # [60, 109, 32, 32, 32, 3]
print(f"\n  Combined: {all_vec.shape}")

# nu: mark train (0) vs test (1) for reference
nu = np.zeros(len(all_vec), dtype=np.float32)
nu[len(train_vec_padded):] = 1.0

# Save HDF5
with h5py.File(OUTPUT_PATH, 'w') as f:
    f.create_dataset('vector', data=all_vec.astype(np.float32))
    f.create_dataset('nu', data=nu)
    # No scalar channels for Burgers (vector-only)

file_size = os.path.getsize(OUTPUT_PATH) / 1e6
print(f"\nSaved: {OUTPUT_PATH}")
print(f"  vector: {all_vec.shape}, dtype=float32")
print(f"  nu: {nu.shape} (0=train[{len(train_vec_padded)}], 1=test[{len(test_vec)}])")
print(f"  File size: {file_size:.1f} MB")

# Cleanup raw files
print(f"\nCleaning up {RAW_DIR}/...")
os.remove(os.path.join(RAW_DIR, "train.npy"))
os.remove(os.path.join(RAW_DIR, "test.npy"))
os.rmdir(RAW_DIR)
print("Done!")
