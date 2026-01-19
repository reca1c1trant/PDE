"""
Check boundary data format and coordinates.
"""

import torch
import numpy as np
from dataset_burgers import BurgersDataset, burgers_collate_fn

# Load dataset
dataset = BurgersDataset(
    data_path="./burgers2d_nu0.1_0.15_res128_t1000_n100.h5",
    temporal_length=16,
    split='train',
    train_ratio=0.9,
    seed=42,
    clips_per_sample=1,
)

# Get one sample
sample = dataset[0]
print("=" * 60)
print("Dataset Sample Keys:")
print("=" * 60)
for key, value in sample.items():
    if isinstance(value, (torch.Tensor, np.ndarray)):
        print(f"  {key}: shape={value.shape}, dtype={value.dtype if hasattr(value, 'dtype') else type(value)}")
    else:
        print(f"  {key}: {value}")

# Get batch
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=2, collate_fn=burgers_collate_fn)
batch = next(iter(loader))

print("\n" + "=" * 60)
print("Batch Keys:")
print("=" * 60)
for key, value in batch.items():
    if isinstance(value, torch.Tensor):
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    else:
        print(f"  {key}: {value}")

# Check boundary data
print("\n" + "=" * 60)
print("Boundary Data Details:")
print("=" * 60)

if 'boundary_left' in batch:
    bl = batch['boundary_left']
    br = batch['boundary_right']
    bb = batch['boundary_bottom']
    bt = batch['boundary_top']

    print(f"\nboundary_left:   shape={bl.shape}")
    print(f"boundary_right:  shape={br.shape}")
    print(f"boundary_bottom: shape={bb.shape}")
    print(f"boundary_top:    shape={bt.shape}")

# Check data shape
data = batch['data']
print(f"\n" + "=" * 60)
print("Data Shape Analysis:")
print("=" * 60)
print(f"data shape: {data.shape}")  # [B, T, H, W, C]

B, T, H, W, C = data.shape
print(f"  B (batch): {B}")
print(f"  T (time):  {T}")
print(f"  H (height): {H}")
print(f"  W (width):  {W}")
print(f"  C (channels): {C}")

# Grid coordinates
print(f"\n" + "=" * 60)
print("Grid Coordinates:")
print("=" * 60)
dx = 1.0 / H
dy = 1.0 / W
print(f"dx = 1/{H} = {dx:.6f}")
print(f"dy = 1/{W} = {dy:.6f}")

print(f"\nX coordinates (first 5): {[i*dx for i in range(5)]}")
print(f"X coordinates (last 5):  {[i*dx for i in range(H-5, H)]}")
print(f"\nY coordinates (first 5): {[j*dy for j in range(5)]}")
print(f"Y coordinates (last 5):  {[j*dy for j in range(W-5, W)]}")

# Boundary indices
print(f"\n" + "=" * 60)
print("Boundary Definitions (indices):")
print("=" * 60)
print(f"Left boundary:   data[:, :, :, 0, :]     → x = 0")
print(f"Right boundary:  data[:, :, :, -1, :]    → x = {(W-1)*dx:.6f}")
print(f"Bottom boundary: data[:, :, 0, :, :]     → y = 0")
print(f"Top boundary:    data[:, :, -1, :, :]    → y = {(H-1)*dy:.6f}")

# What user mentioned: 1/256 to 255/256
print(f"\n" + "=" * 60)
print("User's Coordinate Range (1/256 to 255/256):")
print("=" * 60)
print(f"1/256   = {1/256:.6f}")
print(f"255/256 = {255/256:.6f}")
print(f"\nIf grid is 128x128:")
print(f"  Index 0   → x = 0")
print(f"  Index 1   → x = 1/128 = {1/128:.6f}")
print(f"  Index 126 → x = 126/128 = {126/128:.6f}")
print(f"  Index 127 → x = 127/128 = {127/128:.6f}")

print(f"\n" + "=" * 60)
print("Proposed Boundary Loss Definition:")
print("=" * 60)
print(f"Left boundary:   data[:, :, 1:-1, 0, :]   → y from 1/128 to 126/128 (exclude corners)")
print(f"Right boundary:  data[:, :, 1:-1, -1, :]  → y from 1/128 to 126/128 (exclude corners)")
print(f"Bottom boundary: data[:, :, 0, 1:-1, :]   → x from 1/128 to 126/128 (exclude corners)")
print(f"Top boundary:    data[:, :, -1, 1:-1, :]  → x from 1/128 to 126/128 (exclude corners)")

# Print actual boundary values
print(f"\n" + "=" * 60)
print("Actual Boundary Values (sample):")
print("=" * 60)
u = data[0, 0, :, :, 0]  # First sample, first timestep, u component
print(f"u field shape: {u.shape}")
print(f"\nLeft boundary (u[:, 0]):")
print(f"  shape: {u[:, 0].shape}")
print(f"  values (first 5): {u[:5, 0].tolist()}")

print(f"\nRight boundary (u[:, -1]):")
print(f"  shape: {u[:, -1].shape}")
print(f"  values (first 5): {u[:5, -1].tolist()}")

print(f"\nBottom boundary (u[0, :]):")
print(f"  shape: {u[0, :].shape}")
print(f"  values (first 5): {u[0, :5].tolist()}")

print(f"\nTop boundary (u[-1, :]):")
print(f"  shape: {u[-1, :].shape}")
print(f"  values (first 5): {u[-1, :5].tolist()}")
