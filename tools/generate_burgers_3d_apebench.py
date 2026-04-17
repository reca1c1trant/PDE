"""
Generate 3D Burgers dataset using APEBench ETDRK solver.

Config (from APEBench paper Section H.6, Table 13):
  - PDE: Vector Burgers 3D (3 channels: u, v, w)
  - gammas = (0, 0, 1.5, 0, 0), convection_delta = -1.5
  - Resolution: 32^3, Domain: [0,1]^3, periodic BC
  - Train: 50 trajectories, 51 snapshots
  - Test: 10 trajectories, 109 snapshots
  - IC: truncated Fourier K=5, clamp |u| <= 1

Usage (in apebench conda environment):
    conda activate apebench
    python tools/generate_burgers_3d_apebench.py
"""

import numpy as np
import os
import time

import apebench
import jax

# ========== Config ==========
NUM_SPATIAL_DIMS = 3
NUM_POINTS = 32
TRAIN_SAMPLES = 50
TEST_SAMPLES = 10
TRAIN_HORIZON = 50    # 51 snapshots (IC + 50 steps)
TEST_HORIZON = 108    # 109 snapshots (IC + 108 steps)

RAW_DIR = "/scratch-share/SONG0304/finetune/burgers_3d_raw"

# ========== Create scenario ==========
scene = apebench.scenarios.difficulty.Burgers(
    num_spatial_dims=NUM_SPATIAL_DIMS,
    num_points=NUM_POINTS,
    num_train_samples=TRAIN_SAMPLES,
    num_test_samples=TEST_SAMPLES,
    train_temporal_horizon=TRAIN_HORIZON,
    test_temporal_horizon=TEST_HORIZON,
)

print("=" * 60)
print("3D Burgers Dataset Generation (APEBench ETDRK)")
print("=" * 60)
print(f"  Scenario: {scene.get_scenario_name()}")
print(f"  num_channels: {scene.num_channels}")
print(f"  num_points: {scene.num_points} (grid: {scene.num_points}^3)")
print(f"  gammas: {scene.gammas}")
print(f"  convection_delta: {scene.convection_delta}")
print(f"  conservative: {scene.conservative}")
print(f"  Train: {TRAIN_SAMPLES} samples x {TRAIN_HORIZON+1} snapshots")
print(f"  Test:  {TEST_SAMPLES} samples x {TEST_HORIZON+1} snapshots")
print("=" * 60)

stepper = scene.get_ref_stepper()

# ========== Generate train data ==========
print(f"\n[1/2] Generating train data...")
t0 = time.time()
train_data = scene.produce_data(
    stepper=stepper,
    num_samples=TRAIN_SAMPLES,
    num_warmup_steps=scene.num_warmup_steps,
    temporal_horizon=TRAIN_HORIZON,
    key=jax.random.PRNGKey(scene.train_seed),
)
train_np = np.array(train_data).astype(np.float32)
print(f"  Shape: {train_np.shape}")  # [50, 51, 3, 32, 32, 32]
print(f"  Range: [{train_np.min():.4f}, {train_np.max():.4f}]")
print(f"  Time: {time.time()-t0:.1f}s")

# ========== Generate test data ==========
print(f"\n[2/2] Generating test data...")
t0 = time.time()
test_data = scene.produce_data(
    stepper=stepper,
    num_samples=TEST_SAMPLES,
    num_warmup_steps=scene.num_warmup_steps,
    temporal_horizon=TEST_HORIZON,
    key=jax.random.PRNGKey(scene.test_seed),
)
test_np = np.array(test_data).astype(np.float32)
print(f"  Shape: {test_np.shape}")  # [10, 109, 3, 32, 32, 32]
print(f"  Range: [{test_np.min():.4f}, {test_np.max():.4f}]")
print(f"  Time: {time.time()-t0:.1f}s")

# ========== Save raw npy ==========
os.makedirs(RAW_DIR, exist_ok=True)
np.save(os.path.join(RAW_DIR, "train.npy"), train_np)
np.save(os.path.join(RAW_DIR, "test.npy"), test_np)

print(f"\nSaved raw data to {RAW_DIR}/")
print(f"  train.npy: {train_np.shape}, {os.path.getsize(os.path.join(RAW_DIR, 'train.npy'))/1e6:.1f} MB")
print(f"  test.npy: {test_np.shape}, {os.path.getsize(os.path.join(RAW_DIR, 'test.npy'))/1e6:.1f} MB")
print("\nDone! Switch to icml env and run convert script.")
