"""
Generate 3D Swift-Hohenberg dataset using APEBench ETDRK solver.

PDE: u_t = r*u - (k + Δ)²*u + u² - u³
  r = 0.7 (reactivity), k = 1.0 (critical number)
  Domain: [0, 10π]³, periodic BC, N=32³
  dt = 0.1, num_substeps = 5

Config (from APEBench paper Section H.6, Table 13):
  - 3D, N=32, 1 channel
  - Test: 30 trajectories, 201 snapshots (test_temporal_horizon=200)
  - Train: 50 trajectories, 51 snapshots (not used, for reference only)

Usage (in apebench conda environment):
    conda activate apebench
    python tools/generate_swift_hohenberg_3d_apebench.py
"""

import numpy as np
import os
import time

import apebench
import jax

# ========== Config ==========
NUM_SPATIAL_DIMS = 3
NUM_POINTS = 32
TEST_SAMPLES = 30
TEST_HORIZON = 200    # 201 snapshots

RAW_DIR = "/scratch-share/SONG0304/finetune/swift_hohenberg_3d_raw"

# ========== Create scenario ==========
scene = apebench.scenarios.physical.SwiftHohenberg(
    num_spatial_dims=NUM_SPATIAL_DIMS,
    num_points=NUM_POINTS,
    num_test_samples=TEST_SAMPLES,
    test_temporal_horizon=TEST_HORIZON,
)

print("=" * 60)
print("3D Swift-Hohenberg Dataset Generation (APEBench ETDRK)")
print("=" * 60)
print(f"  Scenario: {scene.get_scenario_name()}")
print(f"  num_channels: {scene.num_channels}")
print(f"  num_points: {scene.num_points} (grid: {scene.num_points}^3)")
print(f"  domain_extent: {scene.domain_extent}")
print(f"  dt: {scene.dt}")
print(f"  num_substeps: {scene.num_substeps}")
print(f"  reactivity: {scene.reactivity}")
print(f"  critical_number: {scene.critical_number}")
print(f"  polynomial_coefficients: {scene.polynomial_coefficients}")
print(f"  Test: {TEST_SAMPLES} samples x {TEST_HORIZON+1} snapshots")
print("=" * 60)

# ========== Generate test data only ==========
print(f"\nGenerating test data...")
t0 = time.time()
stepper = scene.get_ref_stepper()
test_data = scene.produce_data(
    stepper=stepper,
    num_samples=TEST_SAMPLES,
    num_warmup_steps=scene.num_warmup_steps,
    temporal_horizon=TEST_HORIZON,
    key=jax.random.PRNGKey(scene.test_seed),
)
# APEBench shape: [N, T+1, C, X, Y, Z]
test_np = np.array(test_data).astype(np.float32)
print(f"  Shape: {test_np.shape}")
print(f"  Range: [{test_np.min():.4f}, {test_np.max():.4f}]")
print(f"  Time: {time.time()-t0:.1f}s")

# ========== Save raw npy ==========
os.makedirs(RAW_DIR, exist_ok=True)
np.save(os.path.join(RAW_DIR, "test.npy"), test_np)

print(f"\nSaved raw data to {RAW_DIR}/")
print(f"  test.npy: {test_np.shape}, {os.path.getsize(os.path.join(RAW_DIR, 'test.npy'))/1e6:.1f} MB")
print("\nDone! Switch to icml env and run convert script.")
