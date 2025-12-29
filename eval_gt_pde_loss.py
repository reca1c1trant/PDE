"""
Evaluate PDE loss on Ground Truth data.

This script computes the PDE residual loss on the actual simulation data
to establish a baseline. If GT data has high PDE loss, it indicates
numerical discretization error rather than model failure.

Usage:
    python eval_gt_pde_loss.py
"""

import h5py
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from pde_loss import burgers_pde_loss_upwind


def evaluate_gt_pde_loss(
    data_path: str,
    temporal_length: int = 17,  # Same as training (16 + 1)
    batch_size: int = 8,
    dt: float = 1/999,
):
    """
    Evaluate PDE loss on ground truth data.

    Args:
        data_path: Path to HDF5 file
        temporal_length: Number of frames per clip (17 for fair comparison)
        batch_size: Batch size for evaluation
        dt: Time step (must match training)
    """
    print(f"\n{'='*60}")
    print(f"Evaluating Ground Truth PDE Loss")
    print(f"{'='*60}")
    print(f"Data: {data_path}")
    print(f"Temporal length: {temporal_length}")
    print(f"Batch size: {batch_size}")
    print(f"dt: {dt}")
    print(f"{'='*60}\n")

    # Collect all clips
    all_clips = []  # List of (data, boundaries, nu)

    with h5py.File(data_path, 'r') as f:
        sample_keys = sorted(f.keys(), key=lambda x: int(x))
        print(f"Found {len(sample_keys)} samples")

        for sample_key in tqdm(sample_keys, desc="Loading samples"):
            grp = f[sample_key]
            nu = float(grp['nu'][()])

            # Load full trajectory [T=1000, H=128, W=128, C=2]
            data_full = np.array(grp['vector']['data'], dtype=np.float32)
            total_t = data_full.shape[0]

            # Load full boundaries
            bnd = grp['vector']['boundary']
            boundary_left_full = np.array(bnd['left'], dtype=np.float32)
            boundary_right_full = np.array(bnd['right'], dtype=np.float32)
            boundary_bottom_full = np.array(bnd['bottom'], dtype=np.float32)
            boundary_top_full = np.array(bnd['top'], dtype=np.float32)

            # Slide window to get all possible clips
            max_start = total_t - temporal_length
            for start_t in range(0, max_start + 1):
                end_t = start_t + temporal_length

                # Extract clip
                data_clip = data_full[start_t:end_t]  # [17, H, W, 2]
                bl = boundary_left_full[start_t:end_t]
                br = boundary_right_full[start_t:end_t]
                bb = boundary_bottom_full[start_t:end_t]
                bt = boundary_top_full[start_t:end_t]

                all_clips.append({
                    'data': data_clip,
                    'boundary_left': bl,
                    'boundary_right': br,
                    'boundary_bottom': bb,
                    'boundary_top': bt,
                    'nu': nu,
                })

    print(f"\nTotal clips: {len(all_clips)}")
    print(f"  = {len(sample_keys)} samples × {max_start + 1} clips/sample")

    # Evaluate in batches
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    total_pde_loss = 0.0
    total_loss_u = 0.0
    total_loss_v = 0.0
    total_constraint_error = 0.0
    num_batches = 0

    # Process in batches
    num_clips = len(all_clips)

    for batch_start in tqdm(range(0, num_clips, batch_size), desc="Evaluating"):
        batch_end = min(batch_start + batch_size, num_clips)
        batch_clips = all_clips[batch_start:batch_end]
        actual_batch_size = len(batch_clips)

        # Stack batch
        data_batch = np.stack([c['data'] for c in batch_clips], axis=0)  # [B, 17, H, W, 2]
        bl_batch = np.stack([c['boundary_left'] for c in batch_clips], axis=0)
        br_batch = np.stack([c['boundary_right'] for c in batch_clips], axis=0)
        bb_batch = np.stack([c['boundary_bottom'] for c in batch_clips], axis=0)
        bt_batch = np.stack([c['boundary_top'] for c in batch_clips], axis=0)
        nu_batch = np.array([c['nu'] for c in batch_clips])

        # Convert to torch
        data_t = torch.from_numpy(data_batch).to(device)  # [B, 17, H, W, 2]
        bl_t = torch.from_numpy(bl_batch).to(device)
        br_t = torch.from_numpy(br_batch).to(device)
        bb_t = torch.from_numpy(bb_batch).to(device)
        bt_t = torch.from_numpy(bt_batch).to(device)
        nu_mean = nu_batch.mean()

        # Compute PDE loss (same as training)
        with torch.no_grad():
            pde_loss, loss_u, loss_v, _, _ = burgers_pde_loss_upwind(
                pred=data_t,  # [B, 17, H, W, 2] - GT data as "prediction"
                boundary_left=bl_t,
                boundary_right=br_t,
                boundary_bottom=bb_t,
                boundary_top=bt_t,
                nu=nu_mean,
                dt=dt
            )

            # Constraint error: u + v = 1.5
            u = data_t[..., 0]
            v = data_t[..., 1]
            constraint = torch.mean(torch.abs(u + v - 1.5))

        total_pde_loss += pde_loss.item()
        total_loss_u += loss_u.item()
        total_loss_v += loss_v.item()
        total_constraint_error += constraint.item()
        num_batches += 1

    # Average
    avg_pde_loss = total_pde_loss / num_batches
    avg_loss_u = total_loss_u / num_batches
    avg_loss_v = total_loss_v / num_batches
    avg_constraint_error = total_constraint_error / num_batches

    # Print results
    print(f"\n{'='*60}")
    print(f"Ground Truth PDE Loss Results")
    print(f"{'='*60}")
    print(f"Total clips evaluated: {num_clips}")
    print(f"Number of batches: {num_batches}")
    print(f"{'-'*60}")
    print(f"Average PDE Loss:        {avg_pde_loss:.6f}")
    print(f"  - Loss U:              {avg_loss_u:.6f}")
    print(f"  - Loss V:              {avg_loss_v:.6f}")
    print(f"Constraint Error (|u+v-1.5|): {avg_constraint_error:.6f}")
    print(f"{'='*60}\n")

    return {
        'pde_loss': avg_pde_loss,
        'loss_u': avg_loss_u,
        'loss_v': avg_loss_v,
        'constraint_error': avg_constraint_error,
        'num_clips': num_clips,
    }


if __name__ == "__main__":
    data_path = "./burgers2d_nu0.1_0.15_res128_t1000_n100.h5"

    if not Path(data_path).exists():
        print(f"Data file not found: {data_path}")
        exit(1)

    results = evaluate_gt_pde_loss(
        data_path=data_path,
        temporal_length=17,  # Same as training
        batch_size=8,        # Same as training
        dt=1/999,            # Same as training
    )
