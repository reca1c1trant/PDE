"""
Test script for PDE Causal AR model evaluation.

nRMSE = sqrt(MSE(pred, gt) / MSE(0, gt))

Usage:
    python test.py --config configs/test_v2.yaml
"""

import os
import argparse
import random
import numpy as np
import torch
import yaml
from pathlib import Path
from tqdm import tqdm

from pipeline import PDECausalModel
from dataset import PDEDataset


def parse_args():
    parser = argparse.ArgumentParser(description="PDE Causal AR Testing")
    parser.add_argument('--config', type=str, default='configs/test_v2.yaml',
                        help='Path to test config file')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(checkpoint_path: str, model_config: dict, device: torch.device, dtype: torch.dtype = torch.bfloat16):
    """Load model from checkpoint."""
    model = PDECausalModel(model_config)

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']

    # Handle DDP/FSDP wrapped state dict
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        if k.startswith('_orig_mod.'):
            k = k[10:]
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model = model.to(device=device, dtype=dtype)
    model.eval()

    return model, dtype


def compute_nrmse(pred: torch.Tensor, gt: torch.Tensor, channel_mask: torch.Tensor):
    """
    Compute nRMSE = sqrt(MSE(pred, gt) / MSE(0, gt)) per channel and overall.

    Args:
        pred: [B, H, W, C] predictions
        gt: [B, H, W, C] ground truth
        channel_mask: [C] valid channel mask

    Returns:
        nrmse_per_channel: [C_valid] nRMSE per valid channel
        nrmse_overall: scalar overall nRMSE (computed across all valid channels together)
    """
    # Filter valid channels
    valid_mask = channel_mask.bool()
    pred_valid = pred[..., valid_mask]  # [B, H, W, C_valid]
    gt_valid = gt[..., valid_mask]  # [B, H, W, C_valid]

    C_valid = pred_valid.shape[-1]

    # Flatten spatial dims: [B, H*W, C_valid]
    pred_flat = pred_valid.reshape(pred_valid.shape[0], -1, C_valid)
    gt_flat = gt_valid.reshape(gt_valid.shape[0], -1, C_valid)

    # Per-channel nRMSE
    mse_pred_gt_per_ch = ((pred_flat - gt_flat) ** 2).mean(dim=(0, 1))  # [C_valid]
    mse_zero_gt_per_ch = (gt_flat ** 2).mean(dim=(0, 1))  # [C_valid]
    nrmse_per_channel = torch.sqrt(mse_pred_gt_per_ch / (mse_zero_gt_per_ch + 1e-8))  # [C_valid]

    # Overall nRMSE: compute across all channels together (not mean of per-channel)
    mse_pred_gt_all = ((pred_valid - gt_valid) ** 2).mean()  # scalar
    mse_zero_gt_all = (gt_valid ** 2).mean()  # scalar
    nrmse_overall = torch.sqrt(mse_pred_gt_all / (mse_zero_gt_all + 1e-8))

    return nrmse_per_channel, nrmse_overall


def main():
    args = parse_args()

    # Load configs
    test_config = load_config(args.config)
    model_config = load_config(test_config['model_config'])

    # Set seed
    seed = test_config['test'].get('seed', 42)
    set_seed(seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    checkpoint_path = test_config['checkpoint']
    print(f"Checkpoint: {checkpoint_path}")
    model, model_dtype = load_model(checkpoint_path, model_config, device)
    print("Model loaded")

    # Create dataset (full temporal length)
    base_dataset = PDEDataset(
        data_dir=test_config['dataset']['path'],
        temporal_length=16,  
        split='val',
        train_ratio=test_config['dataset'].get('train_ratio', 0.9),
        seed=seed,
    )

    # Test config
    input_steps = test_config['test']['input_steps']
    batch_size = test_config['test'].get('batch_size', 8)
    num_samples = test_config['test'].get('num_samples', len(base_dataset))
    num_samples = min(num_samples, len(base_dataset))

    print(f"Samples: {num_samples} | Input steps: {input_steps} | Batch size: {batch_size}")

    # Evaluation
    all_nrmse_per_channel = []
    all_nrmse_overall = []
    valid_mask = None
    rng = random.Random(seed)

    print("\nEvaluating...")
    with torch.no_grad():
        for sample_idx in tqdm(range(num_samples)):
            sample = base_dataset[sample_idx]
            data = sample['data']  # [T, H, W, C]
            channel_mask = sample['channel_mask']  # [C]

            T = data.shape[0]
            max_start = T - input_steps - 1

            if max_start < 0:
                print(f"Warning: Sample {sample_idx} has T={T}, skipping (need at least {input_steps + 1})")
                continue

            # Generate batch_size random start points
            starts = [rng.randint(0, max_start) for _ in range(batch_size)]

            # Build batch
            inputs = []
            gts = []
            for start in starts:
                inp = data[start : start + input_steps]  # [input_steps, H, W, C]
                gt = data[start + input_steps]  # [H, W, C]
                inputs.append(inp)
                gts.append(gt)

            input_batch = torch.stack(inputs, dim=0)  # [B, input_steps, H, W, C]
            gt_batch = torch.stack(gts, dim=0)  # [B, H, W, C]

            # Move to device
            input_batch = input_batch.to(device=device, dtype=model_dtype)
            gt_batch = gt_batch.to(device=device, dtype=model_dtype)
            channel_mask_device = channel_mask.to(device=device)

            # Forward
            output = model(input_batch)  # [B, input_steps, H, W, C]
            pred = output[:, -1]  # [B, H, W, C] - last timestep

            # Save valid mask
            if valid_mask is None:
                valid_mask = channel_mask.bool().cpu()

            # Compute nRMSE for this batch
            nrmse_per_ch, nrmse_overall = compute_nrmse(pred.float(), gt_batch.float(), channel_mask_device)

            all_nrmse_per_channel.append(nrmse_per_ch.cpu())
            all_nrmse_overall.append(nrmse_overall.cpu().item())

    # Average nRMSE across all samples
    nrmse_per_channel = torch.stack(all_nrmse_per_channel, dim=0).mean(dim=0)  # [C_valid]
    nrmse_overall = np.mean(all_nrmse_overall)

    # Channel names
    full_names = ['vx', 'vy', 'vz', 'p', 'rho', 'T']
    channel_names = [n for n, v in zip(full_names, valid_mask.tolist()) if v]

    # Print results
    print("\n" + "=" * 50)
    print("              Test Results")
    print("=" * 50)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Samples: {num_samples} | Input steps: {input_steps} | Batch size: {batch_size}")
    print()
    print("Per-channel nRMSE:")
    print("-" * 50)
    print(f"{'Channel':<10} {'nRMSE':<12}")
    print("-" * 50)
    for i, name in enumerate(channel_names):
        print(f"{name:<10} {nrmse_per_channel[i].item():<12.6f}")
    print("-" * 50)
    print()
    print(f"Overall nRMSE: {nrmse_overall:.6f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
