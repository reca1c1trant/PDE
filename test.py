"""
Test script for PDE Causal AR model evaluation.
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
    parser.add_argument('--config', type=str, default='configs/test.yaml',
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


def load_model(checkpoint_path: str, model_config: dict, device: torch.device) -> PDECausalModel:
    """Load model from checkpoint."""
    model = PDECausalModel(model_config)

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Handle DDP/FSDP wrapped state dict
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove 'module.' or '_orig_mod.' prefix if present
        if k.startswith('module.'):
            k = k[7:]
        if k.startswith('_orig_mod.'):
            k = k[10:]
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()

    return model


def compute_metrics(pred: torch.Tensor, gt: torch.Tensor) -> dict:
    """
    Compute MSE and nRMSE per sample, then average over batch.

    Args:
        pred: [B, H, W, C] or [B, H, C] - prediction for last timestep
        gt: [B, H, W, C] or [B, H, C] - ground truth for last timestep

    Returns:
        dict with mse, nrmse, and per-sample values
    """
    # Flatten spatial and channel dims: [B, -1]
    B = pred.shape[0]
    pred_flat = pred.reshape(B, -1)
    gt_flat = gt.reshape(B, -1)

    # Per-sample MSE: mean over spatial*channel
    mse_per_sample = ((pred_flat - gt_flat) ** 2).mean(dim=1)  # [B]

    # Per-sample RMSE
    rmse_per_sample = torch.sqrt(mse_per_sample)  # [B]

    # Per-sample nRMSE: normalize by GT std
    gt_std = gt_flat.std(dim=1)  # [B]
    nrmse_per_sample = rmse_per_sample / (gt_std + 1e-8)  # [B]

    return {
        'mse_per_sample': mse_per_sample,
        'nrmse_per_sample': nrmse_per_sample,
        'mse': mse_per_sample.mean().item(),
        'nrmse': nrmse_per_sample.mean().item(),
    }


class TestDataset(torch.utils.data.Dataset):
    """
    Wrapper dataset for testing with random timestep sampling.
    """
    def __init__(self, base_dataset: PDEDataset, input_steps: int, seed: int = 42):
        self.base_dataset = base_dataset
        self.input_steps = input_steps
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Get full sequence from base dataset
        data, channel_mask = self.base_dataset[idx]

        # data shape: [T, *spatial, C] where T=17
        T = data.shape[0]

        # Random start point: need input_steps + 1 frames
        max_start = T - self.input_steps - 1
        t_start = self.rng.randint(0, max_start)

        # Extract input and ground truth
        input_data = data[t_start : t_start + self.input_steps]  # [input_steps, *spatial, C]
        gt_data = data[t_start + self.input_steps]  # [*spatial, C]

        return input_data, gt_data, channel_mask


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
    print(f"Using device: {device}")

    # Load model
    checkpoint_path = test_config['checkpoint']
    print(f"Loading checkpoint: {checkpoint_path}")
    model = load_model(checkpoint_path, model_config, device)
    print("Model loaded successfully")

    # Create dataset
    base_dataset = PDEDataset(
        data_dir=test_config['dataset']['path'],
        temporal_length=17,  # Full sequence
        split='val',
        train_ratio=0.9,
        seed=seed,
    )

    input_steps = test_config['test']['input_steps']
    test_dataset = TestDataset(base_dataset, input_steps, seed)

    # Limit samples if specified
    num_samples = test_config['test'].get('num_samples')
    if num_samples is not None:
        indices = list(range(min(num_samples, len(test_dataset))))
        test_dataset = torch.utils.data.Subset(test_dataset, indices)

    # DataLoader
    batch_size = test_config['test'].get('batch_size', 1)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Test samples: {len(test_dataset)}")
    print(f"Input steps: {input_steps}")
    print()

    # Evaluation
    all_mse = []
    all_nrmse = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_data, gt_data, channel_mask = batch

            # Move to device
            input_data = input_data.to(device)  # [B, input_steps, *spatial, C]
            gt_data = gt_data.to(device)  # [B, *spatial, C]
            channel_mask = channel_mask.to(device)

            # Forward pass
            output = model(input_data)  # [B, input_steps, *spatial, C]

            # Get last timestep prediction
            pred = output[:, -1]  # [B, *spatial, C]

            # Apply channel mask (only compute on valid channels)
            # channel_mask: [B, C] -> expand to match spatial dims
            if pred.ndim == 4:  # 2D: [B, H, W, C]
                mask = channel_mask[:, None, None, :]  # [B, 1, 1, C]
            else:  # 1D: [B, H, C]
                mask = channel_mask[:, None, :]  # [B, 1, C]

            pred_masked = pred * mask
            gt_masked = gt_data * mask

            # Compute metrics
            metrics = compute_metrics(pred_masked, gt_masked)
            all_mse.append(metrics['mse_per_sample'])
            all_nrmse.append(metrics['nrmse_per_sample'])

    # Aggregate results
    all_mse = torch.cat(all_mse)
    all_nrmse = torch.cat(all_nrmse)

    mse_mean = all_mse.mean().item()
    mse_std = all_mse.std().item()
    nrmse_mean = all_nrmse.mean().item()
    nrmse_std = all_nrmse.std().item()

    # Print results
    print()
    print("=" * 42)
    print("           Test Results")
    print("=" * 42)
    print(f"Checkpoint:   {checkpoint_path}")
    print(f"Samples:      {len(test_dataset)} | Input steps: {input_steps}")
    print()
    print(f"MSE:          {mse_mean:.6f} ± {mse_std:.6f}")
    print(f"nRMSE:        {nrmse_mean:.6f} ± {nrmse_std:.6f}")
    print("=" * 42)


if __name__ == "__main__":
    main()
