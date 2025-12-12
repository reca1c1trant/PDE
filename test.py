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


def load_model(checkpoint_path: str, model_config: dict, device: torch.device, dtype: torch.dtype = torch.bfloat16) -> PDECausalModel:
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
    model = model.to(device=device, dtype=dtype)
    model.eval()

    return model, dtype


def compute_final_metrics(all_preds: torch.Tensor, all_gts: torch.Tensor, sigma: torch.Tensor, channel_names: list = None) -> dict:
    """
    Compute MSE and nRMSE using the formula:
        L_c = sqrt(mean(((pred_c - gt_c) / sigma_c) ** 2))
        L_sim = mean(L_c for all channels)

    Args:
        all_preds: [N, *spatial, C] - all predictions concatenated
        all_gts: [N, *spatial, C] - all ground truths concatenated
        sigma: [C] - global std per channel (from training set)
        channel_names: Optional list of channel names for display

    Returns:
        dict with mse, nrmse, per-channel metrics
    """
    C = all_preds.shape[-1]
    if channel_names is None:
        channel_names = [f'ch{i}' for i in range(C)]

    # Flatten spatial dims: [N, -1, C] -> then work per channel
    N = all_preds.shape[0]

    # [N, H*W, C] or [N, H, C]
    pred_flat = all_preds.reshape(N, -1, C)  # [N, spatial, C]
    gt_flat = all_gts.reshape(N, -1, C)  # [N, spatial, C]

    # Per-channel nRMSE: L_c = sqrt(mean(((pred - gt) / sigma) ** 2))
    # Normalize error by sigma (from training set)
    normalized_error = (pred_flat - gt_flat) / (sigma + 1e-8)  # [N, spatial, C]

    # Mean over samples and spatial, then sqrt
    mse_per_channel = (normalized_error ** 2).mean(dim=(0, 1))  # [C]
    nrmse_per_channel = torch.sqrt(mse_per_channel)  # [C]

    # L_sim = mean over channels
    nrmse = nrmse_per_channel.mean().item()

    # Also compute regular MSE (not normalized)
    mse_per_channel_raw = ((pred_flat - gt_flat) ** 2).mean(dim=(0, 1))  # [C]
    mse = mse_per_channel_raw.mean().item()

    return {
        'mse': mse,
        'nrmse': nrmse,
        'sigma_per_channel': sigma,
        'nrmse_per_channel': nrmse_per_channel,
        'mse_per_channel': mse_per_channel_raw,
        'channel_names': channel_names,
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
        sample = self.base_dataset[idx]
        data = sample['data']
        channel_mask = sample['channel_mask']

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
    model, model_dtype = load_model(checkpoint_path, model_config, device)
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

    # Evaluation: collect all predictions and ground truths
    all_preds = []
    all_gts = []
    valid_channels_mask = None  # Track which channels are valid

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_data, gt_data, channel_mask = batch

            # Move to device and cast to model dtype
            input_data = input_data.to(device=device, dtype=model_dtype)  # [B, input_steps, *spatial, C]
            gt_data = gt_data.to(device=device, dtype=model_dtype)  # [B, *spatial, C]
            channel_mask = channel_mask.to(device=device, dtype=model_dtype)

            # Forward pass
            output = model(input_data)  # [B, input_steps, *spatial, C]

            # Get last timestep prediction
            pred = output[:, -1]  # [B, *spatial, C]

            # Get valid channels (assume same for all samples in dataset)
            if valid_channels_mask is None:
                valid_channels_mask = channel_mask[0].bool().cpu()  # [C]

            # Collect (move to CPU to save GPU memory)
            all_preds.append(pred.cpu())
            all_gts.append(gt_data.cpu())

    # Concatenate all samples
    all_preds = torch.cat(all_preds, dim=0)  # [N, *spatial, C]
    all_gts = torch.cat(all_gts, dim=0)  # [N, *spatial, C]

    # Filter to only valid channels
    all_preds = all_preds[..., valid_channels_mask]  # [N, *spatial, C_valid]
    all_gts = all_gts[..., valid_channels_mask]  # [N, *spatial, C_valid]

    # Get valid channel names
    full_channel_names = ['vx', 'vy', 'vz', 'p', 'rho', 'T']
    channel_names = [name for name, valid in zip(full_channel_names, valid_channels_mask.tolist()) if valid]

    # Get sigma from config (training set global sigma)
    nrmse_sigma = torch.tensor(test_config['test']['nrmse_sigma'], dtype=torch.float32)
    sigma_valid = nrmse_sigma[valid_channels_mask]  # Filter to valid channels

    # Compute metrics with global sigma
    metrics = compute_final_metrics(all_preds, all_gts, sigma_valid, channel_names)

    # Print results
    print()
    print("=" * 50)
    print("                 Test Results")
    print("=" * 50)
    print(f"Checkpoint:   {checkpoint_path}")
    print(f"Samples:      {len(test_dataset)} | Input steps: {input_steps}")
    print()
    print("Per-channel metrics:")
    print("-" * 50)
    print(f"{'Channel':<10} {'sigma':<12} {'MSE':<12} {'nRMSE':<12}")
    print("-" * 50)
    for i, name in enumerate(metrics['channel_names']):
        sigma = metrics['sigma_per_channel'][i].item()
        mse_c = metrics['mse_per_channel'][i].item()
        nrmse_c = metrics['nrmse_per_channel'][i].item()
        print(f"{name:<10} {sigma:<12.6f} {mse_c:<12.6f} {nrmse_c:<12.6f}")
    print("-" * 50)
    print()
    print(f"Overall MSE:   {metrics['mse']:.6f}")
    print(f"Overall nRMSE: {metrics['nrmse']:.6f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
