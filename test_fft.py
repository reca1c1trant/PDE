"""
Test script for FFT V3 PDE Causal AR model evaluation.

Usage:
    python test_fft.py --config configs/test_v3.yaml
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
    parser = argparse.ArgumentParser(description="FFT V3 PDE Testing")
    parser.add_argument('--config', type=str, default='configs/test_v3.yaml',
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

    # Handle DDP/FSDP wrapped state dict
    state_dict = checkpoint['model_state_dict']
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

    # Print model info
    enc_params = sum(p.numel() for p in model.encoder_2d.parameters())
    dec_params = sum(p.numel() for p in model.decoder_2d.parameters())
    trans_params = sum(p.numel() for p in model.transformer.parameters())
    print(f"Model params: Encoder={enc_params/1e6:.1f}M, Decoder={dec_params/1e6:.1f}M, Transformer={trans_params/1e6:.1f}M")

    return model, dtype


def compute_metrics(all_preds: torch.Tensor, all_gts: torch.Tensor, sigma: torch.Tensor = None, channel_names: list = None) -> dict:
    """
    Compute MSE, RMSE, and nRMSE.

    Args:
        all_preds: [N, *spatial, C]
        all_gts: [N, *spatial, C]
        sigma: [C] for nRMSE, or None for RMSE only
        channel_names: channel names for display
    """
    C = all_preds.shape[-1]
    if channel_names is None:
        channel_names = [f'ch{i}' for i in range(C)]

    N = all_preds.shape[0]
    pred_flat = all_preds.reshape(N, -1, C)
    gt_flat = all_gts.reshape(N, -1, C)

    # MSE per channel
    mse_per_channel = ((pred_flat - gt_flat) ** 2).mean(dim=(0, 1))
    rmse_per_channel = torch.sqrt(mse_per_channel + 1e-8)

    # nRMSE if sigma provided
    if sigma is not None:
        normalized_error = (pred_flat - gt_flat) / (sigma + 1e-8)
        nrmse_mse_per_channel = (normalized_error ** 2).mean(dim=(0, 1))
        nrmse_per_channel = torch.sqrt(nrmse_mse_per_channel + 1e-8)
        nrmse = nrmse_per_channel.mean().item()
    else:
        nrmse_per_channel = rmse_per_channel
        nrmse = rmse_per_channel.mean().item()

    mse = mse_per_channel.mean().item()
    rmse = rmse_per_channel.mean().item()

    return {
        'mse': mse,
        'rmse': rmse,
        'nrmse': nrmse,
        'sigma': sigma,
        'mse_per_channel': mse_per_channel,
        'rmse_per_channel': rmse_per_channel,
        'nrmse_per_channel': nrmse_per_channel,
        'channel_names': channel_names,
    }


def main():
    args = parse_args()

    # Load configs
    test_config = load_config(args.config)
    model_config = load_config(test_config['model_config'])

    # Set seed
    seed = test_config.get('seed', 42)
    set_seed(seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    checkpoint_path = test_config['checkpoint']
    print(f"Checkpoint: {checkpoint_path}")
    model, model_dtype = load_model(checkpoint_path, model_config, device)

    # Create dataset
    dataset_config = test_config['dataset']
    val_dataset = PDEDataset(
        data_dir=dataset_config['path'],
        temporal_length=17,
        split='val',
        train_ratio=dataset_config.get('train_ratio', 0.9),
        seed=seed,
    )

    # Limit samples if specified
    num_samples = test_config.get('num_samples')
    if num_samples is not None and num_samples < len(val_dataset):
        indices = list(range(num_samples))
        val_dataset = torch.utils.data.Subset(val_dataset, indices)

    batch_size = test_config.get('batch_size', 4)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda batch: {
            'data': torch.stack([b['data'] for b in batch]),
            'channel_mask': torch.stack([b['channel_mask'] for b in batch]),
        }
    )

    print(f"Val samples: {len(val_dataset)}")

    # Sigma for nRMSE
    nrmse_sigma = test_config.get('nrmse_sigma')
    if nrmse_sigma:
        sigma = torch.tensor(nrmse_sigma, dtype=torch.float32)
    else:
        sigma = None

    # Evaluation
    all_preds = []
    all_gts = []
    valid_mask = None

    print("\nEvaluating...")
    with torch.no_grad():
        for batch in tqdm(val_loader):
            data = batch['data'].to(device=device, dtype=model_dtype)
            channel_mask = batch['channel_mask'].to(device=device)

            # input: t[0:16], target: t[1:17]
            input_data = data[:, :-1]  # [B, 16, H, W, C]
            target_data = data[:, 1:]  # [B, 16, H, W, C]

            # Forward
            output = model(input_data)  # [B, 16, H, W, C]

            # Get valid channel mask
            if valid_mask is None:
                valid_mask = channel_mask[0].bool().cpu()

            # Collect all timesteps
            all_preds.append(output.cpu().float())
            all_gts.append(target_data.cpu().float())

    # Concatenate: [N, T, H, W, C]
    all_preds = torch.cat(all_preds, dim=0)
    all_gts = torch.cat(all_gts, dim=0)

    # Reshape: [N*T, H, W, C]
    N, T, H, W, C = all_preds.shape
    all_preds = all_preds.reshape(N * T, H, W, C)
    all_gts = all_gts.reshape(N * T, H, W, C)

    # Filter valid channels
    all_preds = all_preds[..., valid_mask]
    all_gts = all_gts[..., valid_mask]

    # Channel names
    full_names = ['vx', 'vy', 'vz', 'p', 'rho', 'T']
    channel_names = [n for n, v in zip(full_names, valid_mask.tolist()) if v]

    # Sigma for valid channels
    if sigma is not None:
        sigma_valid = sigma[valid_mask]
    else:
        sigma_valid = None

    # Compute metrics
    metrics = compute_metrics(all_preds, all_gts, sigma_valid, channel_names)

    # Print results
    print("\n" + "=" * 70)
    print("                        FFT V3 Test Results")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Samples: {N} | Timesteps: {T} | Total frames: {N*T}")
    print()

    print("Per-channel metrics:")
    print("-" * 70)
    if sigma_valid is not None:
        print(f"{'Channel':<10} {'sigma':<12} {'MSE':<14} {'RMSE':<14} {'nRMSE':<14}")
        print("-" * 70)
        for i, name in enumerate(metrics['channel_names']):
            s = sigma_valid[i].item()
            mse_c = metrics['mse_per_channel'][i].item()
            rmse_c = metrics['rmse_per_channel'][i].item()
            nrmse_c = metrics['nrmse_per_channel'][i].item()
            print(f"{name:<10} {s:<12.6f} {mse_c:<14.6e} {rmse_c:<14.6f} {nrmse_c:<14.6f}")
    else:
        print(f"{'Channel':<10} {'MSE':<14} {'RMSE':<14}")
        print("-" * 70)
        for i, name in enumerate(metrics['channel_names']):
            mse_c = metrics['mse_per_channel'][i].item()
            rmse_c = metrics['rmse_per_channel'][i].item()
            print(f"{name:<10} {mse_c:<14.6e} {rmse_c:<14.6f}")

    print("-" * 70)
    print()
    print(f"Overall MSE:   {metrics['mse']:.6e}")
    print(f"Overall RMSE:  {metrics['rmse']:.6f}")
    if sigma_valid is not None:
        print(f"Overall nRMSE: {metrics['nrmse']:.6f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
