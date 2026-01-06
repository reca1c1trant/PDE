"""
For PDE-Transformer
Test script for PDE Causal AR model evaluation using metrics.py

VERSION: 3.0 - Multi-dataset evaluation via YAML config

Usage:
    python test.py --config configs/test_multi.yaml
"""

import os
import argparse
import random
import numpy as np
import torch
import yaml
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pipeline import PDECausalModel
from dataset import PDEDataset
from metrics import metric_func


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


def save_visualization(pred, target, channel_idx, model_name, H, W, channel_name="channel"):
    """Save prediction and ground truth visualization for 2D spatial data."""
    plt.ioff()
    
    # Prediction plot
    fig, ax = plt.subplots(figsize=(6.5, 6))
    h = ax.imshow(
        pred.squeeze().t().detach().cpu(),
        extent=[0, W, 0, H],
        origin="lower",
        aspect="auto",
    )
    h.set_clim(target.min(), target.max())
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=20)
    ax.set_title(f"Prediction - {channel_name}", fontsize=20)
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.set_ylabel("$y$", fontsize=20)
    ax.set_xlabel("$x$", fontsize=20)
    plt.tight_layout()
    filename = f"{model_name}_{channel_name}_pred.pdf"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filename}")

    # Ground truth plot
    fig, ax = plt.subplots(figsize=(6.5, 6))
    h = ax.imshow(
        target.squeeze().t().detach().cpu(),
        extent=[0, W, 0, H],
        origin="lower",
        aspect="auto",
    )
    h.set_clim(target.min(), target.max())
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=20)
    ax.set_title(f"Ground Truth - {channel_name}", fontsize=20)
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.set_ylabel("$y$", fontsize=20)
    ax.set_xlabel("$x$", fontsize=20)
    plt.tight_layout()
    filename = f"{model_name}_{channel_name}_gt.pdf"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filename}")



def evaluate_dataset(
    model,
    model_dtype,
    device,
    dataset_config: dict,
    test_config: dict,
    dataset_name: str,
) -> dict:
    """
    Evaluate model on a single dataset.

    Returns:
        dict with metrics results
    """
    seed = test_config.get('seed', 42)
    rng = random.Random(seed)

    # Create dataset
    base_dataset = PDEDataset(
        data_dir=dataset_config['path'],
        temporal_length=16,
        split='val',
        train_ratio=dataset_config.get('train_ratio', 0.9),
        seed=dataset_config.get('seed', 42),
        clips_per_sample=None,
    )

    # Test parameters
    input_steps = test_config.get('input_steps', 16)
    batch_size = test_config.get('batch_size', 8)
    num_samples = test_config.get('num_samples', len(base_dataset))
    num_samples = min(num_samples, len(base_dataset))

    # Domain size
    Lx = dataset_config.get('Lx', 1.0)
    Ly = dataset_config.get('Ly', 1.0)

    # Get spatial resolution
    sample_data = base_dataset[0]['data']
    H, W = sample_data.shape[1:3]

    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*70}")
    print(f"Path: {dataset_config['path']}")
    print(f"Samples: {num_samples} | Input steps: {input_steps} | Batch size: {batch_size}")
    print(f"Resolution: {H}x{W} | Domain: Lx={Lx}, Ly={Ly}")

    # Collect predictions
    all_preds = []
    all_gts = []
    valid_mask = None

    print("Evaluating...")
    with torch.no_grad():
        for sample_idx in tqdm(range(num_samples), desc=dataset_name):
            sample = base_dataset[sample_idx]
            data = sample['data']
            channel_mask = sample['channel_mask']

            T = data.shape[0]
            max_start = T - input_steps - 1

            if max_start < 0:
                continue

            starts = [rng.randint(0, max_start) for _ in range(batch_size)]

            inputs = []
            gts = []
            for start in starts:
                inp = data[start : start + input_steps]
                gt = data[start + input_steps]
                inputs.append(inp)
                gts.append(gt)

            input_batch = torch.stack(inputs, dim=0).to(device=device, dtype=model_dtype)
            gt_batch = torch.stack(gts, dim=0).to(device=device, dtype=model_dtype)
            channel_mask_device = channel_mask.to(device=device)

            output = model(input_batch)
            pred = output[:, -1]

            if valid_mask is None:
                valid_mask = channel_mask.bool().cpu()

            valid_channel_mask = channel_mask_device.bool()
            pred_valid = pred[..., valid_channel_mask]
            gt_valid = gt_batch[..., valid_channel_mask]

            all_preds.append(pred_valid.cpu())
            all_gts.append(gt_valid.cpu())

    # Compute metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_gts = torch.cat(all_gts, dim=0)

    all_preds = all_preds.unsqueeze(-2).to(device=device)
    all_gts = all_gts.unsqueeze(-2).to(device=device)

    err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F = metric_func(
        all_preds.float(), all_gts.float(),
        if_mean=True, Lx=Lx, Ly=Ly, Lz=1.0,
        iLow=4, iHigh=12, initial_step=0
    )

    # Per-channel nRMSE
    C_valid = all_preds.shape[-1]
    per_channel_nRMSE = []
    for ch_idx in range(C_valid):
        _, ch_nRMSE, _, _, _, _ = metric_func(
            all_preds[..., ch_idx:ch_idx+1].float(),
            all_gts[..., ch_idx:ch_idx+1].float(),
            if_mean=True, Lx=Lx, Ly=Ly, Lz=1.0,
            iLow=4, iHigh=12, initial_step=0
        )
        per_channel_nRMSE.append(ch_nRMSE.cpu().item())

    # Channel names
    full_names = ['vx', 'vy', 'vz', 'p', 'rho', 'T']
    channel_names = [n for n, v in zip(full_names, valid_mask.tolist()) if v]

    # Print results
    print(f"\nResults for {dataset_name}:")
    print("-" * 50)
    print(f"{'nRMSE (overall)':<25} {err_nRMSE.item():.4e}")
    print(f"{'RMSE':<25} {err_RMSE.item():.4e}")
    print(f"{'Max Error':<25} {err_Max.item():.4e}")
    print("-" * 50)
    print("Per-channel nRMSE:")
    for i, name in enumerate(channel_names):
        print(f"  {name:<10} {per_channel_nRMSE[i]:.4e}")
    print("-" * 50)

    return {
        'name': dataset_name,
        'nRMSE': err_nRMSE.item(),
        'RMSE': err_RMSE.item(),
        'Max': err_Max.item(),
        'per_channel_nRMSE': dict(zip(channel_names, per_channel_nRMSE)),
        'num_samples': all_preds.shape[0],
    }


def main():
    args = parse_args()

    # Load configs
    test_config = load_config(args.config)
    model_config = load_config(test_config['model_config'])

    # Set seed
    seed = test_config.get('test', {}).get('seed', 42)
    set_seed(seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    checkpoint_path = test_config['checkpoint']
    print(f"Checkpoint: {checkpoint_path}")
    model, model_dtype = load_model(checkpoint_path, model_config, device)
    print("Model loaded")

    # Get test parameters
    test_params = test_config.get('test', {})

    # Get datasets to evaluate
    datasets = test_config.get('datasets', [])

    # Backward compatibility: if no 'datasets' key, use old single 'dataset' format
    if not datasets and 'dataset' in test_config:
        datasets = [{
            'name': 'default',
            'path': test_config['dataset']['path'],
            'train_ratio': test_config['dataset'].get('train_ratio', 0.9),
            'seed': test_config['dataset'].get('seed', 42),
            'Lx': test_params.get('Lx', 1.0),
            'Ly': test_params.get('Ly', 1.0),
        }]

    if not datasets:
        print("Error: No datasets specified in config!")
        return

    print(f"\nEvaluating {len(datasets)} dataset(s)...")

    # Evaluate each dataset
    all_results = []
    for ds_config in datasets:
        ds_name = ds_config.get('name', Path(ds_config['path']).stem)
        result = evaluate_dataset(
            model=model,
            model_dtype=model_dtype,
            device=device,
            dataset_config=ds_config,
            test_config=test_params,
            dataset_name=ds_name,
        )
        all_results.append(result)

    # Print summary
    print("\n" + "=" * 70)
    print("                      EVALUATION SUMMARY")
    print("                     [Script Version 3.0]")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Datasets evaluated: {len(all_results)}")
    print()
    print(f"{'Dataset':<25} {'nRMSE':<15} {'RMSE':<15} {'Samples':<10}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['name']:<25} {r['nRMSE']:<15.4e} {r['RMSE']:<15.4e} {r['num_samples']:<10}")
    print("-" * 70)

    # Average across datasets
    if len(all_results) > 1:
        avg_nRMSE = sum(r['nRMSE'] for r in all_results) / len(all_results)
        print(f"{'Average':<25} {avg_nRMSE:<15.4e}")
        print("-" * 70)

    print("=" * 70)


if __name__ == "__main__":
    main()