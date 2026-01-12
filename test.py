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
    unload_model_after: bool = True,
) -> dict:
    """
    Evaluate model on a single dataset.

    Args:
        unload_model_after: If True, move model to CPU after inference to free GPU memory

    Returns:
        dict with metrics results
    """
    seed = test_config.get('seed', 42)
    rng = random.Random(seed)

    # Test parameters
    input_steps = test_config.get('input_steps', 16)

    # Create dataset (temporal_length = input_steps, PDEDataset will +1 internally)
    base_dataset = PDEDataset(
        data_dir=dataset_config['path'],
        temporal_length=input_steps,  # 根据 input_steps 动态设置
        split='val',
        train_ratio=dataset_config.get('train_ratio', 0.9),
        seed=dataset_config.get('seed', 42),
        clips_per_sample=None,
    )
    batch_size = test_config.get('batch_size', 8)
    num_samples = test_config.get('num_samples')
    if num_samples is None:
        num_samples = len(base_dataset)
    else:
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

    # Free GPU memory before computing metrics
    if unload_model_after:
        model.cpu()
        torch.cuda.empty_cache()
        print("Model moved to CPU, GPU memory freed.")

    # Compute metrics in batches to avoid OOM
    all_preds = torch.cat(all_preds, dim=0)  # [N, H, W, C_valid]
    all_gts = torch.cat(all_gts, dim=0)

    print(f"Computing metrics on {all_preds.shape[0]} predictions...")

    # Add time dimension for metric_func
    all_preds = all_preds.unsqueeze(-2)  # [N, H, W, 1, C_valid]
    all_gts = all_gts.unsqueeze(-2)

    # Compute metrics in batches on GPU
    metrics_batch_size = 1000  # Process 1000 samples at a time
    N = all_preds.shape[0]

    # Accumulate metrics
    total_RMSE = 0.0
    total_nRMSE = 0.0
    total_CSV = 0.0
    total_Max = 0.0
    total_BD = 0.0
    total_F = None  # [3] for low/mid/high freq
    num_batches = 0

    for i in range(0, N, metrics_batch_size):
        batch_preds = all_preds[i:i+metrics_batch_size].to(device=device)
        batch_gts = all_gts[i:i+metrics_batch_size].to(device=device)

        err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F = metric_func(
            batch_preds.float(), batch_gts.float(),
            if_mean=True, Lx=Lx, Ly=Ly, Lz=1.0,
            iLow=4, iHigh=12, initial_step=0
        )

        total_RMSE += err_RMSE.item()
        total_nRMSE += err_nRMSE.item()
        total_CSV += err_CSV.item()
        total_Max = max(total_Max, err_Max.item())
        total_BD += err_BD.item()
        if total_F is None:
            total_F = err_F.cpu().clone()
        else:
            total_F += err_F.cpu()
        num_batches += 1

        # Free batch memory
        del batch_preds, batch_gts
        torch.cuda.empty_cache()

    # Average over batches
    err_RMSE = total_RMSE / num_batches
    err_nRMSE = total_nRMSE / num_batches
    err_CSV = total_CSV / num_batches
    err_BD = total_BD / num_batches
    err_F = total_F / num_batches  # [3] tensor

    # Per-channel nRMSE (also in batches)
    C_valid = all_preds.shape[-1]
    per_channel_nRMSE = []
    for ch_idx in range(C_valid):
        ch_nRMSE_sum = 0.0
        ch_batches = 0
        for i in range(0, N, metrics_batch_size):
            batch_preds = all_preds[i:i+metrics_batch_size, ..., ch_idx:ch_idx+1].to(device=device)
            batch_gts = all_gts[i:i+metrics_batch_size, ..., ch_idx:ch_idx+1].to(device=device)

            _, ch_nRMSE, _, _, _, _ = metric_func(
                batch_preds.float(), batch_gts.float(),
                if_mean=True, Lx=Lx, Ly=Ly, Lz=1.0,
                iLow=4, iHigh=12, initial_step=0
            )
            ch_nRMSE_sum += ch_nRMSE.item()
            ch_batches += 1

            del batch_preds, batch_gts
            torch.cuda.empty_cache()

        per_channel_nRMSE.append(ch_nRMSE_sum / ch_batches)

    # Channel names
    full_names = ['vx', 'vy', 'vz', 'p', 'rho', 'T']
    channel_names = [n for n, v in zip(full_names, valid_mask.tolist()) if v]

    # Print results
    print(f"\nResults for {dataset_name}:")
    print("-" * 50)
    print(f"{'Metric':<30} {'Value':<15}")
    print("-" * 50)
    print(f"{'RMSE':<30} {err_RMSE:.4e}")
    print(f"{'nRMSE':<30} {err_nRMSE:.4e}")
    print(f"{'Max Error':<30} {total_Max:.4e}")
    print(f"{'Conserved Variables RMSE':<30} {err_CSV:.4e}")
    print(f"{'Boundary RMSE':<30} {err_BD:.4e}")
    print("-" * 50)
    print("Fourier Space RMSE:")
    freq_labels = ['  Low Frequency', '  Mid Frequency', '  High Frequency']
    for i, label in enumerate(freq_labels):
        print(f"{label:<30} {err_F[i].item():.4e}")
    print("-" * 50)
    print("Per-channel nRMSE:")
    for i, name in enumerate(channel_names):
        print(f"  {name:<10} {per_channel_nRMSE[i]:.4e}")
    print("-" * 50)

    return {
        'name': dataset_name,
        'nRMSE': err_nRMSE,
        'RMSE': err_RMSE,
        'Max': total_Max,
        'CSV': err_CSV,
        'BD': err_BD,
        'F_low': err_F[0].item(),
        'F_mid': err_F[1].item(),
        'F_high': err_F[2].item(),
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
    for idx, ds_config in enumerate(datasets):
        ds_name = ds_config.get('name', Path(ds_config['path']).stem)

        # Reload model to GPU if it was moved to CPU (for subsequent datasets)
        if next(model.parameters()).device != device:
            model.to(device)
            print(f"Model reloaded to {device}")

        # Only unload model after last dataset
        is_last = (idx == len(datasets) - 1)

        result = evaluate_dataset(
            model=model,
            model_dtype=model_dtype,
            device=device,
            dataset_config=ds_config,
            test_config=test_params,
            dataset_name=ds_name,
            unload_model_after=True,  # Always unload to free memory for metrics
        )
        all_results.append(result)

    # Print summary
    print("\n" + "=" * 90)
    print("                              EVALUATION SUMMARY")
    print("                             [Script Version 3.1]")
    print("=" * 90)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Datasets evaluated: {len(all_results)}")
    print()

    # Main metrics table
    print(f"{'Dataset':<20} {'nRMSE':<12} {'RMSE':<12} {'Max':<12} {'CSV':<12} {'BD':<12} {'Samples':<8}")
    print("-" * 90)
    for r in all_results:
        print(f"{r['name']:<20} {r['nRMSE']:<12.4e} {r['RMSE']:<12.4e} {r['Max']:<12.4e} {r['CSV']:<12.4e} {r['BD']:<12.4e} {r['num_samples']:<8}")
    print("-" * 90)

    # Average across datasets
    if len(all_results) > 1:
        avg_nRMSE = sum(r['nRMSE'] for r in all_results) / len(all_results)
        avg_RMSE = sum(r['RMSE'] for r in all_results) / len(all_results)
        avg_CSV = sum(r['CSV'] for r in all_results) / len(all_results)
        avg_BD = sum(r['BD'] for r in all_results) / len(all_results)
        print(f"{'Average':<20} {avg_nRMSE:<12.4e} {avg_RMSE:<12.4e} {'-':<12} {avg_CSV:<12.4e} {avg_BD:<12.4e}")
        print("-" * 90)

    # Fourier space metrics
    print()
    print("Fourier Space RMSE (Low / Mid / High frequency):")
    print("-" * 90)
    for r in all_results:
        print(f"  {r['name']:<18} {r['F_low']:<12.4e} {r['F_mid']:<12.4e} {r['F_high']:<12.4e}")
    print("-" * 90)

    print("=" * 90)


if __name__ == "__main__":
    main()