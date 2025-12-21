"""
For PDE-Transformer
Test script for PDE Causal AR model evaluation using metrics.py

VERSION: 2.0 - Global normalization + Per-channel nRMSE

Usage:
    python test_with_metrics.py --config configs/test_v2.yaml
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

    # Get spatial domain size (assuming square domain)
    sample_data = base_dataset[0]['data']  # [T, H, W, C]
    H, W = sample_data.shape[1:3]
    Lx = test_config['test'].get('Lx', 1.0)
    Ly = test_config['test'].get('Ly', 1.0)
    
    # Plot config
    do_plot = test_config['test'].get('plot', False)
    channel_plot = test_config['test'].get('channel_plot', 0)
    model_name = test_config['test'].get('model_name', 'pde_transformer')
    
    print(f"Samples: {num_samples} | Input steps: {input_steps} | Batch size: {batch_size}")
    print(f"Spatial resolution: {H}x{W} | Domain size: Lx={Lx}, Ly={Ly}")
    if do_plot:
        print(f"Plotting enabled: channel {channel_plot}, output prefix '{model_name}'")
    
    # Collect all predictions and ground truths
    all_preds = []
    all_gts = []
    
    valid_mask = None
    plot_saved = False
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

            # Filter valid channels
            valid_channel_mask = channel_mask_device.bool()
            pred_valid = pred[..., valid_channel_mask]  # [B, H, W, C_valid]
            gt_valid = gt_batch[..., valid_channel_mask]  # [B, H, W, C_valid]

            # Save visualization for first batch if requested
            if do_plot and not plot_saved:
                full_names = ['vx', 'vy', 'vz', 'p', 'rho', 'T']
                channel_names = [n for n, v in zip(full_names, valid_mask.tolist()) if v]
                
                if channel_plot < len(channel_names):
                    channel_name = channel_names[channel_plot]
                    print(f"\nSaving visualization for channel '{channel_name}'...")
                    save_visualization(
                        pred_valid[0, :, :, channel_plot],
                        gt_valid[0, :, :, channel_plot],
                        channel_plot,
                        model_name,
                        H, W,
                        channel_name
                    )
                    plot_saved = True
                else:
                    print(f"\nWarning: channel_plot={channel_plot} is invalid (max={len(channel_names)-1}), skipping plot")
                    plot_saved = True

            # Collect predictions and ground truths
            all_preds.append(pred_valid.cpu())  # Move to CPU to save GPU memory
            all_gts.append(gt_valid.cpu())

    # Concatenate all batches into one large batch
    print("\nComputing metrics on all collected data...")
    all_preds = torch.cat(all_preds, dim=0)  # [N_total, H, W, C_valid]
    all_gts = torch.cat(all_gts, dim=0)  # [N_total, H, W, C_valid]
    
    print(f"Total samples: {all_preds.shape[0]}")
    
    # Add time dimension
    all_preds = all_preds.unsqueeze(-2)  # [N_total, H, W, 1, C_valid]
    all_gts = all_gts.unsqueeze(-2)  # [N_total, H, W, 1, C_valid]
    
    # Move to device for computation
    all_preds = all_preds.to(device=device)
    all_gts = all_gts.to(device=device)
    
    # Compute overall metrics on entire dataset at once
    err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F = metric_func(
        all_preds.float(),
        all_gts.float(),
        if_mean=True,
        Lx=Lx, Ly=Ly, Lz=1.0,
        iLow=4, iHigh=12,
        initial_step=0
    )
    
    # Compute per-channel nRMSE
    C_valid = all_preds.shape[-1]
    per_channel_nRMSE = []
    
    for ch_idx in range(C_valid):
        pred_ch = all_preds[..., ch_idx:ch_idx+1]
        gt_ch = all_gts[..., ch_idx:ch_idx+1]
        
        _, ch_nRMSE, _, _, _, _ = metric_func(
            pred_ch.float(),
            gt_ch.float(),
            if_mean=True,
            Lx=Lx, Ly=Ly, Lz=1.0,
            iLow=4, iHigh=12,
            initial_step=0
        )
        per_channel_nRMSE.append(ch_nRMSE.cpu())
    
    per_channel_nRMSE = torch.stack(per_channel_nRMSE)

    # Channel names
    full_names = ['vx', 'vy', 'vz', 'p', 'rho', 'T']
    channel_names = [n for n, v in zip(full_names, valid_mask.tolist()) if v]

    # Print results
    print("\n" + "=" * 70)
    print("                         Test Results")
    print("                   [Script Version 2.0]")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Samples: {num_samples} | Input steps: {input_steps} | Batch size: {batch_size}")
    print(f"Total predictions: {all_preds.shape[0]}")
    print(f"Spatial resolution: {H}x{W} | Domain: Lx={Lx}, Ly={Ly}")
    print()
    
    # Overall metrics (averaged over all valid channels)
    print("Overall Metrics (averaged over all valid channels):")
    print("-" * 70)
    print(f"{'Metric':<30} {'Value':<20}")
    print("-" * 70)
    print(f"{'RMSE':<30} {err_RMSE.item():<20.1e}")
    print(f"{'Normalized RMSE (nRMSE)':<30} {err_nRMSE.item():<20.1e}")
    print(f"{'Maximum Error':<30} {err_Max.item():<20.1e}")
    print(f"{'Conserved Variables RMSE':<30} {err_CSV.item():<20.1e}")
    print(f"{'Boundary RMSE':<30} {err_BD.item():<20.1e}")
    print("-" * 70)
    print()
    
    # Fourier space errors
    print("Fourier Space RMSE (Low/Mid/High frequency):")
    print("-" * 70)
    freq_labels = ['Low Frequency', 'Mid Frequency', 'High Frequency']
    for i, label in enumerate(freq_labels):
        print(f"{label:<30} {err_F[i].item():<20.1e}")
    print("-" * 70)
    print()
    
    # Per-channel nRMSE breakdown
    print("Per-channel nRMSE:")
    print("-" * 70)
    print(f"{'Channel':<15} {'nRMSE':<20}")
    print("-" * 70)
    for i, name in enumerate(channel_names):
        print(f"{name:<15} {per_channel_nRMSE[i].item():<20.1e}")
    print("-" * 70)
    print(f"{'Average':<15} {per_channel_nRMSE.mean().item():<20.1e}")
    print("-" * 70)
    print()
    
    print("Note: All metrics computed on entire dataset with global normalization.")
    print("=" * 70)


if __name__ == "__main__":
    main()