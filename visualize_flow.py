"""
Flow Mixing Visualization Tool.

Generates:
A. Prediction vs Ground Truth comparison
B. PDE Residual heatmap

Usage:
    # UNet → visualization_unet.png
  python visualize_flow.py --checkpoint checkpoints_unet_baseline_flow_v2/best.pt --config configs/unet_baseline_flow_v2.yaml --output_dir ./vis_results 

  # MLP → visualization_mlp.png
  python visualize_flow.py --checkpoint checkpoints_mlp/best.pt --config configs/mlp_baseline_flow_v2.yaml --output_dir ./vis_results   

  # LoRA → visualization_lora.png
  python visualize_flow.py --checkpoint checkpoints_flow_lora_v2/best_lora.pt --config configs/finetune_flow_v2.yaml --lora --output_dir ./vis_results
"""

import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple

from dataset_flow import FlowMixingDataset, flow_mixing_collate_fn
from pde_loss_flow import flow_mixing_pde_loss


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model(config: dict, checkpoint_path: str, is_lora: bool = False, device: str = 'cuda'):
    """Load model from checkpoint."""
    if is_lora:
        from model_lora import PDELoRAModel
        model = PDELoRAModel(config, pretrained_path=None)

        # Load LoRA checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'lora_state_dict' in checkpoint:
            if 'base_state_dict' in checkpoint:
                model.model.load_state_dict(checkpoint['base_state_dict'], strict=False)
            model.model.transformer.load_state_dict(checkpoint['lora_state_dict'], strict=False)
        else:
            model.model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=False)

        model = model.to(device)
        model.eval()
        return model, True
    else:
        # Try to determine model type
        model_type = config.get('model', {}).get('type', 'unet')

        if model_type == 'mlp':
            from model_mlp_baseline import create_mlp_baseline
            model = create_mlp_baseline(config)
        elif model_type == 'unet':
            from model_unet_baseline import create_unet_baseline
            model = create_unet_baseline(config)
        else:
            # Try FNO as fallback
            try:
                from model_fno_baseline import create_fno_baseline
                model = create_fno_baseline(config)
            except ImportError:
                from model_unet_baseline import create_unet_baseline
                model = create_unet_baseline(config)

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        model.eval()
        return model, False


def get_sample(config: dict, sample_idx: int = 0, clip_idx: int = 0) -> dict:
    """Get a single sample from the dataset."""
    dataset = FlowMixingDataset(
        data_path=config['dataset']['path'],
        temporal_length=config['dataset']['temporal_length'],
        split='val',
        train_ratio=config['dataset']['train_ratio'],
        seed=config['dataset']['seed'],
        clips_per_sample=None,
    )

    # Get sample index
    # dataset.samples is a list of (sample_key, vtmax, total_timesteps)
    n_samples = len(dataset.samples)
    clips_per_sample = len(dataset) // n_samples if n_samples > 0 else 1

    idx = sample_idx * clips_per_sample + clip_idx
    idx = min(idx, len(dataset) - 1)
    sample = dataset[idx]

    # Collate
    batch = flow_mixing_collate_fn([sample])
    return batch, dataset


@torch.no_grad()
def run_inference(model, batch: dict, device: str = 'cuda', is_lora: bool = False, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Run model inference."""
    if is_lora:
        # LoRA uses all 6 channels with bf16
        data = batch['data'].to(device=device, dtype=torch.bfloat16)
        input_data = data[:, :-1]  # [B, 16, H, W, 6]
        output = model(input_data)  # [B, 16, H, W, 6]
    else:
        # Baseline (UNet/MLP) uses only first channel
        data = batch['data'][..., :1].to(device=device, dtype=dtype)  # [B, T, H, W, 1]
        output = model(data)  # [B, 16, H, W, 1]

    return output.float()  # Convert back to float32 for visualization


def compute_pde_residual(
    pred: torch.Tensor,
    batch: dict,
    config: dict,
    device: str = 'cuda',
) -> Tuple[torch.Tensor, float]:
    """
    Compute PDE residual field using pde_loss_flow (2nd order upwind).

    Returns:
        residual: [B, T-1, H, W] PDE residual field
        pde_loss: float - MSE of residual (same as training)
    """
    pred_u = pred[..., :1].to(device).float()  # [B, T, H, W, 1]

    dt = config.get('physics', {}).get('dt', 1/999)
    Lx = config.get('physics', {}).get('Lx', 1.0)
    Ly = config.get('physics', {}).get('Ly', 1.0)

    boundary_left = batch['boundary_left'].to(device).float()
    boundary_right = batch['boundary_right'].to(device).float()
    boundary_bottom = batch['boundary_bottom'].to(device).float()
    boundary_top = batch['boundary_top'].to(device).float()
    vtmax = batch['vtmax'].to(device).float().mean().item()

    # Use same function as training (2nd order upwind)
    pde_loss, loss_time, loss_advection, residual = flow_mixing_pde_loss(
        pred=pred_u,
        boundary_left=boundary_left,
        boundary_right=boundary_right,
        boundary_bottom=boundary_bottom,
        boundary_top=boundary_top,
        vtmax=vtmax,
        dt=dt,
        Lx=Lx,
        Ly=Ly,
    )

    return residual, pde_loss.item()


def plot_single_timestep(
    gt: torch.Tensor,
    pred: torch.Tensor,
    residual: torch.Tensor,
    timestep: int = -1,
    channel: int = 0,
    vtmax: float = 0.0,
    save_path: Optional[str] = None,
):
    """
    Plot GT, Prediction, PDE Residual in one row (3 columns).

    Args:
        gt: [B, T, H, W, C] ground truth
        pred: [B, T, H, W, C] prediction
        residual: [B, T-1, H, W] PDE residual
        timestep: which timestep to plot (-1 = last)
        channel: which channel to plot
        vtmax: vtmax value for title
        save_path: path to save figure
    """
    # Get the specified timestep
    gt_np = gt[0, timestep, :, :, channel].cpu().numpy()
    pred_np = pred[0, timestep, :, :, channel].cpu().numpy()

    # Residual has T-1 timesteps, adjust index
    res_idx = timestep if timestep >= 0 else residual.shape[1] + timestep
    res_np = residual[0, res_idx].cpu().numpy()

    # Coordinate range [0, 1]
    extent = [0, 1, 0, 1]

    # Create figure: 1 row, 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Determine color range for GT and Pred
    vmin = min(gt_np.min(), pred_np.min())
    vmax = max(gt_np.max(), pred_np.max())

    # 1. Ground Truth
    im0 = axes[0].imshow(gt_np, origin='lower', extent=extent, cmap='jet', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'Ground Truth (t={timestep})', fontsize=12)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. Prediction
    im1 = axes[1].imshow(pred_np, origin='lower', extent=extent, cmap='jet', vmin=vmin, vmax=vmax)
    pred_rmse = np.sqrt(np.mean((pred_np - gt_np)**2))
    axes[1].set_title(f'Prediction (RMSE={pred_rmse:.4f})', fontsize=12)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. PDE Residual (use 95th percentile for colorbar)
    res_percentile = np.percentile(np.abs(res_np), 95)
    res_max = res_percentile if res_percentile > 0 else 1.0
    im2 = axes[2].imshow(res_np, origin='lower', extent=extent, cmap='RdBu_r', vmin=-res_max, vmax=res_max)
    pde_mse = np.mean(res_np**2)
    axes[2].set_title(f'PDE Residual (MSE={pde_mse:.2f})', fontsize=12)
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(f'vtmax={vtmax:.2f}', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def plot_multiple_samples(
    results: list,
    save_path: Optional[str] = None,
):
    """
    Plot multiple samples in rows (each row: GT | Prediction | PDE Residual).

    Args:
        results: list of dicts with keys:
            - gt: [H, W] ground truth
            - pred: [H, W] prediction
            - residual: [H, W] PDE residual
            - vtmax: float
            - sample_idx: int
        save_path: path to save figure
    """
    n_samples = len(results)
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 4 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    extent = [0, 1, 0, 1]

    for row, res in enumerate(results):
        gt_np = res['gt']
        pred_np = res['pred']
        res_np = res['residual']
        vtmax = res['vtmax']
        sample_idx = res['sample_idx']

        # Color range for GT and Pred
        vmin = min(gt_np.min(), pred_np.min())
        vmax = max(gt_np.max(), pred_np.max())

        # 1. Ground Truth
        im0 = axes[row, 0].imshow(gt_np, origin='lower', extent=extent, cmap='jet', vmin=vmin, vmax=vmax)
        axes[row, 0].set_title(f'Ground Truth', fontsize=11)
        axes[row, 0].set_ylabel(f'Sample {sample_idx}\nvtmax={vtmax:.2f}', fontsize=10)
        axes[row, 0].set_xlabel('x')
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)

        # 2. Prediction
        im1 = axes[row, 1].imshow(pred_np, origin='lower', extent=extent, cmap='jet', vmin=vmin, vmax=vmax)
        pred_rmse = np.sqrt(np.mean((pred_np - gt_np)**2))
        axes[row, 1].set_title(f'Prediction (RMSE={pred_rmse:.4f})', fontsize=11)
        axes[row, 1].set_xlabel('x')
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)

        # 3. PDE Residual (use 95th percentile for colorbar to avoid outliers)
        res_percentile = np.percentile(np.abs(res_np), 95)
        res_max = res_percentile if res_percentile > 0 else 1.0
        im2 = axes[row, 2].imshow(res_np, origin='lower', extent=extent, cmap='RdBu_r', vmin=-res_max, vmax=res_max)
        pde_mse = np.mean(res_np**2)
        axes[row, 2].set_title(f'PDE Residual (MSE={pde_mse:.2f})', fontsize=11)
        axes[row, 2].set_xlabel('x')
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Flow Mixing Visualization")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='./vis_results', help='Output directory')
    parser.add_argument('--lora', action='store_true', help='Use LoRA checkpoint')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples (overrides config)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    config = load_config(args.config)

    # Get num_samples from args or config
    num_samples = args.num_samples
    if num_samples is None:
        num_samples = config.get('visualization', {}).get('num_samples', 3)

    print(f"Loading model from: {args.checkpoint}")
    model, is_lora = load_model(config, args.checkpoint, is_lora=args.lora, device=device)

    # Determine model type for output filename
    if is_lora:
        model_suffix = "lora"
    else:
        model_suffix = config.get('model', {}).get('type', 'unet')

    # Create dataset
    dataset = FlowMixingDataset(
        data_path=config['dataset']['path'],
        temporal_length=config['dataset']['temporal_length'],
        split='val',
        train_ratio=config['dataset']['train_ratio'],
        seed=config['dataset']['seed'],
        clips_per_sample=None,
    )

    # Random sample selection
    np.random.seed(args.seed)
    total_clips = len(dataset)
    n_val_samples = len(dataset.samples)
    clips_per_sample = total_clips // n_val_samples if n_val_samples > 0 else 1

    # Select random samples and random start times
    sample_indices = np.random.choice(n_val_samples, min(num_samples, n_val_samples), replace=False)

    print(f"Visualizing {len(sample_indices)} samples...")

    results = []
    all_rmse = []
    all_pde_mse = []

    # Determine model dtype
    model_dtype = torch.bfloat16 if is_lora else torch.float32

    for i, sample_idx in enumerate(sample_indices):
        # Random clip within this sample
        clip_idx = np.random.randint(0, clips_per_sample)
        idx = sample_idx * clips_per_sample + clip_idx
        idx = min(idx, total_clips - 1)

        sample = dataset[idx]
        batch = flow_mixing_collate_fn([sample])

        # Run inference
        pred = run_inference(model, batch, device=device, is_lora=is_lora, dtype=model_dtype)

        vtmax = batch['vtmax'][0].item()

        if is_lora:
            # LoRA: data is [B, T, H, W, 6], pred is [B, 16, H, W, 6]
            data = batch['data'].to(device)
            gt = data[:, 1:]  # [B, 16, H, W, 6]
            t0 = data[:, 0:1]
            pred_with_t0 = torch.cat([t0.float(), pred], dim=1)
            gt_last = gt[0, -1, :, :, 0].float().cpu().numpy()
            pred_last = pred[0, -1, :, :, 0].cpu().numpy()
        else:
            # Baseline: data is [B, T, H, W, 1], pred is [B, 16, H, W, 1]
            data = batch['data'][..., :1].to(device)  # Only first channel
            gt = data[:, 1:]  # [B, 16, H, W, 1]
            t0 = data[:, 0:1]
            pred_with_t0 = torch.cat([t0.float(), pred], dim=1)
            gt_last = gt[0, -1, :, :, 0].float().cpu().numpy()
            pred_last = pred[0, -1, :, :, 0].cpu().numpy()

        # Compute PDE residual (using same method as training: 2nd order upwind)
        residual, pde_mse = compute_pde_residual(
            pred=pred_with_t0,
            batch=batch,
            config=config,
            device=device,
        )
        res_last = residual[0, -1].cpu().numpy()

        results.append({
            'gt': gt_last,
            'pred': pred_last,
            'residual': res_last,
            'vtmax': vtmax,
            'sample_idx': sample_idx,
        })

        # Metrics
        rmse = np.sqrt(np.mean((pred_last - gt_last)**2))
        all_rmse.append(rmse)
        all_pde_mse.append(pde_mse)

        print(f"  Sample {sample_idx}: vtmax={vtmax:.2f}, RMSE={rmse:.4f}, PDE_MSE={pde_mse:.2f}")

    # Plot all samples
    output_filename = f"visualization_{model_suffix}.png"
    print("Plotting visualization...")
    plot_multiple_samples(
        results=results,
        save_path=str(output_dir / output_filename),
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Visualization Complete")
    print("=" * 60)
    print(f"Output: {output_dir / output_filename}")
    print(f"\nAverage Metrics ({len(results)} samples):")
    print(f"  - RMSE (pred vs gt): {np.mean(all_rmse):.6f}")
    print(f"  - PDE Loss (MSE): {np.mean(all_pde_mse):.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
