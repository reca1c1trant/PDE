"""
Flow Mixing Visualization Tool.

Generates:
A. Prediction vs Ground Truth comparison
B. PDE Residual heatmap

Usage:
    # For baseline models (UNet/MLP)
    python visualize_flow.py --checkpoint path/to/best.pt --config path/to/config.yaml --output_dir ./vis_results

    # For LoRA checkpoints
    python visualize_flow.py --checkpoint path/to/best_lora.pt --config path/to/config.yaml --lora --output_dir ./vis_results
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
from pde_loss_flow_v2 import (
    flow_mixing_pde_loss_v2,
    compute_flow_coefficients,
    pad_with_boundaries_1x,
    central_derivative_x,
    central_derivative_y,
)


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

    # Get sample
    idx = sample_idx * dataset.clips_per_sample + clip_idx
    idx = min(idx, len(dataset) - 1)
    sample = dataset[idx]

    # Collate
    batch = flow_mixing_collate_fn([sample])
    return batch, dataset


@torch.no_grad()
def run_inference(model, batch: dict, device: str = 'cuda', is_lora: bool = False) -> torch.Tensor:
    """Run model inference."""
    data = batch['data'].to(device=device, dtype=torch.float32)

    if is_lora:
        input_data = data[:, :-1]  # [B, 16, H, W, 6]
        output = model(input_data)  # [B, 16, H, W, 6]
    else:
        output = model(data)  # [B, 16, H, W, 6]

    return output


def compute_pde_residual(
    pred: torch.Tensor,
    batch: dict,
    config: dict,
    pde_version: str = "v1",
    device: str = 'cuda',
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute PDE residual field.

    Returns:
        residual: [B, T-1, H, W] PDE residual
        du_dt: [B, T-1, H, W] time derivative
        advection: [B, T-1, H, W] advection term
    """
    pred_u = pred[..., :1].to(device).float()
    B, T, H, W, C = pred_u.shape

    dt = config.get('physics', {}).get('dt', 1/999)
    Lx = config.get('physics', {}).get('Lx', 1.0)
    Ly = config.get('physics', {}).get('Ly', 1.0)
    dx = Lx / W
    dy = Ly / H

    boundary_left = batch['boundary_left'].to(device).float()
    boundary_right = batch['boundary_right'].to(device).float()
    boundary_bottom = batch['boundary_bottom'].to(device).float()
    boundary_top = batch['boundary_top'].to(device).float()
    vtmax = batch['vtmax'].to(device).float().mean().item()

    u = pred_u[..., 0]  # [B, T, H, W]

    a, b = compute_flow_coefficients(H, W, vtmax, device, torch.float32)

    bnd_left = boundary_left[..., 0]
    bnd_right = boundary_right[..., 0]
    bnd_bottom = boundary_bottom[..., 0]
    bnd_top = boundary_top[..., 0]

    u_padded = pad_with_boundaries_1x(u, bnd_left, bnd_right, bnd_bottom, bnd_top)
    du_dx = central_derivative_x(u_padded, H, W, dx)
    du_dy = central_derivative_y(u_padded, H, W, dy)

    du_dt = (u[:, 1:] - u[:, :-1]) / dt
    advection = a * du_dx[:, 1:] + b * du_dy[:, 1:]
    residual = du_dt + advection

    return residual, du_dt, advection


def plot_pred_vs_gt(
    pred: torch.Tensor,
    gt: torch.Tensor,
    timesteps: list = [0, 4, 8, 12, 15],
    channel: int = 0,
    save_path: Optional[str] = None,
    title: str = "Prediction vs Ground Truth",
):
    """Plot prediction vs ground truth comparison."""
    pred_np = pred[0, :, :, :, channel].cpu().numpy()
    gt_np = gt[0, :, :, :, channel].cpu().numpy()

    n_times = len(timesteps)
    fig, axes = plt.subplots(3, n_times, figsize=(4 * n_times, 12))

    vmin = min(pred_np.min(), gt_np.min())
    vmax = max(pred_np.max(), gt_np.max())

    error = np.abs(pred_np - gt_np)
    emax = error.max()

    for i, t in enumerate(timesteps):
        im0 = axes[0, i].imshow(gt_np[t], cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')
        axes[0, i].set_title(f't={t} (GT)')
        axes[0, i].axis('off')

        im1 = axes[1, i].imshow(pred_np[t], cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')
        axes[1, i].set_title(f't={t} (Pred)')
        axes[1, i].axis('off')

        im2 = axes[2, i].imshow(error[t], cmap='hot', vmin=0, vmax=emax, origin='lower')
        axes[2, i].set_title(f't={t} (|Error|)')
        axes[2, i].axis('off')

    fig.colorbar(im0, ax=axes[0, :], shrink=0.6, label='Value')
    fig.colorbar(im1, ax=axes[1, :], shrink=0.6, label='Value')
    fig.colorbar(im2, ax=axes[2, :], shrink=0.6, label='|Error|')

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def plot_pde_residual(
    residual: torch.Tensor,
    du_dt: torch.Tensor,
    advection: torch.Tensor,
    timesteps: list = [0, 4, 8, 12, 14],
    save_path: Optional[str] = None,
    title: str = "PDE Residual Analysis",
):
    """Plot PDE residual heatmap."""
    res_np = residual[0].cpu().numpy()
    dt_np = du_dt[0].cpu().numpy()
    adv_np = advection[0].cpu().numpy()

    n_times = len(timesteps)
    fig, axes = plt.subplots(3, n_times, figsize=(4 * n_times, 12))

    res_max = np.abs(res_np).max()
    dt_max = np.abs(dt_np).max()
    adv_max = np.abs(adv_np).max()

    for i, t in enumerate(timesteps):
        im0 = axes[0, i].imshow(dt_np[t], cmap='RdBu_r', vmin=-dt_max, vmax=dt_max, origin='lower')
        axes[0, i].set_title(f't={t} (du/dt)')
        axes[0, i].axis('off')

        im1 = axes[1, i].imshow(adv_np[t], cmap='RdBu_r', vmin=-adv_max, vmax=adv_max, origin='lower')
        axes[1, i].set_title(f't={t} (Advection)')
        axes[1, i].axis('off')

        im2 = axes[2, i].imshow(res_np[t], cmap='RdBu_r', vmin=-res_max, vmax=res_max, origin='lower')
        axes[2, i].set_title(f't={t} (Residual)')
        axes[2, i].axis('off')

    fig.colorbar(im0, ax=axes[0, :], shrink=0.6, label='du/dt')
    fig.colorbar(im1, ax=axes[1, :], shrink=0.6, label='a*du/dx + b*du/dy')
    fig.colorbar(im2, ax=axes[2, :], shrink=0.6, label='PDE Residual')

    rmse = np.sqrt(np.mean(res_np**2))
    mae = np.mean(np.abs(res_np))
    fig.suptitle(f"{title}\nRMSE: {rmse:.4f}, MAE: {mae:.4f}", fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def plot_residual_distribution(
    residual: torch.Tensor,
    save_path: Optional[str] = None,
    title: str = "PDE Residual Distribution",
):
    """Plot histogram of PDE residual values."""
    res_np = residual.cpu().numpy().flatten()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(res_np, bins=100, density=True, alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Residual Value')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Distribution')
    axes[0].axvline(0, color='red', linestyle='--', alpha=0.5)

    mean = np.mean(res_np)
    std = np.std(res_np)
    axes[0].axvline(mean, color='green', linestyle='-', alpha=0.7, label=f'Mean: {mean:.4f}')
    axes[0].legend()

    axes[1].hist(np.abs(res_np), bins=100, density=True, alpha=0.7, color='coral')
    axes[1].set_xlabel('|Residual|')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Absolute Value Distribution')
    axes[1].set_yscale('log')

    fig.suptitle(f"{title}\nMean: {mean:.4f}, Std: {std:.4f}", fontsize=12)
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
    parser.add_argument('--sample_idx', type=int, default=0, help='Sample index')
    parser.add_argument('--clip_idx', type=int, default=0, help='Clip index')
    parser.add_argument('--pde_version', type=str, default='v1', choices=['v1', 'v2'], help='PDE version')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    config = load_config(args.config)
    print(f"Loading model from: {args.checkpoint}")
    model, is_lora = load_model(config, args.checkpoint, is_lora=args.lora, device=device)

    print(f"Loading sample {args.sample_idx}, clip {args.clip_idx}...")
    batch, dataset = get_sample(config, args.sample_idx, args.clip_idx)

    print("Running inference...")
    pred = run_inference(model, batch, device=device, is_lora=is_lora)

    data = batch['data'].to(device)
    gt = data[:, 1:]  # [B, 16, H, W, 6]
    t0 = data[:, 0:1]
    pred_with_t0 = torch.cat([t0, pred], dim=1)  # [B, 17, H, W, 6]

    # A: Prediction vs GT
    print("Plotting prediction vs ground truth...")
    plot_pred_vs_gt(
        pred=pred,
        gt=gt,
        timesteps=[0, 4, 8, 12, 15],
        channel=0,
        save_path=str(output_dir / 'pred_vs_gt.png'),
        title=f"Sample {args.sample_idx}, Clip {args.clip_idx} - Channel 0 (u)",
    )

    # B: PDE Residual
    print(f"Computing PDE residual (version: {args.pde_version})...")
    residual, du_dt, advection = compute_pde_residual(
        pred=pred_with_t0,
        batch=batch,
        config=config,
        pde_version=args.pde_version,
        device=device,
    )

    print("Plotting PDE residual heatmap...")
    plot_pde_residual(
        residual=residual,
        du_dt=du_dt,
        advection=advection,
        timesteps=[0, 4, 8, 12, 14],
        save_path=str(output_dir / 'pde_residual.png'),
        title=f"PDE Residual (version: {args.pde_version})",
    )

    print("Plotting residual distribution...")
    plot_residual_distribution(
        residual=residual,
        save_path=str(output_dir / 'residual_distribution.png'),
        title=f"PDE Residual Distribution (version: {args.pde_version})",
    )

    print("\n" + "=" * 60)
    print("Visualization Complete")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Files generated:")
    print(f"  - pred_vs_gt.png")
    print(f"  - pde_residual.png")
    print(f"  - residual_distribution.png")

    rmse = torch.sqrt(torch.mean((pred - gt)**2)).item()
    pde_rmse = torch.sqrt(torch.mean(residual**2)).item()
    print(f"\nMetrics:")
    print(f"  - RMSE (pred vs gt): {rmse:.6f}")
    print(f"  - PDE Residual RMSE: {pde_rmse:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
