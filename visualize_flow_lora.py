"""
LoRA Model Visualization for Flow Mixing.

This script follows the EXACT same inference logic as train_flow_lora.py validation.

Usage:
    python visualize_flow_lora.py --checkpoint checkpoints_flow_lora_v2/best_lora.pt --config configs/finetune_flow_v2.yaml --output_dir ./vis_results
"""

import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple

from dataset_flow import FlowMixingDataset, flow_mixing_collate_fn
from pde_loss_flow import flow_mixing_pde_loss
from pde_loss_flow_v2 import flow_mixing_pde_loss_v2
from model_lora import PDELoRAModel, load_lora_checkpoint


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_lora_model(config: dict, checkpoint_path: str, device: str = 'cuda'):
    """
    Load LoRA model correctly:
    1. Create model with pretrained base weights
    2. Load LoRA checkpoint
    """
    # Get pretrained path from config
    pretrained_path = config.get('model', {}).get('pretrained_path')

    if pretrained_path is None:
        raise ValueError("Config must specify 'model.pretrained_path' for LoRA inference")

    print(f"Loading base model from: {pretrained_path}")

    # Create model with pretrained base weights
    model = PDELoRAModel(config, pretrained_path=pretrained_path)

    # Load LoRA checkpoint
    print(f"Loading LoRA checkpoint from: {checkpoint_path}")
    load_lora_checkpoint(model.model, checkpoint_path)

    model = model.to(device)
    model.eval()

    return model


@torch.no_grad()
def run_inference(model, batch: dict, device: str = 'cuda') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run inference exactly like train_flow_lora.py validation.

    Returns:
        output: [B, 16, H, W, 6] model output
        target: [B, 16, H, W, 6] ground truth target
    """
    # Same as train_flow_lora.py line 273-278
    data = batch['data'].to(device=device, dtype=torch.bfloat16)

    input_data = data[:, :-1]  # [B, 16, H, W, 6] (t=0 to t=15)
    target = data[:, 1:]       # [B, 16, H, W, 6] (t=1 to t=16)

    output = model(input_data)  # [B, 16, H, W, 6]

    return output, target, input_data


def compute_pde_residual(
    output: torch.Tensor,
    input_data: torch.Tensor,
    batch: dict,
    config: dict,
    device: str = 'cuda',
) -> Tuple[torch.Tensor, float]:
    """
    Compute PDE residual exactly like train_flow_lora.py compute_pde_loss.

    Returns:
        residual: [B, T-1, H, W] PDE residual field
        pde_loss: float - MSE of residual
    """
    # Same as train_flow_lora.py line 155-156
    t0_frame = input_data[:, 0:1]  # [B, 1, H, W, 6]
    pred_with_t0 = torch.cat([t0_frame, output], dim=1)  # [B, 17, H, W, 6]

    # Extract u channel and convert to float32
    pred_u = pred_with_t0[..., :1].float()  # [B, 17, H, W, 1]

    # Get boundaries
    boundary_left = batch['boundary_left'].to(device).float()
    boundary_right = batch['boundary_right'].to(device).float()
    boundary_bottom = batch['boundary_bottom'].to(device).float()
    boundary_top = batch['boundary_top'].to(device).float()
    vtmax = batch['vtmax'].to(device).float()

    # Get dt from config
    dt = config.get('physics', {}).get('dt', 1/999)
    Lx = config.get('physics', {}).get('Lx', 1.0)
    Ly = config.get('physics', {}).get('Ly', 1.0)

    # Get pde_version from config (must match training!)
    pde_version = config.get('training', {}).get('pde_version', 'v1')

    # Use mean vtmax for batch
    vtmax_mean = vtmax.mean().item()

    # Select PDE loss function based on version (same as train_flow_lora.py)
    if pde_version == "v2":
        pde_loss, loss_time, loss_advection, residual = flow_mixing_pde_loss_v2(
            pred=pred_u,
            boundary_left=boundary_left,
            boundary_right=boundary_right,
            boundary_bottom=boundary_bottom,
            boundary_top=boundary_top,
            vtmax=vtmax_mean,
            dt=dt,
            Lx=Lx,
            Ly=Ly,
        )
    else:
        pde_loss, loss_time, loss_advection, residual = flow_mixing_pde_loss(
            pred=pred_u,
            boundary_left=boundary_left,
            boundary_right=boundary_right,
            boundary_bottom=boundary_bottom,
            boundary_top=boundary_top,
            vtmax=vtmax_mean,
            dt=dt,
            Lx=Lx,
            Ly=Ly,
        )

    return residual, pde_loss.item()


def compute_rmse_loss(output: torch.Tensor, target: torch.Tensor, channel: int = 0) -> float:
    """
    Compute RMSE loss for a specific channel.

    For Flow Mixing, only channel 0 (u) is real, others are padding.
    """
    # Only compute on the real channel
    out_ch = output[..., channel].float()
    tgt_ch = target[..., channel].float()
    mse = torch.mean((out_ch - tgt_ch) ** 2)
    rmse = torch.sqrt(mse + 1e-8)
    return rmse.item()


def plot_results(
    results: list,
    save_path: str,
):
    """
    Plot multiple samples: each row is GT | Prediction | PDE Residual.
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
        axes[row, 0].set_title('Ground Truth', fontsize=11)
        axes[row, 0].set_ylabel(f'Sample {sample_idx}\nvtmax={vtmax:.2f}', fontsize=10)
        axes[row, 0].set_xlabel('x')
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)

        # 2. Prediction
        im1 = axes[row, 1].imshow(pred_np, origin='lower', extent=extent, cmap='jet', vmin=vmin, vmax=vmax)
        pred_rmse = np.sqrt(np.mean((pred_np - gt_np)**2))
        axes[row, 1].set_title(f'Prediction (RMSE={pred_rmse:.4f})', fontsize=11)
        axes[row, 1].set_xlabel('x')
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)

        # 3. PDE Residual (use 95th percentile for colorbar)
        res_percentile = np.percentile(np.abs(res_np), 95)
        res_max = res_percentile if res_percentile > 0 else 1.0
        im2 = axes[row, 2].imshow(res_np, origin='lower', extent=extent, cmap='RdBu_r', vmin=-res_max, vmax=res_max)
        pde_mse = np.mean(res_np**2)
        axes[row, 2].set_title(f'PDE Residual (MSE={pde_mse:.2f})', fontsize=11)
        axes[row, 2].set_xlabel('x')
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="LoRA Model Visualization for Flow Mixing")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to LoRA checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='./vis_results', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples')
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

    # Print PDE version
    pde_version = config.get('training', {}).get('pde_version', 'v1')
    print(f"PDE Version: {pde_version} ({'central diff' if pde_version == 'v2' else '2nd upwind'})")

    # Load model correctly
    model = load_lora_model(config, args.checkpoint, device=device)

    # Create validation dataset
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

    sample_indices = np.random.choice(n_val_samples, min(num_samples, n_val_samples), replace=False)

    print(f"Visualizing {len(sample_indices)} samples...")

    results = []
    all_rmse = []
    all_pde_mse = []

    for i, sample_idx in enumerate(sample_indices):
        # Random clip within this sample
        clip_idx = np.random.randint(0, clips_per_sample)
        idx = sample_idx * clips_per_sample + clip_idx
        idx = min(idx, total_clips - 1)

        sample = dataset[idx]
        batch = flow_mixing_collate_fn([sample])

        # Run inference (exactly like train_flow_lora.py)
        output, target, input_data = run_inference(model, batch, device=device)

        vtmax = batch['vtmax'][0].item()

        # Compute PDE residual (exactly like train_flow_lora.py)
        residual, pde_mse = compute_pde_residual(
            output=output,
            input_data=input_data,
            batch=batch,
            config=config,
            device=device,
        )

        # Compute RMSE (exactly like train_flow_lora.py)
        rmse = compute_rmse_loss(output, target)

        # Get last timestep for visualization
        gt_last = target[0, -1, :, :, 0].float().cpu().numpy()
        pred_last = output[0, -1, :, :, 0].float().cpu().numpy()
        res_last = residual[0, -1].cpu().numpy()

        results.append({
            'gt': gt_last,
            'pred': pred_last,
            'residual': res_last,
            'vtmax': vtmax,
            'sample_idx': sample_idx,
        })

        all_rmse.append(rmse)
        all_pde_mse.append(pde_mse)

        # Per-sample metrics (last timestep only for display)
        last_rmse = np.sqrt(np.mean((pred_last - gt_last)**2))
        last_pde_mse = np.mean(res_last**2)
        print(f"  Sample {sample_idx}: vtmax={vtmax:.2f}, RMSE={last_rmse:.4f}, PDE_MSE(last)={last_pde_mse:.2f}, PDE_MSE(all)={pde_mse:.2f}")

    # Plot results
    output_filename = "visualization_lora.png"
    print("Plotting visualization...")
    plot_results(results, str(output_dir / output_filename))

    # Print summary
    print("\n" + "=" * 60)
    print("Visualization Complete")
    print("=" * 60)
    print(f"Output: {output_dir / output_filename}")
    print(f"\nAverage Metrics ({len(results)} samples):")
    print(f"  - RMSE (all 16 timesteps): {np.mean(all_rmse):.6f}")
    print(f"  - PDE Loss (MSE, all 16 timesteps): {np.mean(all_pde_mse):.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
