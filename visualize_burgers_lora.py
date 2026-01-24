"""
LoRA Model Visualization for 2D Burgers Equation.

This script follows the EXACT same inference logic as train_burgers_lora.py validation.

Usage:
    python visualize_burgers_lora.py --checkpoint checkpoints_burgers_lora_v3/best_lora.pt --config configs/finetune_burgers.yaml --output_dir ./vis_results
"""

import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple

from dataset_burgers import BurgersDataset, burgers_collate_fn
from pde_loss import burgers_pde_loss
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

    # Convert to fp32 for numerical stability (base model may be saved in bf16)
    model = model.float()
    model = model.to(device)
    model.eval()

    return model


@torch.no_grad()
def run_inference(model, batch: dict, device: str = 'cuda') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run inference exactly like train_burgers_lora.py validation.

    Returns:
        output: [B, 16, H, W, 6] model output
        target: [B, 16, H, W, 6] ground truth target
    """
    # Use fp32 for numerical stability
    data = batch['data'].to(device=device, dtype=torch.float32)

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
) -> Tuple[torch.Tensor, torch.Tensor, float, float, float]:
    """
    Compute PDE residual exactly like train_burgers_lora.py compute_pde_loss.

    Returns:
        residual_u: [B, T-1, H, W] PDE residual for u
        residual_v: [B, T-1, H, W] PDE residual for v
        pde_loss: float - total PDE loss
        loss_u: float - u equation loss
        loss_v: float - v equation loss
    """
    # Same as train_burgers_lora.py
    t0_frame = input_data[:, 0:1, ..., :2].float()  # [B, 1, H, W, 2]
    output_uv = output[..., :2].float()  # [B, 16, H, W, 2]
    pred_with_t0 = torch.cat([t0_frame, output_uv], dim=1)  # [B, 17, H, W, 2]

    # Get boundaries
    boundary_left = batch['boundary_left'].to(device).float()
    boundary_right = batch['boundary_right'].to(device).float()
    boundary_bottom = batch['boundary_bottom'].to(device).float()
    boundary_top = batch['boundary_top'].to(device).float()
    nu = batch['nu'].to(device).float()

    # Get physics params from config
    dt = config.get('physics', {}).get('dt', 1/999)
    Lx = config.get('physics', {}).get('Lx', 1.0)
    Ly = config.get('physics', {}).get('Ly', 1.0)

    # Use mean nu for batch
    nu_mean = nu.mean().item()

    # Compute PDE loss (n-PINN conservative flux form)
    pde_loss, loss_u, loss_v, residual_u, residual_v = burgers_pde_loss(
        pred=pred_with_t0,
        boundary_left=boundary_left,
        boundary_right=boundary_right,
        boundary_bottom=boundary_bottom,
        boundary_top=boundary_top,
        nu=nu_mean,
        dt=dt,
        Lx=Lx,
        Ly=Ly,
    )

    return residual_u, residual_v, pde_loss.item(), loss_u.item(), loss_v.item()


def compute_rmse_loss(output: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    """
    Compute RMSE loss for u and v channels.

    Returns:
        rmse_u: RMSE for u channel
        rmse_v: RMSE for v channel
    """
    out_u = output[..., 0].float()
    tgt_u = target[..., 0].float()
    out_v = output[..., 1].float()
    tgt_v = target[..., 1].float()

    mse_u = torch.mean((out_u - tgt_u) ** 2)
    mse_v = torch.mean((out_v - tgt_v) ** 2)

    rmse_u = torch.sqrt(mse_u + 1e-8).item()
    rmse_v = torch.sqrt(mse_v + 1e-8).item()

    return rmse_u, rmse_v


def plot_results(
    results: list,
    save_path: str,
):
    """
    Plot multiple samples: each row shows u and v channels.
    Layout: GT_u | Pred_u | Residual_u | GT_v | Pred_v | Residual_v
    """
    n_samples = len(results)
    fig, axes = plt.subplots(n_samples, 6, figsize=(24, 4 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    extent = [0, 1, 0, 1]

    for row, res in enumerate(results):
        gt_u = res['gt_u']
        pred_u = res['pred_u']
        res_u = res['residual_u']
        gt_v = res['gt_v']
        pred_v = res['pred_v']
        res_v = res['residual_v']
        nu = res['nu']
        sample_idx = res['sample_idx']

        # ===== U channel =====
        vmin_u = min(gt_u.min(), pred_u.min())
        vmax_u = max(gt_u.max(), pred_u.max())

        # GT u
        im0 = axes[row, 0].imshow(gt_u, origin='lower', extent=extent, cmap='jet', vmin=vmin_u, vmax=vmax_u)
        axes[row, 0].set_title('GT (u)', fontsize=11)
        axes[row, 0].set_ylabel(f'Sample {sample_idx}\nnu={nu:.3f}', fontsize=10)
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)

        # Pred u
        im1 = axes[row, 1].imshow(pred_u, origin='lower', extent=extent, cmap='jet', vmin=vmin_u, vmax=vmax_u)
        rmse_u = np.sqrt(np.mean((pred_u - gt_u)**2))
        axes[row, 1].set_title(f'Pred u (RMSE={rmse_u:.4f})', fontsize=11)
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)

        # Residual u
        res_u_max = np.percentile(np.abs(res_u), 95)
        res_u_max = res_u_max if res_u_max > 0 else 1.0
        im2 = axes[row, 2].imshow(res_u, origin='lower', extent=extent, cmap='RdBu_r', vmin=-res_u_max, vmax=res_u_max)
        mse_u = np.mean(res_u**2)
        axes[row, 2].set_title(f'Residual u (MSE={mse_u:.2f})', fontsize=11)
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)

        # ===== V channel =====
        vmin_v = min(gt_v.min(), pred_v.min())
        vmax_v = max(gt_v.max(), pred_v.max())

        # GT v
        im3 = axes[row, 3].imshow(gt_v, origin='lower', extent=extent, cmap='jet', vmin=vmin_v, vmax=vmax_v)
        axes[row, 3].set_title('GT (v)', fontsize=11)
        plt.colorbar(im3, ax=axes[row, 3], fraction=0.046, pad=0.04)

        # Pred v
        im4 = axes[row, 4].imshow(pred_v, origin='lower', extent=extent, cmap='jet', vmin=vmin_v, vmax=vmax_v)
        rmse_v = np.sqrt(np.mean((pred_v - gt_v)**2))
        axes[row, 4].set_title(f'Pred v (RMSE={rmse_v:.4f})', fontsize=11)
        plt.colorbar(im4, ax=axes[row, 4], fraction=0.046, pad=0.04)

        # Residual v
        res_v_max = np.percentile(np.abs(res_v), 95)
        res_v_max = res_v_max if res_v_max > 0 else 1.0
        im5 = axes[row, 5].imshow(res_v, origin='lower', extent=extent, cmap='RdBu_r', vmin=-res_v_max, vmax=res_v_max)
        mse_v = np.mean(res_v**2)
        axes[row, 5].set_title(f'Residual v (MSE={mse_v:.2f})', fontsize=11)
        plt.colorbar(im5, ax=axes[row, 5], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="LoRA Model Visualization for Burgers Equation")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to LoRA checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='./vis_results', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--scan_all', action='store_true', help='Scan all validation samples')
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

    # Load model correctly
    model = load_lora_model(config, args.checkpoint, device=device)

    # Create validation dataset
    dataset = BurgersDataset(
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

    print(f"Validation: {n_val_samples} samples, {clips_per_sample} clips/sample")

    # ========== SCAN ALL MODE ==========
    if args.scan_all:
        print(f"\n{'='*60}")
        print(f"SCANNING ALL {n_val_samples} VALIDATION SAMPLES")
        print(f"{'='*60}")

        all_pde = []
        all_loss_u = []
        all_loss_v = []
        all_rmse_u = []
        all_rmse_v = []

        for sample_idx in range(n_val_samples):
            idx = sample_idx * clips_per_sample
            sample = dataset[idx]
            batch = burgers_collate_fn([sample])

            output, target, input_data = run_inference(model, batch, device=device)
            nu = batch['nu'][0].item()

            # Compute PDE residual
            _, _, pde_loss, loss_u, loss_v = compute_pde_residual(
                output=output,
                input_data=input_data,
                batch=batch,
                config=config,
                device=device,
            )

            # Compute RMSE
            rmse_u, rmse_v = compute_rmse_loss(output, target)

            all_pde.append(pde_loss)
            all_loss_u.append(loss_u)
            all_loss_v.append(loss_v)
            all_rmse_u.append(rmse_u)
            all_rmse_v.append(rmse_v)
            print(f"  Sample {sample_idx:2d}: nu={nu:.3f}, PDE={pde_loss:.4f}, RMSE_u={rmse_u:.4f}, RMSE_v={rmse_v:.4f}")

        all_pde = np.array(all_pde)
        all_rmse_u = np.array(all_rmse_u)
        all_rmse_v = np.array(all_rmse_v)

        print(f"\n{'='*60}")
        print(f"Statistics ({n_val_samples} samples):")
        print(f"  PDE Loss:  Mean={np.mean(all_pde):.4f}, Std={np.std(all_pde):.4f}")
        print(f"             Min={np.min(all_pde):.4f} (Sample {np.argmin(all_pde)})")
        print(f"             Max={np.max(all_pde):.4f} (Sample {np.argmax(all_pde)})")
        print(f"  RMSE (u):  Mean={np.mean(all_rmse_u):.6f}, Std={np.std(all_rmse_u):.6f}")
        print(f"  RMSE (v):  Mean={np.mean(all_rmse_v):.6f}, Std={np.std(all_rmse_v):.6f}")
        print(f"{'='*60}")
        return

    # ========== NORMAL VISUALIZATION MODE ==========
    sample_indices = np.random.choice(n_val_samples, min(num_samples, n_val_samples), replace=False)
    print(f"Visualizing {len(sample_indices)} samples...")

    results = []
    all_rmse_u = []
    all_rmse_v = []
    all_pde_loss = []

    for i, sample_idx in enumerate(sample_indices):
        # Random clip within this sample
        clip_idx = np.random.randint(0, clips_per_sample)
        idx = sample_idx * clips_per_sample + clip_idx
        idx = min(idx, total_clips - 1)

        sample = dataset[idx]
        batch = burgers_collate_fn([sample])

        # Run inference (exactly like train_burgers_lora.py)
        output, target, input_data = run_inference(model, batch, device=device)

        nu = batch['nu'][0].item()

        # Compute PDE residual (exactly like train_burgers_lora.py)
        residual_u, residual_v, pde_loss, loss_u, loss_v = compute_pde_residual(
            output=output,
            input_data=input_data,
            batch=batch,
            config=config,
            device=device,
        )

        # Compute RMSE
        rmse_u, rmse_v = compute_rmse_loss(output, target)

        # Get last timestep for visualization
        gt_u = target[0, -1, :, :, 0].float().cpu().numpy()
        gt_v = target[0, -1, :, :, 1].float().cpu().numpy()
        pred_u = output[0, -1, :, :, 0].float().cpu().numpy()
        pred_v = output[0, -1, :, :, 1].float().cpu().numpy()
        res_u_last = residual_u[0, -1].cpu().numpy()
        res_v_last = residual_v[0, -1].cpu().numpy()

        results.append({
            'gt_u': gt_u,
            'gt_v': gt_v,
            'pred_u': pred_u,
            'pred_v': pred_v,
            'residual_u': res_u_last,
            'residual_v': res_v_last,
            'nu': nu,
            'sample_idx': sample_idx,
        })

        all_rmse_u.append(rmse_u)
        all_rmse_v.append(rmse_v)
        all_pde_loss.append(pde_loss)

        # Per-sample metrics (last timestep only for display)
        last_rmse_u = np.sqrt(np.mean((pred_u - gt_u)**2))
        last_rmse_v = np.sqrt(np.mean((pred_v - gt_v)**2))
        print(f"  Sample {sample_idx}: nu={nu:.3f}, RMSE_u={last_rmse_u:.4f}, RMSE_v={last_rmse_v:.4f}, PDE={pde_loss:.2f}")

    # Plot results
    output_filename = "visualization_burgers_lora.png"
    print("Plotting visualization...")
    plot_results(results, str(output_dir / output_filename))

    # Print summary
    print("\n" + "=" * 60)
    print("Visualization Complete")
    print("=" * 60)
    print(f"Output: {output_dir / output_filename}")
    print(f"\nAverage Metrics ({len(results)} samples, all 16 timesteps):")
    print(f"  - RMSE (u): {np.mean(all_rmse_u):.6f}")
    print(f"  - RMSE (v): {np.mean(all_rmse_v):.6f}")
    print(f"  - PDE Loss: {np.mean(all_pde_loss):.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
