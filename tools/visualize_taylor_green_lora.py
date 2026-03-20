"""
LoRA V3 Model Visualization for Taylor-Green Vortex 2D (Incompressible NS).

Multi-GPU evaluation using Accelerate (same pattern as training).
Supports per-sample nu for PDE loss computation.

Usage:
    # Multi-GPU full evaluation
    torchrun --nproc_per_node=8 tools/visualize_taylor_green_lora.py \
        --config configs/finetune_taylor_green_v3.yaml \
        --checkpoint checkpoints_taylor_green_lora_v3/best_lora.pt --scan_all

    # Single-GPU visualization (plot only)
    python tools/visualize_taylor_green_lora.py \
        --config configs/finetune_taylor_green_v3.yaml \
        --checkpoint checkpoints_taylor_green_lora_v3/best_lora.pt --output_dir ./taylor_green_vis
"""

import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader
from accelerate import Accelerator

from finetune.dataset_finetune import (
    FinetuneDataset, FinetuneSampler, finetune_collate_fn,
)
from finetune.model_lora_v3 import PDELoRAModelV3, load_lora_checkpoint
from finetune.pde_loss_verified import TaylorGreenPDELoss

# Taylor-Green channel indices in 18-channel layout
CH_VX = 0       # vector: Vx
CH_VY = 1       # vector: Vy
CH_PRESS = 15   # scalar[12] = pressure


def _vrmse_np(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-8) -> float:
    """Variance-normalized RMSE."""
    mse = np.mean((pred - gt) ** 2)
    var = np.mean((gt - gt.mean()) ** 2)
    return float(np.sqrt(mse / (var + eps)))


def _vrmse_torch(gt: torch.Tensor, pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Variance-normalized RMSE (torch)."""
    mse = torch.mean((pred - gt) ** 2)
    var = torch.mean((gt - gt.mean()) ** 2)
    return torch.sqrt(mse / (var + eps))


def _nrmse_torch(gt: torch.Tensor, pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """nRMSE: sqrt(MSE(pred, gt) / MSE(0, gt))."""
    mse_pred = torch.mean((pred - gt) ** 2)
    mse_zero = torch.mean(gt ** 2)
    return torch.sqrt(mse_pred / (mse_zero + eps))


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA V3 Visualization for Taylor-Green 2D")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='./vis_results', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of samples to visualize')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--scan_all', action='store_true', help='Scan all validation clips (multi-GPU)')
    parser.add_argument('--t_input', type=int, default=None, help='Override t_input from config')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch_size for eval')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model(config: dict, checkpoint_path: str, is_main: bool = True):
    """Load LoRA V3 model."""
    pretrained_path = config.get('model', {}).get('pretrained_path')
    if pretrained_path is None:
        raise ValueError("Config must specify 'model.pretrained_path'")

    model = PDELoRAModelV3(
        config=config,
        pretrained_path=pretrained_path,
        freeze_encoder=config.get('model', {}).get('freeze_encoder', False),
        freeze_decoder=config.get('model', {}).get('freeze_decoder', False),
    )

    checkpoint = load_lora_checkpoint(model, checkpoint_path)

    if is_main:
        if 'metrics' in checkpoint:
            print(f"  Checkpoint metrics: {checkpoint['metrics']}")
        if 'global_step' in checkpoint:
            print(f"  Global step: {checkpoint['global_step']}")

    model = model.float()
    return model


def create_pde_loss_fn(config: dict) -> TaylorGreenPDELoss:
    physics = config.get('physics', {})
    return TaylorGreenPDELoss(
        nx=physics.get('nx', 256),
        ny=physics.get('ny', 256),
        Lx=physics.get('Lx', 6.283185307179586),
        Ly=physics.get('Ly', 6.283185307179586),
        dt=physics.get('dt', 0.001),
        nu=physics.get('nu_default', 0.001),
    )


def compute_pde_loss_from_output(
    output: torch.Tensor,
    input_data: torch.Tensor,
    pde_loss_fn: TaylorGreenPDELoss,
    nu: torch.Tensor,
) -> torch.Tensor:
    """Compute Taylor-Green PDE residual loss with per-sample nu."""
    with torch.autocast(device_type='cuda', enabled=False):
        t0_u = input_data[:, 0:1, :, :, CH_VX].float()
        t0_v = input_data[:, 0:1, :, :, CH_VY].float()
        t0_p = input_data[:, 0:1, :, :, CH_PRESS].float()

        out_u = output[:, :, :, :, CH_VX].float()
        out_v = output[:, :, :, :, CH_VY].float()
        out_p = output[:, :, :, :, CH_PRESS].float()

        u = torch.cat([t0_u, out_u], dim=1)
        v = torch.cat([t0_v, out_v], dim=1)
        p = torch.cat([t0_p, out_p], dim=1)

        total_loss, losses = pde_loss_fn(u, v, p, nu=nu.float())
    return total_loss


# ============================================================
# scan_all: Multi-GPU distributed evaluation
# ============================================================

@torch.no_grad()
def scan_all_distributed(accelerator, model, val_loader, config, t_input):
    """Distributed evaluation over ALL validation clips."""
    accelerator.wait_for_everyone()
    model.eval()

    pde_loss_fn = create_pde_loss_fn(config)

    total_pde = torch.zeros(1, device=accelerator.device)
    total_rmse_vx = torch.zeros(1, device=accelerator.device)
    total_rmse_vy = torch.zeros(1, device=accelerator.device)
    total_rmse_press = torch.zeros(1, device=accelerator.device)
    total_vrmse_vx = torch.zeros(1, device=accelerator.device)
    total_vrmse_vy = torch.zeros(1, device=accelerator.device)
    total_vrmse_press = torch.zeros(1, device=accelerator.device)
    total_rmse = torch.zeros(1, device=accelerator.device)
    total_vrmse_all = torch.zeros(1, device=accelerator.device)
    total_nrmse_all = torch.zeros(1, device=accelerator.device)
    num_batches = torch.zeros(1, device=accelerator.device)

    for batch in val_loader:
        data = batch['data'].to(device=accelerator.device, dtype=torch.float32)
        channel_mask = batch['channel_mask'].to(device=accelerator.device)
        nu = batch['nu'].to(device=accelerator.device)

        input_data = data[:, :t_input]
        target_data = data[:, 1:t_input + 1]

        output_norm, mean, std = model(input_data, return_normalized=True)
        output = output_norm * std + mean

        # PDE loss (per-sample nu)
        pde_loss = compute_pde_loss_from_output(output, input_data, pde_loss_fn, nu)

        # Per-channel RMSE
        rmse_vx = torch.sqrt(torch.mean((output[..., CH_VX] - target_data[..., CH_VX]) ** 2) + 1e-8)
        rmse_vy = torch.sqrt(torch.mean((output[..., CH_VY] - target_data[..., CH_VY]) ** 2) + 1e-8)
        rmse_press = torch.sqrt(torch.mean((output[..., CH_PRESS] - target_data[..., CH_PRESS]) ** 2) + 1e-8)

        vrmse_vx = _vrmse_torch(target_data[..., CH_VX], output[..., CH_VX])
        vrmse_vy = _vrmse_torch(target_data[..., CH_VY], output[..., CH_VY])
        vrmse_press = _vrmse_torch(target_data[..., CH_PRESS], output[..., CH_PRESS])

        # Overall RMSE (valid channels)
        valid_ch = (
            torch.where(channel_mask[0] > 0)[0]
            if channel_mask.dim() > 1
            else torch.where(channel_mask > 0)[0]
        )
        mse = torch.mean((output[..., valid_ch] - target_data[..., valid_ch]) ** 2)
        rmse = torch.sqrt(mse + 1e-8)

        # Combined vrmse/nrmse as mean of per-channel values
        nrmse_vx = _nrmse_torch(target_data[..., CH_VX], output[..., CH_VX])
        nrmse_vy = _nrmse_torch(target_data[..., CH_VY], output[..., CH_VY])
        nrmse_press = _nrmse_torch(target_data[..., CH_PRESS], output[..., CH_PRESS])
        vrmse_all = (vrmse_vx + vrmse_vy + vrmse_press) / 3
        nrmse_all = (nrmse_vx + nrmse_vy + nrmse_press) / 3

        total_pde += pde_loss.detach()
        total_rmse_vx += rmse_vx.detach()
        total_rmse_vy += rmse_vy.detach()
        total_rmse_press += rmse_press.detach()
        total_vrmse_vx += vrmse_vx.detach()
        total_vrmse_vy += vrmse_vy.detach()
        total_vrmse_press += vrmse_press.detach()
        total_rmse += rmse.detach()
        total_vrmse_all += vrmse_all.detach()
        total_nrmse_all += nrmse_all.detach()
        num_batches += 1

    accelerator.wait_for_everyone()
    total_pde = accelerator.reduce(total_pde, reduction='sum')
    total_rmse_vx = accelerator.reduce(total_rmse_vx, reduction='sum')
    total_rmse_vy = accelerator.reduce(total_rmse_vy, reduction='sum')
    total_rmse_press = accelerator.reduce(total_rmse_press, reduction='sum')
    total_vrmse_vx = accelerator.reduce(total_vrmse_vx, reduction='sum')
    total_vrmse_vy = accelerator.reduce(total_vrmse_vy, reduction='sum')
    total_vrmse_press = accelerator.reduce(total_vrmse_press, reduction='sum')
    total_rmse = accelerator.reduce(total_rmse, reduction='sum')
    total_vrmse_all = accelerator.reduce(total_vrmse_all, reduction='sum')
    total_nrmse_all = accelerator.reduce(total_nrmse_all, reduction='sum')
    num_batches = accelerator.reduce(num_batches, reduction='sum')
    accelerator.wait_for_everyone()

    n = num_batches.item()
    if n > 0:
        return {
            'pde': (total_pde / n).item(),
            'rmse_vx': (total_rmse_vx / n).item(),
            'rmse_vy': (total_rmse_vy / n).item(),
            'rmse_press': (total_rmse_press / n).item(),
            'vrmse_vx': (total_vrmse_vx / n).item(),
            'vrmse_vy': (total_vrmse_vy / n).item(),
            'vrmse_press': (total_vrmse_press / n).item(),
            'rmse': (total_rmse / n).item(),
            'vrmse_all': (total_vrmse_all / n).item(),
            'nrmse_all': (total_nrmse_all / n).item(),
            'num_batches': int(n),
        }
    return {'pde': 0, 'rmse_vx': 0, 'rmse_vy': 0, 'rmse_press': 0,
            'vrmse_vx': 0, 'vrmse_vy': 0, 'vrmse_press': 0,
            'rmse': 0, 'vrmse_all': 0, 'nrmse_all': 0, 'num_batches': 0}


# ============================================================
# Visualization (main process only)
# ============================================================

@torch.no_grad()
def run_visualization(model, dataset, config, device, t_input, num_samples, seed, output_dir):
    """Run visualization on main process only."""
    model.eval()
    pde_loss_fn = create_pde_loss_fn(config)

    total_clips = len(dataset)
    n_val_samples = len(dataset.sample_indices)
    clips_per_sample = total_clips // n_val_samples if n_val_samples > 0 else 1

    np.random.seed(seed)
    sample_indices = np.random.choice(
        n_val_samples, min(num_samples, n_val_samples), replace=False,
    )

    n_vis = len(sample_indices)
    if clips_per_sample > 1 and n_vis > 1:
        clip_offsets = np.linspace(0, clips_per_sample - 1, n_vis, dtype=int)
    else:
        clip_offsets = np.zeros(n_vis, dtype=int)

    print(f"Visualizing {n_vis} samples...")

    results = []
    all_metrics = {'rmse_vx': [], 'rmse_vy': [], 'rmse_press': [],
                    'vrmse_vx': [], 'vrmse_vy': [], 'vrmse_press': [],
                    'vrmse_all': [], 'nrmse_all': [], 'pde': []}

    for i, sample_idx in enumerate(sample_indices):
        idx = sample_idx * clips_per_sample + clip_offsets[i]
        idx = min(idx, total_clips - 1)

        sample = dataset[idx]
        batch = finetune_collate_fn([sample])
        start_t = dataset.clips[idx][2]
        nu_val = sample['nu'].item()

        data = batch['data'].to(device=device, dtype=torch.float32)
        nu = batch['nu'].to(device=device)
        input_data = data[:, :t_input]
        target = data[:, 1:t_input + 1]

        output_norm, mean, std = model(input_data, return_normalized=True)
        output = output_norm * std + mean

        pde_loss = compute_pde_loss_from_output(output, input_data, pde_loss_fn, nu).item()

        rmse_vx = torch.sqrt(torch.mean((output[..., CH_VX] - target[..., CH_VX]) ** 2) + 1e-8).item()
        rmse_vy = torch.sqrt(torch.mean((output[..., CH_VY] - target[..., CH_VY]) ** 2) + 1e-8).item()
        rmse_press = torch.sqrt(torch.mean((output[..., CH_PRESS] - target[..., CH_PRESS]) ** 2) + 1e-8).item()

        vrmse_vx = _vrmse_torch(target[..., CH_VX], output[..., CH_VX]).item()
        vrmse_vy = _vrmse_torch(target[..., CH_VY], output[..., CH_VY]).item()
        vrmse_press = _vrmse_torch(target[..., CH_PRESS], output[..., CH_PRESS]).item()

        # Combined vrmse/nrmse as mean of per-channel values
        nrmse_vx = _nrmse_torch(target[..., CH_VX], output[..., CH_VX]).item()
        nrmse_vy = _nrmse_torch(target[..., CH_VY], output[..., CH_VY]).item()
        nrmse_press = _nrmse_torch(target[..., CH_PRESS], output[..., CH_PRESS]).item()
        vrmse_all_vis = (vrmse_vx + vrmse_vy + vrmse_press) / 3
        nrmse_all_vis = (nrmse_vx + nrmse_vy + nrmse_press) / 3

        # Last timestep for plotting
        last = -1
        res_data = {
            'gt_vx': target[0, last, :, :, CH_VX].float().cpu().numpy(),
            'pred_vx': output[0, last, :, :, CH_VX].float().cpu().numpy(),
            'gt_vy': target[0, last, :, :, CH_VY].float().cpu().numpy(),
            'pred_vy': output[0, last, :, :, CH_VY].float().cpu().numpy(),
            'gt_press': target[0, last, :, :, CH_PRESS].float().cpu().numpy(),
            'pred_press': output[0, last, :, :, CH_PRESS].float().cpu().numpy(),
            'sample_idx': sample_idx, 'start_t': start_t, 'nu': nu_val,
        }
        results.append(res_data)

        all_metrics['rmse_vx'].append(rmse_vx)
        all_metrics['rmse_vy'].append(rmse_vy)
        all_metrics['rmse_press'].append(rmse_press)
        all_metrics['vrmse_vx'].append(vrmse_vx)
        all_metrics['vrmse_vy'].append(vrmse_vy)
        all_metrics['vrmse_press'].append(vrmse_press)
        all_metrics['vrmse_all'].append(vrmse_all_vis)
        all_metrics['nrmse_all'].append(nrmse_all_vis)
        all_metrics['pde'].append(pde_loss)

        print(f"  Sample {sample_idx}: t_start={start_t}, nu={nu_val:.6f}, "
              f"RMSE_vx={rmse_vx:.6f}, RMSE_vy={rmse_vy:.6f}, "
              f"RMSE_p={rmse_press:.6f}, "
              f"VRMSE_vx={vrmse_vx:.6f}, VRMSE_vy={vrmse_vy:.6f}, "
              f"VRMSE_p={vrmse_press:.6f}, PDE={pde_loss:.6f}")

    # Plot velocity fields
    print("Plotting velocity visualization...")
    plot_velocity(results, str(output_dir / "visualization_taylor_green_velocity.png"))

    # Plot pressure field
    print("Plotting pressure visualization...")
    plot_pressure(results, str(output_dir / "visualization_taylor_green_pressure.png"))

    print(f"\n{'='*60}")
    print("Visualization Complete (Taylor-Green LoRA)")
    print(f"{'='*60}")
    print(f"\nAverage Metrics ({len(results)} samples):")
    print(f"  - RMSE (Vx):     {np.mean(all_metrics['rmse_vx']):.6f}")
    print(f"  - RMSE (Vy):     {np.mean(all_metrics['rmse_vy']):.6f}")
    print(f"  - RMSE (press):  {np.mean(all_metrics['rmse_press']):.6f}")
    print(f"  - VRMSE (Vx):    {np.mean(all_metrics['vrmse_vx']):.6f}")
    print(f"  - VRMSE (Vy):    {np.mean(all_metrics['vrmse_vy']):.6f}")
    print(f"  - VRMSE (press): {np.mean(all_metrics['vrmse_press']):.6f}")
    print(f"  - VRMSE (all):   {np.mean(all_metrics['vrmse_all']):.6f}")
    print(f"  - nRMSE (all):   {np.mean(all_metrics['nrmse_all']):.6f}")
    print(f"  - PDE Loss:      {np.mean(all_metrics['pde']):.6f}")
    print(f"{'='*60}")


def plot_velocity(results: list, save_path: str):
    """Plot GT vs Prediction vs Error for Vx and Vy."""
    n_samples = len(results)
    fig, axes = plt.subplots(n_samples, 6, figsize=(24, 4 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for row, res in enumerate(results):
        gt_vx = res['gt_vx']
        pred_vx = res['pred_vx']
        err_vx = pred_vx - gt_vx
        gt_vy = res['gt_vy']
        pred_vy = res['pred_vy']
        err_vy = pred_vy - gt_vy
        sample_idx = res['sample_idx']
        start_t = res.get('start_t', '?')
        nu_val = res.get('nu', '?')

        # Vx
        vmin_vx = min(gt_vx.min(), pred_vx.min())
        vmax_vx = max(gt_vx.max(), pred_vx.max())

        im0 = axes[row, 0].imshow(gt_vx, origin='lower', cmap='RdBu_r',
                                    vmin=vmin_vx, vmax=vmax_vx)
        axes[row, 0].set_title('GT (Vx)', fontsize=11)
        axes[row, 0].set_ylabel(f'S{sample_idx} t0={start_t}\nnu={nu_val:.4f}', fontsize=9)
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)

        im1 = axes[row, 1].imshow(pred_vx, origin='lower', cmap='RdBu_r',
                                    vmin=vmin_vx, vmax=vmax_vx)
        rmse_vx = np.sqrt(np.mean(err_vx**2))
        vrmse_vx = _vrmse_np(gt_vx, pred_vx)
        axes[row, 1].set_title(f'Pred Vx (RMSE={rmse_vx:.4f} VRMSE={vrmse_vx:.4f})', fontsize=11)
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)

        err_vx_max = max(np.percentile(np.abs(err_vx), 95), 1e-6)
        im2 = axes[row, 2].imshow(err_vx, origin='lower', cmap='RdBu_r',
                                    vmin=-err_vx_max, vmax=err_vx_max)
        axes[row, 2].set_title(f'Error Vx (MAE={np.mean(np.abs(err_vx)):.4f})', fontsize=11)
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)

        # Vy
        vmin_vy = min(gt_vy.min(), pred_vy.min())
        vmax_vy = max(gt_vy.max(), pred_vy.max())

        im3 = axes[row, 3].imshow(gt_vy, origin='lower', cmap='RdBu_r',
                                    vmin=vmin_vy, vmax=vmax_vy)
        axes[row, 3].set_title('GT (Vy)', fontsize=11)
        plt.colorbar(im3, ax=axes[row, 3], fraction=0.046, pad=0.04)

        im4 = axes[row, 4].imshow(pred_vy, origin='lower', cmap='RdBu_r',
                                    vmin=vmin_vy, vmax=vmax_vy)
        rmse_vy = np.sqrt(np.mean(err_vy**2))
        vrmse_vy = _vrmse_np(gt_vy, pred_vy)
        axes[row, 4].set_title(f'Pred Vy (RMSE={rmse_vy:.4f} VRMSE={vrmse_vy:.4f})', fontsize=11)
        plt.colorbar(im4, ax=axes[row, 4], fraction=0.046, pad=0.04)

        err_vy_max = max(np.percentile(np.abs(err_vy), 95), 1e-6)
        im5 = axes[row, 5].imshow(err_vy, origin='lower', cmap='RdBu_r',
                                    vmin=-err_vy_max, vmax=err_vy_max)
        axes[row, 5].set_title(f'Error Vy (MAE={np.mean(np.abs(err_vy)):.4f})', fontsize=11)
        plt.colorbar(im5, ax=axes[row, 5], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_pressure(results: list, save_path: str):
    """Plot GT vs Prediction vs Error for pressure."""
    n_samples = len(results)
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for row, res in enumerate(results):
        gt_p = res['gt_press']
        pred_p = res['pred_press']
        err_p = pred_p - gt_p
        sample_idx = res['sample_idx']
        start_t = res.get('start_t', '?')
        nu_val = res.get('nu', '?')

        vmin_p = min(gt_p.min(), pred_p.min())
        vmax_p = max(gt_p.max(), pred_p.max())

        im0 = axes[row, 0].imshow(gt_p, origin='lower', cmap='coolwarm',
                                    vmin=vmin_p, vmax=vmax_p)
        axes[row, 0].set_title('GT (pressure)', fontsize=11)
        axes[row, 0].set_ylabel(f'S{sample_idx} t0={start_t}\nnu={nu_val:.4f}', fontsize=9)
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)

        im1 = axes[row, 1].imshow(pred_p, origin='lower', cmap='coolwarm',
                                    vmin=vmin_p, vmax=vmax_p)
        rmse_p = np.sqrt(np.mean(err_p**2))
        vrmse_p = _vrmse_np(gt_p, pred_p)
        axes[row, 1].set_title(f'Pred press (RMSE={rmse_p:.4f} VRMSE={vrmse_p:.4f})', fontsize=11)
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)

        err_p_max = max(np.percentile(np.abs(err_p), 95), 1e-6)
        im2 = axes[row, 2].imshow(err_p, origin='lower', cmap='RdBu_r',
                                    vmin=-err_p_max, vmax=err_p_max)
        axes[row, 2].set_title(f'Error press (MAE={np.mean(np.abs(err_p)):.4f})', fontsize=11)
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    config = load_config(args.config)

    t_input = args.t_input or config.get('dataset', {}).get('t_input', 8)
    temporal_length = t_input + 1
    batch_size = args.batch_size or config.get('dataloader', {}).get('batch_size', 4)

    accelerator = Accelerator()
    is_main = accelerator.is_main_process

    if is_main:
        print(f"{'='*60}")
        print(f"Taylor-Green 2D LoRA V3 Evaluation")
        print(f"{'='*60}")
        print(f"  Devices: {accelerator.num_processes}")
        print(f"  t_input: {t_input}")
        print(f"  batch_size: {batch_size}")
        print(f"  mode: {'scan_all' if args.scan_all else 'visualize'}")
        print(f"{'='*60}")

    model = load_model(config, args.checkpoint, is_main=is_main)

    val_dataset = FinetuneDataset(
        data_path=config['dataset']['path'],
        temporal_length=temporal_length,
        split='val',
        train_ratio=config['dataset'].get('train_ratio', 0.9),
        seed=config['dataset'].get('seed', 42),
        clips_per_sample=None,
        vector_dim=config['dataset'].get('vector_dim', 2),
        val_time_interval=config['dataset'].get('val_time_interval', 8),
    )

    if is_main:
        print(f"  Val samples: {len(val_dataset.sample_indices)}, "
              f"total clips: {len(val_dataset)}")

    if args.scan_all:
        val_sampler = FinetuneSampler(
            val_dataset, batch_size, shuffle=False,
            seed=config['dataset'].get('seed', 42),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            collate_fn=finetune_collate_fn,
            num_workers=config.get('dataloader', {}).get('num_workers', 4),
            pin_memory=True,
        )

        model, val_loader = accelerator.prepare(model, val_loader)

        if is_main:
            print(f"\nScanning all {len(val_dataset)} clips "
                  f"across {accelerator.num_processes} GPUs...")

        results = scan_all_distributed(
            accelerator, model, val_loader, config, t_input,
        )

        if is_main:
            print(f"\n{'='*60}")
            print(f"Results ({results['num_batches']} batches across all ranks):")
            print(f"  PDE Loss:      {results['pde']:.6f}")
            print(f"  RMSE (Vx):     {results['rmse_vx']:.6f}")
            print(f"  RMSE (Vy):     {results['rmse_vy']:.6f}")
            print(f"  RMSE (press):  {results['rmse_press']:.6f}")
            print(f"  VRMSE (Vx):    {results['vrmse_vx']:.6f}")
            print(f"  VRMSE (Vy):    {results['vrmse_vy']:.6f}")
            print(f"  VRMSE (press): {results['vrmse_press']:.6f}")
            print(f"  VRMSE (all):   {results['vrmse_all']:.6f}")
            print(f"  nRMSE (all):   {results['nrmse_all']:.6f}")
            print(f"  RMSE:          {results['rmse']:.6f}")
            print(f"{'='*60}")

    else:
        model = model.to(accelerator.device)

        if is_main:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            run_visualization(
                model=model,
                dataset=val_dataset,
                config=config,
                device=accelerator.device,
                t_input=t_input,
                num_samples=args.num_samples,
                seed=args.seed,
                output_dir=output_dir,
            )


if __name__ == "__main__":
    main()
