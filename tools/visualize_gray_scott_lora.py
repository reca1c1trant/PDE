"""
LoRA V3 Model Visualization for Gray-Scott Reaction-Diffusion.

Multi-GPU evaluation using Accelerate (same pattern as training).

Usage:
    # Multi-GPU full evaluation
    torchrun --nproc_per_node=8 tools/visualize_gray_scott_lora.py \
        --config configs/finetune_gray_scott_v3.yaml \
        --checkpoint checkpoints_gray_scott_lora_v3/best_lora.pt --scan_all

    # Single-GPU visualization (plot only)
    python tools/visualize_gray_scott_lora.py \
        --config configs/finetune_gray_scott_v3.yaml \
        --checkpoint checkpoints_gray_scott_lora_v3/best_lora.pt --output_dir ./gray_scott_vis
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
from finetune.pde_loss_verified import GrayScottPDELoss

# Gray-Scott channel indices in 18-channel layout
CH_A = 5  # scalar[2] = concentration_u (activator)
CH_B = 6  # scalar[3] = concentration_v (inhibitor)


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


def _nrmse_np(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-8) -> float:
    """nRMSE: sqrt(MSE(pred, gt) / MSE(0, gt))."""
    mse_pred = np.mean((pred - gt) ** 2)
    mse_zero = np.mean(gt ** 2)
    return float(np.sqrt(mse_pred / (mse_zero + eps)))


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA V3 Visualization for Gray-Scott")
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


def create_pde_loss_fn(config: dict) -> GrayScottPDELoss:
    physics = config.get('physics', {})
    return GrayScottPDELoss(
        nx=physics.get('nx', 128),
        ny=physics.get('ny', 128),
        dx=physics.get('dx', 2.0 / 128),
        dy=physics.get('dy', 2.0 / 128),
        dt=physics.get('dt', 10.0),
        F=physics.get('F', 0.098),
        k=physics.get('k', 0.057),
        D_A=physics.get('D_A', 1.81e-5),
        D_B=physics.get('D_B', 1.39e-5),
    )


def compute_pde_loss_from_output(
    output: torch.Tensor,
    input_data: torch.Tensor,
    pde_loss_fn: GrayScottPDELoss,
) -> torch.Tensor:
    """Compute Gray-Scott PDE residual loss."""
    with torch.autocast(device_type='cuda', enabled=False):
        t0_A = input_data[:, 0:1, :, :, CH_A].float()
        t0_B = input_data[:, 0:1, :, :, CH_B].float()
        out_A = output[:, :, :, :, CH_A].float()
        out_B = output[:, :, :, :, CH_B].float()
        A = torch.cat([t0_A, out_A], dim=1)
        B = torch.cat([t0_B, out_B], dim=1)
        total_loss, losses = pde_loss_fn(A, B)
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

    max_batches = len(val_loader)
    local_pde = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_rmse_A = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_rmse_B = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_vrmse_A = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_vrmse_B = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_nrmse_A = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_nrmse_B = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_rmse = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_vrmse_all = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_nrmse_all = torch.full((max_batches,), float('nan'), device=accelerator.device)

    for i, batch in enumerate(val_loader):
        data = batch['data'].to(device=accelerator.device, dtype=torch.float32)
        channel_mask = batch['channel_mask'].to(device=accelerator.device)

        input_data = data[:, :t_input]
        target_data = data[:, 1:t_input + 1]

        output_norm, mean, std = model(input_data, return_normalized=True)
        output = output_norm * std + mean

        pde_loss = compute_pde_loss_from_output(output, input_data, pde_loss_fn)

        out_A = output[..., CH_A].float()
        tgt_A = target_data[..., CH_A].float()
        out_B = output[..., CH_B].float()
        tgt_B = target_data[..., CH_B].float()

        rmse_A = torch.sqrt(torch.mean((out_A - tgt_A) ** 2) + 1e-8)
        rmse_B = torch.sqrt(torch.mean((out_B - tgt_B) ** 2) + 1e-8)

        vrmse_A = _vrmse_torch(tgt_A, out_A)
        vrmse_B = _vrmse_torch(tgt_B, out_B)

        nrmse_A = _nrmse_torch(tgt_A, out_A)
        nrmse_B = _nrmse_torch(tgt_B, out_B)

        valid_ch = (
            torch.where(channel_mask[0] > 0)[0]
            if channel_mask.dim() > 1
            else torch.where(channel_mask > 0)[0]
        )
        mse = torch.mean((output[..., valid_ch] - target_data[..., valid_ch]) ** 2)
        rmse = torch.sqrt(mse + 1e-8)

        # Combined vrmse/nrmse: mean of per-channel values
        vrmse_all = (vrmse_A + vrmse_B) / 2
        nrmse_all = (nrmse_A + nrmse_B) / 2

        local_pde[i] = pde_loss.detach()
        local_rmse_A[i] = rmse_A.detach()
        local_rmse_B[i] = rmse_B.detach()
        local_vrmse_A[i] = vrmse_A.detach()
        local_vrmse_B[i] = vrmse_B.detach()
        local_nrmse_A[i] = nrmse_A.detach()
        local_nrmse_B[i] = nrmse_B.detach()
        local_rmse[i] = rmse.detach()
        local_vrmse_all[i] = vrmse_all.detach()
        local_nrmse_all[i] = nrmse_all.detach()

    accelerator.wait_for_everyone()
    all_pde = accelerator.gather(local_pde)
    all_rmse_A = accelerator.gather(local_rmse_A)
    all_rmse_B = accelerator.gather(local_rmse_B)
    all_vrmse_A = accelerator.gather(local_vrmse_A)
    all_vrmse_B = accelerator.gather(local_vrmse_B)
    all_nrmse_A = accelerator.gather(local_nrmse_A)
    all_nrmse_B = accelerator.gather(local_nrmse_B)
    all_rmse = accelerator.gather(local_rmse)
    all_vrmse_all = accelerator.gather(local_vrmse_all)
    all_nrmse_all = accelerator.gather(local_nrmse_all)
    accelerator.wait_for_everyone()

    valid_mask = ~torch.isnan(all_pde)
    valid_pde = all_pde[valid_mask].cpu().numpy()
    valid_rmse_A = all_rmse_A[valid_mask].cpu().numpy()
    valid_rmse_B = all_rmse_B[valid_mask].cpu().numpy()
    valid_vrmse_A = all_vrmse_A[valid_mask].cpu().numpy()
    valid_vrmse_B = all_vrmse_B[valid_mask].cpu().numpy()
    valid_nrmse_A = all_nrmse_A[valid_mask].cpu().numpy()
    valid_nrmse_B = all_nrmse_B[valid_mask].cpu().numpy()
    valid_rmse = all_rmse[valid_mask].cpu().numpy()
    valid_vrmse_all = all_vrmse_all[valid_mask].cpu().numpy()
    valid_nrmse_all = all_nrmse_all[valid_mask].cpu().numpy()

    n = len(valid_pde)
    if n > 0:
        return {
            'pde': float(np.mean(valid_pde)),
            'rmse_A': float(np.mean(valid_rmse_A)),
            'rmse_B': float(np.mean(valid_rmse_B)),
            'vrmse_A': float(np.mean(valid_vrmse_A)),
            'vrmse_B': float(np.mean(valid_vrmse_B)),
            'nrmse_A': float(np.mean(valid_nrmse_A)),
            'nrmse_B': float(np.mean(valid_nrmse_B)),
            'rmse': float(np.mean(valid_rmse)),
            'vrmse_all': float(np.mean(valid_vrmse_all)),
            'nrmse_all': float(np.mean(valid_nrmse_all)),
            'num_batches': n,
            'per_clip': {
                'pde': valid_pde,
                'rmse_A': valid_rmse_A,
                'rmse_B': valid_rmse_B,
                'vrmse_A': valid_vrmse_A,
                'vrmse_B': valid_vrmse_B,
                'nrmse_A': valid_nrmse_A,
                'nrmse_B': valid_nrmse_B,
                'rmse': valid_rmse,
                'vrmse_all': valid_vrmse_all,
                'nrmse_all': valid_nrmse_all,
            },
        }
    return {'pde': 0, 'rmse_A': 0, 'rmse_B': 0,
            'vrmse_A': 0, 'vrmse_B': 0,
            'nrmse_A': 0, 'nrmse_B': 0,
            'rmse': 0, 'vrmse_all': 0, 'nrmse_all': 0,
            'num_batches': 0, 'per_clip': {}}


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

    # Diversify temporal starting points
    n_vis = len(sample_indices)
    if clips_per_sample > 1 and n_vis > 1:
        clip_offsets = np.linspace(0, clips_per_sample - 1, n_vis, dtype=int)
    else:
        clip_offsets = np.zeros(n_vis, dtype=int)

    print(f"Visualizing {n_vis} samples...")

    results = []
    all_rmse_A = []
    all_rmse_B = []
    all_vrmse_A = []
    all_vrmse_B = []
    all_nrmse_A = []
    all_nrmse_B = []
    all_vrmse_all = []
    all_nrmse_all = []
    all_pde_loss = []

    for i, sample_idx in enumerate(sample_indices):
        idx = sample_idx * clips_per_sample + clip_offsets[i]
        idx = min(idx, total_clips - 1)

        sample = dataset[idx]
        batch = finetune_collate_fn([sample])
        start_t = dataset.clips[idx][2]

        data = batch['data'].to(device=device, dtype=torch.float32)
        input_data = data[:, :t_input]
        target = data[:, 1:t_input + 1]

        output_norm, mean, std = model(input_data, return_normalized=True)
        output = output_norm * std + mean

        pde_loss = compute_pde_loss_from_output(output, input_data, pde_loss_fn).item()

        out_A = output[..., CH_A].float()
        tgt_A = target[..., CH_A].float()
        out_B = output[..., CH_B].float()
        tgt_B = target[..., CH_B].float()
        rmse_A = torch.sqrt(torch.mean((out_A - tgt_A) ** 2) + 1e-8).item()
        rmse_B = torch.sqrt(torch.mean((out_B - tgt_B) ** 2) + 1e-8).item()

        vrmse_A = _vrmse_torch(tgt_A, out_A).item()
        vrmse_B = _vrmse_torch(tgt_B, out_B).item()

        nrmse_A = _nrmse_torch(tgt_A, out_A).item()
        nrmse_B = _nrmse_torch(tgt_B, out_B).item()

        # Combined vrmse/nrmse: mean of per-channel values
        vrmse_all = (vrmse_A + vrmse_B) / 2
        nrmse_all = (nrmse_A + nrmse_B) / 2

        # Last timestep for plotting
        gt_A = target[0, -1, :, :, CH_A].float().cpu().numpy()
        gt_B = target[0, -1, :, :, CH_B].float().cpu().numpy()
        pred_A = output[0, -1, :, :, CH_A].float().cpu().numpy()
        pred_B = output[0, -1, :, :, CH_B].float().cpu().numpy()

        results.append({
            'gt_A': gt_A, 'gt_B': gt_B,
            'pred_A': pred_A, 'pred_B': pred_B,
            'sample_idx': sample_idx, 'start_t': start_t,
        })

        all_rmse_A.append(rmse_A)
        all_rmse_B.append(rmse_B)
        all_vrmse_A.append(vrmse_A)
        all_vrmse_B.append(vrmse_B)
        all_nrmse_A.append(nrmse_A)
        all_nrmse_B.append(nrmse_B)
        all_vrmse_all.append(vrmse_all)
        all_nrmse_all.append(nrmse_all)
        all_pde_loss.append(pde_loss)

        print(f"  Sample {sample_idx}: t_start={start_t}, "
              f"RMSE_A={rmse_A:.6f}, RMSE_B={rmse_B:.6f}, "
              f"VRMSE_A={vrmse_A:.6f}, VRMSE_B={vrmse_B:.6f}, "
              f"nRMSE_A={nrmse_A:.6f}, nRMSE_B={nrmse_B:.6f}, PDE={pde_loss:.6f}")

    output_filename = "visualization_gray_scott_lora.png"
    print("Plotting visualization...")
    plot_results(results, str(output_dir / output_filename))

    print(f"\n{'='*60}")
    print("Visualization Complete (Gray-Scott LoRA)")
    print(f"{'='*60}")
    print(f"Output: {output_dir / output_filename}")
    print(f"\nAverage Metrics ({len(results)} samples):")
    print(f"  - RMSE (A):  {np.mean(all_rmse_A):.6f}")
    print(f"  - RMSE (B):  {np.mean(all_rmse_B):.6f}")
    print(f"  - VRMSE (A): {np.mean(all_vrmse_A):.6f}")
    print(f"  - VRMSE (B): {np.mean(all_vrmse_B):.6f}")
    print(f"  - nRMSE (A): {np.mean(all_nrmse_A):.6f}")
    print(f"  - nRMSE (B): {np.mean(all_nrmse_B):.6f}")
    print(f"  - VRMSE (all): {np.mean(all_vrmse_all):.6f}")
    print(f"  - nRMSE (all): {np.mean(all_nrmse_all):.6f}")
    print(f"  - PDE Loss:  {np.mean(all_pde_loss):.6f}")
    print(f"{'='*60}")


def plot_results(results: list, save_path: str):
    """Plot GT vs Prediction vs Error for A and B."""
    n_samples = len(results)
    fig, axes = plt.subplots(n_samples, 6, figsize=(24, 4 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for row, res in enumerate(results):
        gt_A = res['gt_A']
        pred_A = res['pred_A']
        err_A = pred_A - gt_A
        gt_B = res['gt_B']
        pred_B = res['pred_B']
        err_B = pred_B - gt_B
        sample_idx = res['sample_idx']
        start_t = res.get('start_t', '?')

        # A channel
        vmin_A = min(gt_A.min(), pred_A.min())
        vmax_A = max(gt_A.max(), pred_A.max())

        im0 = axes[row, 0].imshow(gt_A, cmap='viridis',
                                    vmin=vmin_A, vmax=vmax_A)
        axes[row, 0].set_title('GT (A)', fontsize=11)
        axes[row, 0].set_ylabel(f'Sample {sample_idx}\nt0={start_t}', fontsize=10)
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)

        im1 = axes[row, 1].imshow(pred_A, cmap='viridis',
                                    vmin=vmin_A, vmax=vmax_A)
        rmse_A = np.sqrt(np.mean((pred_A - gt_A)**2))
        vrmse_A = _vrmse_np(gt_A, pred_A)
        axes[row, 1].set_title(f'Pred A (RMSE={rmse_A:.4f} VRMSE={vrmse_A:.4f})', fontsize=11)
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)

        err_A_max = max(np.percentile(np.abs(err_A), 95), 1e-6)
        im2 = axes[row, 2].imshow(err_A, cmap='RdBu_r',
                                    vmin=-err_A_max, vmax=err_A_max)
        axes[row, 2].set_title(f'Error A (MAE={np.mean(np.abs(err_A)):.4f})', fontsize=11)
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)

        # B channel
        vmin_B = min(gt_B.min(), pred_B.min())
        vmax_B = max(gt_B.max(), pred_B.max())

        im3 = axes[row, 3].imshow(gt_B, cmap='viridis',
                                    vmin=vmin_B, vmax=vmax_B)
        axes[row, 3].set_title('GT (B)', fontsize=11)
        plt.colorbar(im3, ax=axes[row, 3], fraction=0.046, pad=0.04)

        im4 = axes[row, 4].imshow(pred_B, cmap='viridis',
                                    vmin=vmin_B, vmax=vmax_B)
        rmse_B = np.sqrt(np.mean((pred_B - gt_B)**2))
        vrmse_B = _vrmse_np(gt_B, pred_B)
        axes[row, 4].set_title(f'Pred B (RMSE={rmse_B:.4f} VRMSE={vrmse_B:.4f})', fontsize=11)
        plt.colorbar(im4, ax=axes[row, 4], fraction=0.046, pad=0.04)

        err_B_max = max(np.percentile(np.abs(err_B), 95), 1e-6)
        im5 = axes[row, 5].imshow(err_B, cmap='RdBu_r',
                                    vmin=-err_B_max, vmax=err_B_max)
        axes[row, 5].set_title(f'Error B (MAE={np.mean(np.abs(err_B)):.4f})', fontsize=11)
        plt.colorbar(im5, ax=axes[row, 5], fraction=0.046, pad=0.04)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_loss_distribution(losses: dict, save_path: str):
    """Histogram of per-clip losses from scan_all."""
    n = len(losses)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, vals) in zip(axes, losses.items()):
        ax.hist(vals, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(vals), color='red', linestyle='--',
                   label=f'mean={np.mean(vals):.4e}')
        ax.set_title(name)
        ax.set_xlabel('Loss')
        ax.set_ylabel('Count')
        ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved loss distribution: {save_path}")


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
        print(f"Gray-Scott LoRA V3 Evaluation")
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
        vector_dim=config['dataset'].get('vector_dim', 0),
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
            print(f"  PDE Loss:  {results['pde']:.6f}")
            print(f"  RMSE (A):  {results['rmse_A']:.6f}")
            print(f"  RMSE (B):  {results['rmse_B']:.6f}")
            print(f"  VRMSE (A): {results['vrmse_A']:.6f}")
            print(f"  VRMSE (B): {results['vrmse_B']:.6f}")
            print(f"  nRMSE (A): {results['nrmse_A']:.6f}")
            print(f"  nRMSE (B): {results['nrmse_B']:.6f}")
            print(f"  RMSE:      {results['rmse']:.6f}")
            print(f"  VRMSE (all): {results['vrmse_all']:.6f}")
            print(f"  nRMSE (all): {results['nrmse_all']:.6f}")
            print(f"{'='*60}")

            if results.get('per_clip'):
                ckpt_dir = Path(args.checkpoint).parent
                plot_loss_distribution(
                    results['per_clip'],
                    str(ckpt_dir / "loss_distribution_gray_scott.png"),
                )

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
