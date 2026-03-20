"""
LoRA V4 Model Visualization for 3D Heat Equation.

Multi-GPU evaluation using Accelerate (same pattern as other visualize scripts).
Since data is 3D (64^3), plots central slices (XY, XZ, YZ) of temperature field.

Usage:
    # Multi-GPU full evaluation
    torchrun --nproc_per_node=8 tools/visualize_heat_3d_lora.py \
        --config configs/finetune_heat_3d_v4.yaml \
        --checkpoint checkpoints_heat_3d_lora_v4/best_lora.pt --scan_all

    # Single-GPU visualization (plot only)
    python tools/visualize_heat_3d_lora.py \
        --config configs/finetune_heat_3d_v4.yaml \
        --checkpoint checkpoints_heat_3d_lora_v4/best_lora.pt \
        --output_dir ./vis_results/heat_3d_vis
"""

import os
import sys
import warnings

# Suppress flex_attention warnings (4D NA uses flex backend without torch.compile)
warnings.filterwarnings('ignore', message='.*flex_attention called without torch.compile.*')
warnings.filterwarnings('ignore', message='.*return_lse is deprecated.*')

import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional

from torch.utils.data import DataLoader
from accelerate import Accelerator

from finetune.dataset_finetune import (
    FinetuneDataset, FinetuneSampler, finetune_collate_fn,
)
from finetune.model_lora_v3 import PDELoRAModelV3, load_lora_checkpoint
from finetune.pde_loss_verified import Heat3DPDELoss

# Heat 3D channel index in 18-channel layout
CH_TEMP = 17  # scalar[14] = temperature


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
    parser = argparse.ArgumentParser(description="LoRA V4 Visualization for 3D Heat Equation")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='./vis_results/heat_3d_vis', help='Output directory')
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
    """Load LoRA V4 model."""
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


def create_pde_loss_fn(config: dict) -> Heat3DPDELoss:
    physics = config.get('physics', {})
    return Heat3DPDELoss(
        nx=physics.get('nx', 64),
        ny=physics.get('ny', 64),
        nz=physics.get('nz', 64),
        Lx=physics.get('Lx', 2 * np.pi),
        Ly=physics.get('Ly', 2 * np.pi),
        Lz=physics.get('Lz', 2 * np.pi),
        dt=physics.get('dt', 0.01),
        alpha=physics.get('alpha_default', 0.01),
    )


def compute_pde_loss_from_output(
    output: torch.Tensor,
    input_data: torch.Tensor,
    pde_loss_fn: Heat3DPDELoss,
    alpha: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute Heat equation PDE residual loss."""
    with torch.autocast(device_type='cuda', enabled=False):
        # input_data: [B, T_in, D, H, W, 18], output: [B, T_in, D, H, W, 18]
        t0_temp = input_data[:, 0:1, :, :, :, CH_TEMP].float()
        out_temp = output[:, :, :, :, :, CH_TEMP].float()
        temp = torch.cat([t0_temp, out_temp], dim=1)  # [B, T_in+1, D, H, W]
        total_loss, _ = pde_loss_fn(temp, alpha=alpha)
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
    total_rmse_temp = torch.zeros(1, device=accelerator.device)
    total_rmse = torch.zeros(1, device=accelerator.device)
    total_vrmse_temp = torch.zeros(1, device=accelerator.device)
    total_vrmse_all = torch.zeros(1, device=accelerator.device)
    total_nrmse_all = torch.zeros(1, device=accelerator.device)
    num_batches = torch.zeros(1, device=accelerator.device)

    for batch in val_loader:
        data = batch['data'].to(device=accelerator.device, dtype=torch.float32)
        channel_mask = batch['channel_mask'].to(device=accelerator.device)
        alpha = batch.get('nu')
        if alpha is not None:
            alpha = alpha.to(device=accelerator.device, dtype=torch.float32)

        input_data = data[:, :t_input]
        target_data = data[:, 1:t_input + 1]

        output_norm, mean, std = model(input_data, return_normalized=True)
        output = output_norm * std + mean

        # PDE loss
        pde_loss = compute_pde_loss_from_output(output, input_data, pde_loss_fn, alpha=alpha)

        # Temperature RMSE
        out_temp = output[..., CH_TEMP].float()
        tgt_temp = target_data[..., CH_TEMP].float()
        rmse_temp = torch.sqrt(torch.mean((out_temp - tgt_temp) ** 2) + 1e-8)
        vrmse_temp = _vrmse_torch(tgt_temp, out_temp)

        # Overall RMSE (valid channels)
        valid_ch = (
            torch.where(channel_mask[0] > 0)[0]
            if channel_mask.dim() > 1
            else torch.where(channel_mask > 0)[0]
        )
        mse = torch.mean((output[..., valid_ch] - target_data[..., valid_ch]) ** 2)
        rmse = torch.sqrt(mse + 1e-8)

        # Combined vrmse/nrmse as mean of per-channel values (single channel)
        nrmse_temp = _nrmse_torch(tgt_temp, out_temp)
        vrmse_all = vrmse_temp
        nrmse_all = nrmse_temp

        total_pde += pde_loss.detach()
        total_rmse_temp += rmse_temp.detach()
        total_rmse += rmse.detach()
        total_vrmse_temp += vrmse_temp.detach()
        total_vrmse_all += vrmse_all.detach()
        total_nrmse_all += nrmse_all.detach()
        num_batches += 1

    accelerator.wait_for_everyone()
    total_pde = accelerator.reduce(total_pde, reduction='sum')
    total_rmse_temp = accelerator.reduce(total_rmse_temp, reduction='sum')
    total_rmse = accelerator.reduce(total_rmse, reduction='sum')
    total_vrmse_temp = accelerator.reduce(total_vrmse_temp, reduction='sum')
    total_vrmse_all = accelerator.reduce(total_vrmse_all, reduction='sum')
    total_nrmse_all = accelerator.reduce(total_nrmse_all, reduction='sum')
    num_batches = accelerator.reduce(num_batches, reduction='sum')
    accelerator.wait_for_everyone()

    n = num_batches.item()
    if n > 0:
        return {
            'pde': (total_pde / n).item(),
            'rmse_temp': (total_rmse_temp / n).item(),
            'rmse': (total_rmse / n).item(),
            'vrmse_temp': (total_vrmse_temp / n).item(),
            'vrmse_all': (total_vrmse_all / n).item(),
            'nrmse_all': (total_nrmse_all / n).item(),
            'num_batches': int(n),
        }
    return {'pde': 0, 'rmse_temp': 0, 'rmse': 0, 'vrmse_temp': 0,
            'vrmse_all': 0, 'nrmse_all': 0, 'num_batches': 0}


# ============================================================
# Visualization (main process only)
# ============================================================

def plot_3d_slices(results: list, save_path: str):
    """
    Plot central slices of 3D temperature field.
    For each sample: 3 rows (XY, XZ, YZ) x 3 cols (GT, Pred, Error).
    """
    n_samples = len(results)
    fig, axes = plt.subplots(n_samples * 3, 3, figsize=(14, 4 * n_samples * 3))

    if n_samples * 3 == 1:
        axes = axes.reshape(1, -1)

    for s_idx, res in enumerate(results):
        gt = res['gt_temp']        # [D, H, W]
        pred = res['pred_temp']    # [D, H, W]
        sample_idx = res['sample_idx']
        alpha_val = res.get('alpha', '?')
        start_t = res.get('start_t', '?')

        D, H, W = gt.shape
        # For sin(x)sin(y)sin(z): D//2=π is zero-node; D//4=π/2 is max
        qd, qh, qw = D // 4, H // 4, W // 4

        slices = [
            ('XY (z={})'.format(qw), gt[:, :, qw], pred[:, :, qw]),
            ('XZ (y={})'.format(qh), gt[:, qh, :], pred[:, qh, :]),
            ('YZ (x={})'.format(qd), gt[qd, :, :], pred[qd, :, :]),
        ]

        for sl_idx, (slice_name, gt_sl, pred_sl) in enumerate(slices):
            row = s_idx * 3 + sl_idx
            err_sl = pred_sl - gt_sl

            vmin = min(gt_sl.min(), pred_sl.min())
            vmax = max(gt_sl.max(), pred_sl.max())

            # GT
            im0 = axes[row, 0].imshow(gt_sl, origin='lower', cmap='inferno',
                                        vmin=vmin, vmax=vmax)
            axes[row, 0].set_title(f'GT {slice_name}', fontsize=10)
            if sl_idx == 0:
                axes[row, 0].set_ylabel(
                    f'Sample {sample_idx}\nt0={start_t}\n' + r'$\alpha$' + f'={alpha_val:.4f}',
                    fontsize=9,
                )
            plt.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)

            # Pred
            rmse_sl = np.sqrt(np.mean((pred_sl - gt_sl) ** 2))
            vrmse_sl = _vrmse_np(gt_sl, pred_sl)
            im1 = axes[row, 1].imshow(pred_sl, origin='lower', cmap='inferno',
                                        vmin=vmin, vmax=vmax)
            axes[row, 1].set_title(f'Pred {slice_name} (RMSE={rmse_sl:.4f} VRMSE={vrmse_sl:.4f})', fontsize=10)
            plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)

            # Error
            err_max = max(np.percentile(np.abs(err_sl), 95), 1e-6)
            im2 = axes[row, 2].imshow(err_sl, origin='lower', cmap='RdBu_r',
                                        vmin=-err_max, vmax=err_max)
            axes[row, 2].set_title(f'Error (MAE={np.mean(np.abs(err_sl)):.4f})', fontsize=10)
            plt.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


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
    all_rmse_temp = []
    all_vrmse_temp = []
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
        alpha = batch.get('nu')
        if alpha is not None:
            alpha = alpha.to(device=device, dtype=torch.float32)

        input_data = data[:, :t_input]
        target = data[:, 1:t_input + 1]

        output_norm, mean, std = model(input_data, return_normalized=True)
        output = output_norm * std + mean

        pde_loss = compute_pde_loss_from_output(
            output, input_data, pde_loss_fn, alpha=alpha,
        ).item()

        out_temp = output[..., CH_TEMP].float()
        tgt_temp = target[..., CH_TEMP].float()
        rmse_temp = torch.sqrt(torch.mean((out_temp - tgt_temp) ** 2) + 1e-8).item()
        vrmse_temp = _vrmse_torch(tgt_temp, out_temp).item()

        # Combined vrmse/nrmse as mean of per-channel values (single channel)
        nrmse_temp_vis = _nrmse_torch(tgt_temp, out_temp).item()
        vrmse_all_vis = vrmse_temp
        nrmse_all_vis = nrmse_temp_vis

        # Last timestep for plotting: [D, H, W]
        gt_temp = target[0, -1, :, :, :, CH_TEMP].float().cpu().numpy()
        pred_temp = output[0, -1, :, :, :, CH_TEMP].float().cpu().numpy()

        alpha_val = alpha[0].item() if alpha is not None else config.get('physics', {}).get('alpha_default', 0.01)

        results.append({
            'gt_temp': gt_temp, 'pred_temp': pred_temp,
            'sample_idx': sample_idx, 'start_t': start_t,
            'alpha': alpha_val,
        })

        all_rmse_temp.append(rmse_temp)
        all_vrmse_temp.append(vrmse_temp)
        all_vrmse_all.append(vrmse_all_vis)
        all_nrmse_all.append(nrmse_all_vis)
        all_pde_loss.append(pde_loss)

        print(f"  Sample {sample_idx}: t_start={start_t}, alpha={alpha_val:.5f}, "
              f"RMSE_T={rmse_temp:.6f}, VRMSE_T={vrmse_temp:.6f}, PDE={pde_loss:.6f}")

    output_filename = "visualization_heat_3d_lora.png"
    print("Plotting visualization...")
    plot_3d_slices(results, str(output_dir / output_filename))

    print(f"\n{'='*60}")
    print("Visualization Complete (Heat 3D LoRA)")
    print(f"{'='*60}")
    print(f"Output: {output_dir / output_filename}")
    print(f"\nAverage Metrics ({len(results)} samples):")
    print(f"  - RMSE (Temperature):  {np.mean(all_rmse_temp):.6f}")
    print(f"  - VRMSE (Temperature): {np.mean(all_vrmse_temp):.6f}")
    print(f"  - VRMSE (all):         {np.mean(all_vrmse_all):.6f}")
    print(f"  - nRMSE (all):         {np.mean(all_nrmse_all):.6f}")
    print(f"  - PDE Loss:            {np.mean(all_pde_loss):.6f}")
    print(f"{'='*60}")


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    config = load_config(args.config)

    t_input = args.t_input or config.get('dataset', {}).get('t_input', 8)
    temporal_length = t_input + 1
    batch_size = args.batch_size or config.get('dataloader', {}).get('batch_size', 2)

    accelerator = Accelerator()
    is_main = accelerator.is_main_process

    if is_main:
        print(f"{'='*60}")
        print(f"Heat 3D LoRA V4 Evaluation")
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
            print(f"  PDE Loss:         {results['pde']:.6f}")
            print(f"  RMSE (Temp):      {results['rmse_temp']:.6f}")
            print(f"  VRMSE (Temp):     {results['vrmse_temp']:.6f}")
            print(f"  VRMSE (all):      {results['vrmse_all']:.6f}")
            print(f"  nRMSE (all):      {results['nrmse_all']:.6f}")
            print(f"  RMSE (overall):   {results['rmse']:.6f}")
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
