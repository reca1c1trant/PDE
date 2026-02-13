"""
LoRA V2 Model Visualization for 2D Burgers Equation.

Multi-GPU evaluation using Accelerate (same pattern as training).

Usage:
    # Multi-GPU full evaluation
    torchrun --nproc_per_node=8 tools/visualize_burgers_lora.py \
        --config configs/finetune_burgers.yaml \
        --checkpoint checkpoints_burgers_lora/best_lora.pt --scan_all

    # Single-GPU visualization (plot only)
    python tools/visualize_burgers_lora.py \
        --config configs/finetune_burgers.yaml \
        --checkpoint checkpoints_burgers_lora/best_lora.pt --output_dir ./burgers_vis_results
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
from finetune.model_lora_v2 import PDELoRAModelV2, load_lora_checkpoint
from finetune.pde_loss import burgers_pde_loss


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA V2 Visualization for Burgers")
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
    """Load LoRA V2 model (CPU, will be moved by accelerator)."""
    pretrained_path = config.get('model', {}).get('pretrained_path')
    if pretrained_path is None:
        raise ValueError("Config must specify 'model.pretrained_path'")

    freeze_encoder = config.get('model', {}).get('freeze_encoder', False)
    freeze_decoder = config.get('model', {}).get('freeze_decoder', False)

    model = PDELoRAModelV2(
        config=config,
        pretrained_path=pretrained_path,
        freeze_encoder=freeze_encoder,
        freeze_decoder=freeze_decoder,
    )

    checkpoint = load_lora_checkpoint(model, checkpoint_path)

    if is_main:
        if 'metrics' in checkpoint:
            print(f"  Checkpoint metrics: {checkpoint['metrics']}")
        if 'global_step' in checkpoint:
            print(f"  Global step: {checkpoint['global_step']}")

    model = model.float()
    return model


def compute_pde_loss_from_output(
    output: torch.Tensor,
    input_data: torch.Tensor,
    batch: dict,
    config: dict,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute PDE residual loss. Returns (pde_loss, loss_u, loss_v)."""
    with torch.autocast(device_type='cuda', enabled=False):
        t0_frame = input_data[:, 0:1, ..., :2].float()
        output_uv = output[..., :2].float()
        pred_with_t0 = torch.cat([t0_frame, output_uv], dim=1)

        nu = batch['nu'].to(device).float()
        nu_mean = nu.mean().item()

        dt = config.get('physics', {}).get('dt', 1/999)
        dx = config.get('physics', {}).get('dx', 1/127)
        dy = config.get('physics', {}).get('dy', 1/127)

        pde_loss, loss_u, loss_v, _, _ = burgers_pde_loss(
            pred=pred_with_t0, nu=nu_mean, dt=dt, dx=dx, dy=dy,
        )

    return pde_loss, loss_u, loss_v


def compute_pde_residual(
    output: torch.Tensor,
    input_data: torch.Tensor,
    batch: dict,
    config: dict,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, float, float, float]:
    """Compute PDE residual (returns actual residual maps for visualization)."""
    t0_frame = input_data[:, 0:1, ..., :2].float()
    output_uv = output[..., :2].float()
    pred_with_t0 = torch.cat([t0_frame, output_uv], dim=1)

    nu = batch['nu'].to(device).float()
    nu_mean = nu.mean().item()

    dt = config.get('physics', {}).get('dt', 1/999)
    dx = config.get('physics', {}).get('dx', 1/127)
    dy = config.get('physics', {}).get('dy', 1/127)

    pde_loss, loss_u, loss_v, residual_u, residual_v = burgers_pde_loss(
        pred=pred_with_t0, nu=nu_mean, dt=dt, dx=dx, dy=dy,
    )

    return residual_u, residual_v, pde_loss.item(), loss_u.item(), loss_v.item()


# ============================================================
# scan_all: Multi-GPU distributed evaluation
# ============================================================

@torch.no_grad()
def scan_all_distributed(accelerator, model, val_loader, config, t_input):
    """
    Distributed evaluation over ALL validation clips.

    Same reduce pattern as validate() in train_burgers_lora.py.
    """
    accelerator.wait_for_everyone()
    model.eval()

    total_pde = torch.zeros(1, device=accelerator.device)
    total_rmse_u = torch.zeros(1, device=accelerator.device)
    total_rmse_v = torch.zeros(1, device=accelerator.device)
    total_rmse = torch.zeros(1, device=accelerator.device)
    num_batches = torch.zeros(1, device=accelerator.device)

    for batch in val_loader:
        data = batch['data'].to(device=accelerator.device, dtype=torch.float32)
        channel_mask = batch['channel_mask'].to(device=accelerator.device)

        input_data = data[:, :t_input]
        target_data = data[:, 1:t_input + 1]

        output_norm, mean, std = model(input_data, return_normalized=True)
        output = output_norm * std + mean

        # PDE loss
        if 'nu' in batch:
            pde_loss, _, _ = compute_pde_loss_from_output(
                output, input_data, batch, config, accelerator.device,
            )
        else:
            pde_loss = torch.tensor(0.0, device=accelerator.device)

        # Per-channel RMSE
        out_u = output[..., 0].float()
        tgt_u = target_data[..., 0].float()
        out_v = output[..., 1].float()
        tgt_v = target_data[..., 1].float()

        rmse_u = torch.sqrt(torch.mean((out_u - tgt_u) ** 2) + 1e-8)
        rmse_v = torch.sqrt(torch.mean((out_v - tgt_v) ** 2) + 1e-8)

        # Overall RMSE (valid channels)
        valid_ch = (
            torch.where(channel_mask[0] > 0)[0]
            if channel_mask.dim() > 1
            else torch.where(channel_mask > 0)[0]
        )
        mse = torch.mean((output[..., valid_ch] - target_data[..., valid_ch]) ** 2)
        rmse = torch.sqrt(mse + 1e-8)

        total_pde += pde_loss.detach()
        total_rmse_u += rmse_u.detach()
        total_rmse_v += rmse_v.detach()
        total_rmse += rmse.detach()
        num_batches += 1

    # Reduce across all ranks
    accelerator.wait_for_everyone()
    total_pde = accelerator.reduce(total_pde, reduction='sum')
    total_rmse_u = accelerator.reduce(total_rmse_u, reduction='sum')
    total_rmse_v = accelerator.reduce(total_rmse_v, reduction='sum')
    total_rmse = accelerator.reduce(total_rmse, reduction='sum')
    num_batches = accelerator.reduce(num_batches, reduction='sum')
    accelerator.wait_for_everyone()

    n = num_batches.item()
    if n > 0:
        avg_pde = (total_pde / n).item()
        avg_rmse_u = (total_rmse_u / n).item()
        avg_rmse_v = (total_rmse_v / n).item()
        avg_rmse = (total_rmse / n).item()
    else:
        avg_pde = avg_rmse_u = avg_rmse_v = avg_rmse = 0.0

    return {
        'pde': avg_pde,
        'rmse_u': avg_rmse_u,
        'rmse_v': avg_rmse_v,
        'rmse': avg_rmse,
        'num_batches': int(n),
    }


# ============================================================
# Visualization (main process only)
# ============================================================

@torch.no_grad()
def run_visualization(model, dataset, config, device, t_input, num_samples, seed, output_dir):
    """Run visualization on main process only."""
    model.eval()

    total_clips = len(dataset)
    n_val_samples = len(dataset.sample_indices)
    clips_per_sample = total_clips // n_val_samples if n_val_samples > 0 else 1

    np.random.seed(seed)
    # Filter samples with nu < 0.01
    valid_samples = []
    for s in range(n_val_samples):
        idx = s * clips_per_sample
        nu_val = dataset[idx]['nu'].item()
        if nu_val < 0.01:
            valid_samples.append(s)

    if len(valid_samples) == 0:
        print("No validation samples with nu < 0.01!")
        return

    print(f"Found {len(valid_samples)} samples with nu < 0.01")
    sample_indices = np.random.choice(
        valid_samples, min(num_samples, len(valid_samples)), replace=False,
    )

    # Diversify temporal starting points: early / mid / late
    n_vis = len(sample_indices)
    if clips_per_sample > 1 and n_vis > 1:
        clip_offsets = np.linspace(0, clips_per_sample - 1, n_vis, dtype=int)
    else:
        clip_offsets = np.zeros(n_vis, dtype=int)

    print(f"Visualizing {n_vis} samples with diversified timesteps "
          f"(clip offsets: {clip_offsets.tolist()})...")

    results = []
    all_rmse_u = []
    all_rmse_v = []
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

        nu = batch['nu'][0].item()

        residual_u, residual_v, pde_loss, _, _ = compute_pde_residual(
            output, input_data, batch, config, device,
        )

        out_u = output[..., 0].float()
        tgt_u = target[..., 0].float()
        out_v = output[..., 1].float()
        tgt_v = target[..., 1].float()
        rmse_u = torch.sqrt(torch.mean((out_u - tgt_u) ** 2) + 1e-8).item()
        rmse_v = torch.sqrt(torch.mean((out_v - tgt_v) ** 2) + 1e-8).item()

        # Last timestep for plotting
        gt_u = target[0, -1, :, :, 0].float().cpu().numpy()
        gt_v = target[0, -1, :, :, 1].float().cpu().numpy()
        pred_u = output[0, -1, :, :, 0].float().cpu().numpy()
        pred_v = output[0, -1, :, :, 1].float().cpu().numpy()
        res_u_last = residual_u[0, -1].cpu().numpy()
        res_v_last = residual_v[0, -1].cpu().numpy()

        results.append({
            'gt_u': gt_u, 'gt_v': gt_v,
            'pred_u': pred_u, 'pred_v': pred_v,
            'residual_u': res_u_last, 'residual_v': res_v_last,
            'nu': nu, 'sample_idx': sample_idx, 'start_t': start_t,
        })

        all_rmse_u.append(rmse_u)
        all_rmse_v.append(rmse_v)
        all_pde_loss.append(pde_loss)

        last_rmse_u = np.sqrt(np.mean((pred_u - gt_u)**2))
        last_rmse_v = np.sqrt(np.mean((pred_v - gt_v)**2))
        print(f"  Sample {sample_idx}: nu={nu:.3f}, t_start={start_t}, "
              f"RMSE_u={last_rmse_u:.4f}, RMSE_v={last_rmse_v:.4f}, PDE={pde_loss:.2f}")

    output_filename = "visualization_burgers_lora.png"
    print("Plotting visualization...")
    plot_results(results, str(output_dir / output_filename))

    print("\n" + "=" * 60)
    print("Visualization Complete (LoRA)")
    print("=" * 60)
    print(f"Output: {output_dir / output_filename}")
    print(f"\nAverage Metrics ({len(results)} samples):")
    print(f"  - RMSE (u): {np.mean(all_rmse_u):.6f}")
    print(f"  - RMSE (v): {np.mean(all_rmse_v):.6f}")
    print(f"  - PDE Loss: {np.mean(all_pde_loss):.2f}")
    print("=" * 60)


def plot_results(results: list, save_path: str):
    """Plot GT vs Prediction vs PDE Residual for u and v."""
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
        start_t = res.get('start_t', '?')

        # u channel
        vmin_u = min(gt_u.min(), pred_u.min())
        vmax_u = max(gt_u.max(), pred_u.max())

        im0 = axes[row, 0].imshow(gt_u, origin='lower', extent=extent, cmap='jet',
                                    vmin=vmin_u, vmax=vmax_u)
        axes[row, 0].set_title('GT (u)', fontsize=11)
        axes[row, 0].set_ylabel(f'Sample {sample_idx}\nnu={nu:.3f}, t0={start_t}', fontsize=10)
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)

        im1 = axes[row, 1].imshow(pred_u, origin='lower', extent=extent, cmap='jet',
                                    vmin=vmin_u, vmax=vmax_u)
        rmse_u = np.sqrt(np.mean((pred_u - gt_u)**2))
        axes[row, 1].set_title(f'Pred u (RMSE={rmse_u:.4f})', fontsize=11)
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)

        res_u_max = np.percentile(np.abs(res_u), 95)
        res_u_max = res_u_max if res_u_max > 0 else 1.0
        im2 = axes[row, 2].imshow(res_u, origin='lower', extent=extent, cmap='RdBu_r',
                                    vmin=-res_u_max, vmax=res_u_max)
        mse_u = np.mean(res_u**2)
        axes[row, 2].set_title(f'Residual u (MSE={mse_u:.2f})', fontsize=11)
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)

        # v channel
        vmin_v = min(gt_v.min(), pred_v.min())
        vmax_v = max(gt_v.max(), pred_v.max())

        im3 = axes[row, 3].imshow(gt_v, origin='lower', extent=extent, cmap='jet',
                                    vmin=vmin_v, vmax=vmax_v)
        axes[row, 3].set_title('GT (v)', fontsize=11)
        plt.colorbar(im3, ax=axes[row, 3], fraction=0.046, pad=0.04)

        im4 = axes[row, 4].imshow(pred_v, origin='lower', extent=extent, cmap='jet',
                                    vmin=vmin_v, vmax=vmax_v)
        rmse_v = np.sqrt(np.mean((pred_v - gt_v)**2))
        axes[row, 4].set_title(f'Pred v (RMSE={rmse_v:.4f})', fontsize=11)
        plt.colorbar(im4, ax=axes[row, 4], fraction=0.046, pad=0.04)

        res_v_max = np.percentile(np.abs(res_v), 95)
        res_v_max = res_v_max if res_v_max > 0 else 1.0
        im5 = axes[row, 5].imshow(res_v, origin='lower', extent=extent, cmap='RdBu_r',
                                    vmin=-res_v_max, vmax=res_v_max)
        mse_v = np.mean(res_v**2)
        axes[row, 5].set_title(f'Residual v (MSE={mse_v:.2f})', fontsize=11)
        plt.colorbar(im5, ax=axes[row, 5], fraction=0.046, pad=0.04)

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

    t_input = args.t_input or config.get('training', {}).get('t_input',
              config.get('dataset', {}).get('t_input', 8))
    temporal_length = t_input + 1
    batch_size = args.batch_size or config.get('dataloader', {}).get('batch_size', 4)

    # ---- Accelerator (handles multi-GPU) ----
    accelerator = Accelerator()
    is_main = accelerator.is_main_process

    if is_main:
        print(f"{'='*60}")
        print(f"Burgers2D LoRA V2 Evaluation")
        print(f"{'='*60}")
        print(f"  Devices: {accelerator.num_processes}")
        print(f"  t_input: {t_input}")
        print(f"  batch_size: {batch_size}")
        print(f"  mode: {'scan_all' if args.scan_all else 'visualize'}")
        print(f"{'='*60}")

    # ---- Model ----
    model = load_model(config, args.checkpoint, is_main=is_main)

    # ---- Dataset ----
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

    total_clips = len(val_dataset)
    n_val_samples = len(val_dataset.sample_indices)

    if is_main:
        print(f"  Val samples: {n_val_samples}, total clips: {total_clips}")

    # ---- scan_all: distributed evaluation ----
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
            print(f"\nScanning all {total_clips} clips "
                  f"across {accelerator.num_processes} GPUs...")

        results = scan_all_distributed(
            accelerator, model, val_loader, config, t_input,
        )

        if is_main:
            print(f"\n{'='*60}")
            print(f"Results ({results['num_batches']} batches across all ranks):")
            print(f"  PDE Loss:  {results['pde']:.6f}")
            print(f"  RMSE (u):  {results['rmse_u']:.6f}")
            print(f"  RMSE (v):  {results['rmse_v']:.6f}")
            print(f"  RMSE:      {results['rmse']:.6f}")
            print(f"{'='*60}")

    # ---- Visualization (main process only) ----
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