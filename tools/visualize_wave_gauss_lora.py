"""
LoRA V3 Model Visualization for Wave-Gauss (acoustic wave equation).

Multi-GPU evaluation using Accelerate (same pattern as training).

Channels: displacement u (scalar[0] = ch3), wave speed c (scalar[1] = ch4).
No vector fields (vector_dim=0).

Usage:
    # Multi-GPU full evaluation
    torchrun --nproc_per_node=8 tools/visualize_wave_gauss_lora.py \
        --config configs/finetune_wave_gauss_v3.yaml \
        --checkpoint checkpoints_wave_gauss_lora_v3/best_lora.pt --scan_all

    # Single-GPU visualization (plot only)
    python tools/visualize_wave_gauss_lora.py \
        --config configs/finetune_wave_gauss_v3.yaml \
        --checkpoint checkpoints_wave_gauss_lora_v3/best_lora.pt --output_dir ./wave_gauss_vis
"""

import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from torch.utils.data import DataLoader
from accelerate import Accelerator

from finetune.dataset_finetune import (
    FinetuneDataset, FinetuneSampler, finetune_collate_fn,
)
from finetune.model_lora_v3 import PDELoRAModelV3, load_lora_checkpoint
from finetune.pde_loss_verified import WaveGaussPDELoss

# Wave-Gauss channel indices in 18-channel layout
# vector_dim=0, so vector channels are all zeros
# scalar[0] = displacement u, scalar[1] = wave speed c
CH_U = 3   # scalar[0] = displacement
CH_C = 4   # scalar[1] = wave speed (time-independent)


def _vrmse_np(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-8) -> float:
    """Variance-normalized RMSE."""
    mse = np.mean((pred - gt) ** 2)
    var = np.mean((gt - gt.mean()) ** 2)
    return float(np.sqrt(mse / (var + eps)))


def _nrmse_np(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-8) -> float:
    """nRMSE: sqrt(MSE(pred, gt) / MSE(0, gt))."""
    mse_pred = np.mean((pred - gt) ** 2)
    mse_zero = np.mean(gt ** 2)
    return float(np.sqrt(mse_pred / (mse_zero + eps)))


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
    parser = argparse.ArgumentParser(description="LoRA V3 Visualization for Wave-Gauss")
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
    """Load LoRA V3 model, using checkpoint's saved config for model architecture."""
    ckpt_probe = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'config' in ckpt_probe:
        ckpt_model_cfg = ckpt_probe['config'].get('model', {})
        arch_keys = ['decoder', 'patch_smoother', 'encoder', 'intra_patch', 'na',
                     'in_channels', 'hidden_dim', 'patch_size', 'num_layers', 'num_heads',
                     'vector_channels', 'scalar_channels', 'enable_1d', 'enable_3d']
        model_cfg = dict(config.get('model', {}))
        for k in arch_keys:
            if k in ckpt_model_cfg:
                if is_main and k == 'decoder' and model_cfg.get(k) != ckpt_model_cfg[k]:
                    print(f"  [INFO] Overriding decoder config from checkpoint")
                    print(f"    YAML: {model_cfg.get(k)}")
                    print(f"    Ckpt: {ckpt_model_cfg[k]}")
                model_cfg[k] = ckpt_model_cfg[k]
        config = {**config, 'model': model_cfg}
    del ckpt_probe

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


def create_pde_loss_fn(config: dict) -> WaveGaussPDELoss:
    physics = config.get('physics', {})
    eq_scales = physics.get('eq_scales', None)
    eq_weights = physics.get('eq_weights', None)
    return WaveGaussPDELoss(
        nx=physics.get('nx', 128),
        ny=physics.get('ny', 128),
        Lx=physics.get('Lx', 1.0),
        Ly=physics.get('Ly', 1.0),
        dt=physics.get('dt', 1.0),
        eq_scales=eq_scales,
        eq_weights=eq_weights,
    )


def compute_pde_loss_from_output(
    output: torch.Tensor,
    input_data: torch.Tensor,
    pde_loss_fn: WaveGaussPDELoss,
) -> torch.Tensor:
    """Compute Wave-Gauss PDE residual loss.

    u needs all frames (t0 from input + predicted frames).
    c is time-independent, taken from input.
    """
    with torch.autocast(device_type='cuda', enabled=False):
        # t0 displacement from input
        t0_u = input_data[:, 0:1, :, :, CH_U].float()
        # Predicted displacement (all output frames)
        out_u = output[:, :, :, :, CH_U].float()
        # Full temporal sequence: [t0, t1, ..., tN]
        u = torch.cat([t0_u, out_u], dim=1)

        # Wave speed c is time-independent; take from input frame 0
        c = input_data[:, 0, :, :, CH_C].float()  # (B, H, W)

        total_loss, losses = pde_loss_fn(u, c)
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
    local_rmse_u = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_rmse_c = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_rmse = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_vrmse_u = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_vrmse_c = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_nrmse_u = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_nrmse_c = torch.full((max_batches,), float('nan'), device=accelerator.device)
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

        rmse_u = torch.sqrt(torch.mean((output[..., CH_U] - target_data[..., CH_U]) ** 2) + 1e-8)
        rmse_c = torch.sqrt(torch.mean((output[..., CH_C] - target_data[..., CH_C]) ** 2) + 1e-8)

        vrmse_u = _vrmse_torch(target_data[..., CH_U], output[..., CH_U])
        vrmse_c = _vrmse_torch(target_data[..., CH_C], output[..., CH_C])

        nrmse_u = _nrmse_torch(target_data[..., CH_U], output[..., CH_U])
        nrmse_c = _nrmse_torch(target_data[..., CH_C], output[..., CH_C])

        valid_ch = (
            torch.where(channel_mask[0] > 0)[0]
            if channel_mask.dim() > 1
            else torch.where(channel_mask > 0)[0]
        )
        mse = torch.mean((output[..., valid_ch] - target_data[..., valid_ch]) ** 2)
        rmse = torch.sqrt(mse + 1e-8)

        # Combined vrmse/nrmse as mean of per-channel values
        vrmse_all = (vrmse_u + vrmse_c) / 2
        nrmse_all = (nrmse_u + nrmse_c) / 2

        local_pde[i] = pde_loss.detach()
        local_rmse_u[i] = rmse_u.detach()
        local_rmse_c[i] = rmse_c.detach()
        local_rmse[i] = rmse.detach()
        local_vrmse_u[i] = vrmse_u.detach()
        local_vrmse_c[i] = vrmse_c.detach()
        local_nrmse_u[i] = nrmse_u.detach()
        local_nrmse_c[i] = nrmse_c.detach()
        local_vrmse_all[i] = vrmse_all.detach()
        local_nrmse_all[i] = nrmse_all.detach()

    accelerator.wait_for_everyone()
    all_pde = accelerator.gather(local_pde)
    all_rmse_u = accelerator.gather(local_rmse_u)
    all_rmse_c = accelerator.gather(local_rmse_c)
    all_rmse = accelerator.gather(local_rmse)
    all_vrmse_u = accelerator.gather(local_vrmse_u)
    all_vrmse_c = accelerator.gather(local_vrmse_c)
    all_nrmse_u = accelerator.gather(local_nrmse_u)
    all_nrmse_c = accelerator.gather(local_nrmse_c)
    all_vrmse_all = accelerator.gather(local_vrmse_all)
    all_nrmse_all = accelerator.gather(local_nrmse_all)
    accelerator.wait_for_everyone()

    valid_mask = ~torch.isnan(all_pde)
    valid_pde = all_pde[valid_mask].cpu().numpy()
    valid_rmse_u = all_rmse_u[valid_mask].cpu().numpy()
    valid_rmse_c = all_rmse_c[valid_mask].cpu().numpy()
    valid_rmse = all_rmse[valid_mask].cpu().numpy()
    valid_vrmse_u = all_vrmse_u[valid_mask].cpu().numpy()
    valid_vrmse_c = all_vrmse_c[valid_mask].cpu().numpy()
    valid_nrmse_u = all_nrmse_u[valid_mask].cpu().numpy()
    valid_nrmse_c = all_nrmse_c[valid_mask].cpu().numpy()
    valid_vrmse_all = all_vrmse_all[valid_mask].cpu().numpy()
    valid_nrmse_all = all_nrmse_all[valid_mask].cpu().numpy()

    n = len(valid_pde)
    if n > 0:
        return {
            'pde': float(np.mean(valid_pde)),
            'rmse_u': float(np.mean(valid_rmse_u)),
            'rmse_c': float(np.mean(valid_rmse_c)),
            'rmse': float(np.mean(valid_rmse)),
            'vrmse_u': float(np.mean(valid_vrmse_u)),
            'vrmse_c': float(np.mean(valid_vrmse_c)),
            'nrmse_u': float(np.mean(valid_nrmse_u)),
            'nrmse_c': float(np.mean(valid_nrmse_c)),
            'vrmse_all': float(np.mean(valid_vrmse_all)),
            'nrmse_all': float(np.mean(valid_nrmse_all)),
            'num_batches': n,
            'per_clip': {
                'pde': valid_pde,
                'rmse_u': valid_rmse_u,
                'rmse_c': valid_rmse_c,
                'rmse': valid_rmse,
                'vrmse_u': valid_vrmse_u,
                'vrmse_c': valid_vrmse_c,
                'nrmse_u': valid_nrmse_u,
                'nrmse_c': valid_nrmse_c,
                'vrmse_all': valid_vrmse_all,
                'nrmse_all': valid_nrmse_all,
            },
        }
    return {'pde': 0, 'rmse_u': 0, 'rmse_c': 0,
            'rmse': 0, 'vrmse_u': 0, 'vrmse_c': 0,
            'nrmse_u': 0, 'nrmse_c': 0,
            'vrmse_all': 0, 'nrmse_all': 0,
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

    n_vis = len(sample_indices)
    if clips_per_sample > 1 and n_vis > 1:
        clip_offsets = np.linspace(0, clips_per_sample - 1, n_vis, dtype=int)
    else:
        clip_offsets = np.zeros(n_vis, dtype=int)

    print(f"Visualizing {n_vis} samples...")

    results = []
    all_metrics = {'rmse_u': [], 'rmse_c': [],
                   'vrmse_u': [], 'vrmse_c': [],
                   'nrmse_u': [], 'nrmse_c': [],
                   'vrmse_all': [], 'nrmse_all': [],
                   'pde': []}

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

        rmse_u = torch.sqrt(torch.mean((output[..., CH_U] - target[..., CH_U]) ** 2) + 1e-8).item()
        rmse_c = torch.sqrt(torch.mean((output[..., CH_C] - target[..., CH_C]) ** 2) + 1e-8).item()

        vrmse_u = _vrmse_torch(target[..., CH_U], output[..., CH_U]).item()
        vrmse_c = _vrmse_torch(target[..., CH_C], output[..., CH_C]).item()

        nrmse_u = _nrmse_torch(target[..., CH_U], output[..., CH_U]).item()
        nrmse_c = _nrmse_torch(target[..., CH_C], output[..., CH_C]).item()

        # Combined vrmse/nrmse as mean of per-channel values
        vrmse_all_vis = (vrmse_u + vrmse_c) / 2
        nrmse_all_vis = (nrmse_u + nrmse_c) / 2

        last = -1
        res_data = {
            'gt_u': target[0, last, :, :, CH_U].float().cpu().numpy(),
            'pred_u': output[0, last, :, :, CH_U].float().cpu().numpy(),
            'gt_c': target[0, last, :, :, CH_C].float().cpu().numpy(),
            'pred_c': output[0, last, :, :, CH_C].float().cpu().numpy(),
            'sample_idx': sample_idx, 'start_t': start_t,
        }
        results.append(res_data)

        all_metrics['rmse_u'].append(rmse_u)
        all_metrics['rmse_c'].append(rmse_c)
        all_metrics['vrmse_u'].append(vrmse_u)
        all_metrics['vrmse_c'].append(vrmse_c)
        all_metrics['nrmse_u'].append(nrmse_u)
        all_metrics['nrmse_c'].append(nrmse_c)
        all_metrics['vrmse_all'].append(vrmse_all_vis)
        all_metrics['nrmse_all'].append(nrmse_all_vis)
        all_metrics['pde'].append(pde_loss)

        print(f"  Sample {sample_idx}: t_start={start_t}, "
              f"RMSE_u={rmse_u:.6f}, RMSE_c={rmse_c:.6f}, "
              f"nRMSE_u={nrmse_u:.6f}, nRMSE_c={nrmse_c:.6f}, "
              f"PDE={pde_loss:.6f}")

    print("Plotting Wave-Gauss visualization...")
    plot_wave_gauss(results, str(output_dir / "visualization_wave_gauss.png"))

    print(f"\n{'='*60}")
    print("Visualization Complete (Wave-Gauss LoRA)")
    print(f"{'='*60}")
    print(f"\nAverage Metrics ({len(results)} samples):")
    print(f"  - RMSE (u):        {np.mean(all_metrics['rmse_u']):.6f}")
    print(f"  - RMSE (c):        {np.mean(all_metrics['rmse_c']):.6f}")
    print(f"  - VRMSE (u):       {np.mean(all_metrics['vrmse_u']):.6f}")
    print(f"  - VRMSE (c):       {np.mean(all_metrics['vrmse_c']):.6f}")
    print(f"  - nRMSE (u):       {np.mean(all_metrics['nrmse_u']):.6f}")
    print(f"  - nRMSE (c):       {np.mean(all_metrics['nrmse_c']):.6f}")
    print(f"  - VRMSE (all):     {np.mean(all_metrics['vrmse_all']):.6f}")
    print(f"  - nRMSE (all):     {np.mean(all_metrics['nrmse_all']):.6f}")
    print(f"  - PDE Loss:        {np.mean(all_metrics['pde']):.6f}")
    print(f"{'='*60}")


def plot_wave_gauss(results: list, save_path: str):
    """Plot GT vs Prediction vs Error for displacement u (row 1) and wave speed c (row 2)."""
    n_samples = len(results)
    # 2 rows per sample: row 1 = u, row 2 = c
    fig, axes = plt.subplots(n_samples * 2, 3, figsize=(15, 4 * n_samples * 2))

    if n_samples == 1:
        axes = axes.reshape(2, 3)

    for s, res in enumerate(results):
        gt_u = res['gt_u']
        pred_u = res['pred_u']
        err_u = pred_u - gt_u
        gt_c = res['gt_c']
        pred_c = res['pred_c']
        err_c = pred_c - gt_c
        sample_idx = res['sample_idx']
        start_t = res.get('start_t', '?')

        row_u = s * 2
        row_c = s * 2 + 1

        # --- Row 1: displacement u ---
        vmin_u = min(gt_u.min(), pred_u.min())
        vmax_u = max(gt_u.max(), pred_u.max())

        im0 = axes[row_u, 0].imshow(gt_u, cmap='RdBu_r', vmin=vmin_u, vmax=vmax_u)
        axes[row_u, 0].set_title('GT (displacement u)', fontsize=11)
        axes[row_u, 0].set_ylabel(f'Sample {sample_idx}\nt0={start_t}', fontsize=10)
        plt.colorbar(im0, ax=axes[row_u, 0], fraction=0.046, pad=0.04)

        im1 = axes[row_u, 1].imshow(pred_u, cmap='RdBu_r', vmin=vmin_u, vmax=vmax_u)
        rmse_u_val = np.sqrt(np.mean(err_u**2))
        vrmse_u_val = _vrmse_np(gt_u, pred_u)
        axes[row_u, 1].set_title(f'Pred u (RMSE={rmse_u_val:.4f} VRMSE={vrmse_u_val:.4f})', fontsize=11)
        plt.colorbar(im1, ax=axes[row_u, 1], fraction=0.046, pad=0.04)

        err_u_max = max(np.percentile(np.abs(err_u), 95), 1e-6)
        im2 = axes[row_u, 2].imshow(err_u, cmap='RdBu_r', vmin=-err_u_max, vmax=err_u_max)
        axes[row_u, 2].set_title(f'Error u (MAE={np.mean(np.abs(err_u)):.4f})', fontsize=11)
        plt.colorbar(im2, ax=axes[row_u, 2], fraction=0.046, pad=0.04)

        # --- Row 2: wave speed c ---
        vmin_c = min(gt_c.min(), pred_c.min())
        vmax_c = max(gt_c.max(), pred_c.max())

        im3 = axes[row_c, 0].imshow(gt_c, cmap='viridis', vmin=vmin_c, vmax=vmax_c)
        axes[row_c, 0].set_title('GT (wave speed c)', fontsize=11)
        axes[row_c, 0].set_ylabel(f'Sample {sample_idx}\n(c field)', fontsize=10)
        plt.colorbar(im3, ax=axes[row_c, 0], fraction=0.046, pad=0.04)

        im4 = axes[row_c, 1].imshow(pred_c, cmap='viridis', vmin=vmin_c, vmax=vmax_c)
        rmse_c_val = np.sqrt(np.mean(err_c**2))
        vrmse_c_val = _vrmse_np(gt_c, pred_c)
        axes[row_c, 1].set_title(f'Pred c (RMSE={rmse_c_val:.4f} VRMSE={vrmse_c_val:.4f})', fontsize=11)
        plt.colorbar(im4, ax=axes[row_c, 1], fraction=0.046, pad=0.04)

        err_c_max = max(np.percentile(np.abs(err_c), 95), 1e-6)
        im5 = axes[row_c, 2].imshow(err_c, cmap='RdBu_r', vmin=-err_c_max, vmax=err_c_max)
        axes[row_c, 2].set_title(f'Error c (MAE={np.mean(np.abs(err_c)):.4f})', fontsize=11)
        plt.colorbar(im5, ax=axes[row_c, 2], fraction=0.046, pad=0.04)

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
        print("Wave-Gauss LoRA V3 Evaluation")
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
            print(f"  PDE Loss:       {results['pde']:.6f}")
            print(f"  RMSE (u):       {results['rmse_u']:.6f}")
            print(f"  RMSE (c):       {results['rmse_c']:.6f}")
            print(f"  RMSE:           {results['rmse']:.6f}")
            print(f"  VRMSE (u):      {results['vrmse_u']:.6f}")
            print(f"  VRMSE (c):      {results['vrmse_c']:.6f}")
            print(f"  nRMSE (u):      {results['nrmse_u']:.6f}")
            print(f"  nRMSE (c):      {results['nrmse_c']:.6f}")
            print(f"  VRMSE (all):    {results['vrmse_all']:.6f}")
            print(f"  nRMSE (all):    {results['nrmse_all']:.6f}")
            print(f"{'='*60}")

            if results.get('per_clip'):
                ckpt_dir = Path(args.checkpoint).parent
                plot_loss_distribution(
                    results['per_clip'],
                    str(ckpt_dir / "loss_distribution_wave_gauss.png"),
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
