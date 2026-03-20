"""
LoRA V3 Model Visualization for Shear Flow (Incompressible NS + Tracer).

Multi-GPU evaluation using Accelerate (same pattern as training).

Usage:
    # Multi-GPU full evaluation
    torchrun --nproc_per_node=8 tools/visualize_shear_flow_lora.py \
        --config configs/finetune_shear_flow_v3.yaml \
        --checkpoint checkpoints_shear_flow_lora_v3/best_lora.pt --scan_all

    # Single-GPU visualization (plot only)
    python tools/visualize_shear_flow_lora.py \
        --config configs/finetune_shear_flow_v3.yaml \
        --checkpoint checkpoints_shear_flow_lora_v3/best_lora.pt --output_dir ./shear_flow_vis
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
from finetune.pde_loss_verified import ShearFlowPDELossNPINN

# Shear Flow channel indices in 18-channel layout
CH_VX = 0       # vector: Vx
CH_VY = 1       # vector: Vy
CH_TRACER = 14  # scalar[11] = passive_tracer
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
    parser = argparse.ArgumentParser(description="LoRA V3 Visualization for Shear Flow")
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
    # Use checkpoint's saved config for architecture keys to avoid decoder mismatch
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


def create_pde_loss_fn(config: dict) -> ShearFlowPDELossNPINN:
    physics = config.get('physics', {})
    eq_scales = physics.get('eq_scales', None)
    eq_weights = physics.get('eq_weights', None)
    return ShearFlowPDELossNPINN(
        nx=physics.get('nx', 256),
        ny=physics.get('ny', 512),
        Lx=physics.get('Lx', 1.0),
        Ly=physics.get('Ly', 2.0),
        dt=physics.get('dt', 0.1),
        nu=physics.get('nu', 1e-4),
        D=physics.get('D', 1e-3),
        use_div_correction=True,
        eq_scales=eq_scales,
        eq_weights=eq_weights,
    )


def compute_pde_loss_from_output(
    output: torch.Tensor,
    input_data: torch.Tensor,
    pde_loss_fn: ShearFlowPDELossNPINN,
) -> torch.Tensor:
    """Compute Shear Flow PDE residual loss."""
    with torch.autocast(device_type='cuda', enabled=False):
        t0_u = input_data[:, 0:1, :, :, CH_VX].float()
        t0_v = input_data[:, 0:1, :, :, CH_VY].float()
        t0_p = input_data[:, 0:1, :, :, CH_PRESS].float()
        t0_s = input_data[:, 0:1, :, :, CH_TRACER].float()

        out_u = output[:, :, :, :, CH_VX].float()
        out_v = output[:, :, :, :, CH_VY].float()
        out_p = output[:, :, :, :, CH_PRESS].float()
        out_s = output[:, :, :, :, CH_TRACER].float()

        u = torch.cat([t0_u, out_u], dim=1)
        v = torch.cat([t0_v, out_v], dim=1)
        p = torch.cat([t0_p, out_p], dim=1)
        s = torch.cat([t0_s, out_s], dim=1)

        total_loss, losses = pde_loss_fn(u, v, p, s)
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
    local_rmse_vx = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_rmse_vy = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_rmse_tracer = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_rmse_press = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_rmse = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_vrmse_vx = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_vrmse_vy = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_vrmse_tracer = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_vrmse_press = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_nrmse_vx = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_nrmse_vy = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_nrmse_tracer = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_nrmse_press = torch.full((max_batches,), float('nan'), device=accelerator.device)
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

        rmse_vx = torch.sqrt(torch.mean((output[..., CH_VX] - target_data[..., CH_VX]) ** 2) + 1e-8)
        rmse_vy = torch.sqrt(torch.mean((output[..., CH_VY] - target_data[..., CH_VY]) ** 2) + 1e-8)
        rmse_tracer = torch.sqrt(torch.mean((output[..., CH_TRACER] - target_data[..., CH_TRACER]) ** 2) + 1e-8)
        rmse_press = torch.sqrt(torch.mean((output[..., CH_PRESS] - target_data[..., CH_PRESS]) ** 2) + 1e-8)

        vrmse_vx = _vrmse_torch(target_data[..., CH_VX], output[..., CH_VX])
        vrmse_vy = _vrmse_torch(target_data[..., CH_VY], output[..., CH_VY])
        vrmse_tracer = _vrmse_torch(target_data[..., CH_TRACER], output[..., CH_TRACER])
        vrmse_press = _vrmse_torch(target_data[..., CH_PRESS], output[..., CH_PRESS])

        nrmse_vx = _nrmse_torch(target_data[..., CH_VX], output[..., CH_VX])
        nrmse_vy = _nrmse_torch(target_data[..., CH_VY], output[..., CH_VY])
        nrmse_tracer = _nrmse_torch(target_data[..., CH_TRACER], output[..., CH_TRACER])
        nrmse_press = _nrmse_torch(target_data[..., CH_PRESS], output[..., CH_PRESS])

        valid_ch = (
            torch.where(channel_mask[0] > 0)[0]
            if channel_mask.dim() > 1
            else torch.where(channel_mask > 0)[0]
        )
        mse = torch.mean((output[..., valid_ch] - target_data[..., valid_ch]) ** 2)
        rmse = torch.sqrt(mse + 1e-8)

        # Combined vrmse/nrmse: mean of per-channel values
        vrmse_all = (vrmse_vx + vrmse_vy + vrmse_tracer + vrmse_press) / 4
        nrmse_all = (nrmse_vx + nrmse_vy + nrmse_tracer + nrmse_press) / 4

        local_pde[i] = pde_loss.detach()
        local_rmse_vx[i] = rmse_vx.detach()
        local_rmse_vy[i] = rmse_vy.detach()
        local_rmse_tracer[i] = rmse_tracer.detach()
        local_rmse_press[i] = rmse_press.detach()
        local_rmse[i] = rmse.detach()
        local_vrmse_vx[i] = vrmse_vx.detach()
        local_vrmse_vy[i] = vrmse_vy.detach()
        local_vrmse_tracer[i] = vrmse_tracer.detach()
        local_vrmse_press[i] = vrmse_press.detach()
        local_nrmse_vx[i] = nrmse_vx.detach()
        local_nrmse_vy[i] = nrmse_vy.detach()
        local_nrmse_tracer[i] = nrmse_tracer.detach()
        local_nrmse_press[i] = nrmse_press.detach()
        local_vrmse_all[i] = vrmse_all.detach()
        local_nrmse_all[i] = nrmse_all.detach()

    accelerator.wait_for_everyone()
    all_pde = accelerator.gather(local_pde)
    all_rmse_vx = accelerator.gather(local_rmse_vx)
    all_rmse_vy = accelerator.gather(local_rmse_vy)
    all_rmse_tracer = accelerator.gather(local_rmse_tracer)
    all_rmse_press = accelerator.gather(local_rmse_press)
    all_rmse = accelerator.gather(local_rmse)
    all_vrmse_vx = accelerator.gather(local_vrmse_vx)
    all_vrmse_vy = accelerator.gather(local_vrmse_vy)
    all_vrmse_tracer = accelerator.gather(local_vrmse_tracer)
    all_vrmse_press = accelerator.gather(local_vrmse_press)
    all_nrmse_vx = accelerator.gather(local_nrmse_vx)
    all_nrmse_vy = accelerator.gather(local_nrmse_vy)
    all_nrmse_tracer = accelerator.gather(local_nrmse_tracer)
    all_nrmse_press = accelerator.gather(local_nrmse_press)
    all_vrmse_all = accelerator.gather(local_vrmse_all)
    all_nrmse_all = accelerator.gather(local_nrmse_all)
    accelerator.wait_for_everyone()

    valid_mask = ~torch.isnan(all_pde)
    valid_pde = all_pde[valid_mask].cpu().numpy()
    valid_rmse_vx = all_rmse_vx[valid_mask].cpu().numpy()
    valid_rmse_vy = all_rmse_vy[valid_mask].cpu().numpy()
    valid_rmse_tracer = all_rmse_tracer[valid_mask].cpu().numpy()
    valid_rmse_press = all_rmse_press[valid_mask].cpu().numpy()
    valid_rmse = all_rmse[valid_mask].cpu().numpy()
    valid_vrmse_vx = all_vrmse_vx[valid_mask].cpu().numpy()
    valid_vrmse_vy = all_vrmse_vy[valid_mask].cpu().numpy()
    valid_vrmse_tracer = all_vrmse_tracer[valid_mask].cpu().numpy()
    valid_vrmse_press = all_vrmse_press[valid_mask].cpu().numpy()
    valid_nrmse_vx = all_nrmse_vx[valid_mask].cpu().numpy()
    valid_nrmse_vy = all_nrmse_vy[valid_mask].cpu().numpy()
    valid_nrmse_tracer = all_nrmse_tracer[valid_mask].cpu().numpy()
    valid_nrmse_press = all_nrmse_press[valid_mask].cpu().numpy()
    valid_vrmse_all = all_vrmse_all[valid_mask].cpu().numpy()
    valid_nrmse_all = all_nrmse_all[valid_mask].cpu().numpy()

    n = len(valid_pde)
    if n > 0:
        return {
            'pde': float(np.mean(valid_pde)),
            'rmse_vx': float(np.mean(valid_rmse_vx)),
            'rmse_vy': float(np.mean(valid_rmse_vy)),
            'rmse_tracer': float(np.mean(valid_rmse_tracer)),
            'rmse_press': float(np.mean(valid_rmse_press)),
            'rmse': float(np.mean(valid_rmse)),
            'vrmse_vx': float(np.mean(valid_vrmse_vx)),
            'vrmse_vy': float(np.mean(valid_vrmse_vy)),
            'vrmse_tracer': float(np.mean(valid_vrmse_tracer)),
            'vrmse_press': float(np.mean(valid_vrmse_press)),
            'nrmse_vx': float(np.mean(valid_nrmse_vx)),
            'nrmse_vy': float(np.mean(valid_nrmse_vy)),
            'nrmse_tracer': float(np.mean(valid_nrmse_tracer)),
            'nrmse_press': float(np.mean(valid_nrmse_press)),
            'vrmse_all': float(np.mean(valid_vrmse_all)),
            'nrmse_all': float(np.mean(valid_nrmse_all)),
            'num_batches': n,
            'per_clip': {
                'pde': valid_pde,
                'rmse_vx': valid_rmse_vx,
                'rmse_vy': valid_rmse_vy,
                'rmse_tracer': valid_rmse_tracer,
                'rmse_press': valid_rmse_press,
                'rmse': valid_rmse,
                'vrmse_vx': valid_vrmse_vx,
                'vrmse_vy': valid_vrmse_vy,
                'vrmse_tracer': valid_vrmse_tracer,
                'vrmse_press': valid_vrmse_press,
                'nrmse_vx': valid_nrmse_vx,
                'nrmse_vy': valid_nrmse_vy,
                'nrmse_tracer': valid_nrmse_tracer,
                'nrmse_press': valid_nrmse_press,
                'vrmse_all': valid_vrmse_all,
                'nrmse_all': valid_nrmse_all,
            },
        }
    return {'pde': 0, 'rmse_vx': 0, 'rmse_vy': 0, 'rmse_tracer': 0,
            'rmse_press': 0, 'rmse': 0, 'vrmse_vx': 0, 'vrmse_vy': 0,
            'vrmse_tracer': 0, 'vrmse_press': 0, 'nrmse_vx': 0, 'nrmse_vy': 0,
            'nrmse_tracer': 0, 'nrmse_press': 0,
            'vrmse_all': 0, 'nrmse_all': 0, 'num_batches': 0, 'per_clip': {}}


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
    all_metrics = {'rmse_vx': [], 'rmse_vy': [], 'rmse_tracer': [], 'rmse_press': [],
                    'vrmse_vx': [], 'vrmse_vy': [], 'vrmse_tracer': [], 'vrmse_press': [],
                    'nrmse_vx': [], 'nrmse_vy': [], 'nrmse_tracer': [], 'nrmse_press': [],
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

        rmse_vx = torch.sqrt(torch.mean((output[..., CH_VX] - target[..., CH_VX]) ** 2) + 1e-8).item()
        rmse_vy = torch.sqrt(torch.mean((output[..., CH_VY] - target[..., CH_VY]) ** 2) + 1e-8).item()
        rmse_tracer = torch.sqrt(torch.mean((output[..., CH_TRACER] - target[..., CH_TRACER]) ** 2) + 1e-8).item()
        rmse_press = torch.sqrt(torch.mean((output[..., CH_PRESS] - target[..., CH_PRESS]) ** 2) + 1e-8).item()

        vrmse_vx = _vrmse_torch(target[..., CH_VX], output[..., CH_VX]).item()
        vrmse_vy = _vrmse_torch(target[..., CH_VY], output[..., CH_VY]).item()
        vrmse_tracer = _vrmse_torch(target[..., CH_TRACER], output[..., CH_TRACER]).item()
        vrmse_press = _vrmse_torch(target[..., CH_PRESS], output[..., CH_PRESS]).item()

        nrmse_vx = _nrmse_torch(target[..., CH_VX], output[..., CH_VX]).item()
        nrmse_vy = _nrmse_torch(target[..., CH_VY], output[..., CH_VY]).item()
        nrmse_tracer = _nrmse_torch(target[..., CH_TRACER], output[..., CH_TRACER]).item()
        nrmse_press = _nrmse_torch(target[..., CH_PRESS], output[..., CH_PRESS]).item()

        # Combined vrmse/nrmse: mean of per-channel values
        vrmse_all_val = (vrmse_vx + vrmse_vy + vrmse_tracer + vrmse_press) / 4
        nrmse_all_val = (nrmse_vx + nrmse_vy + nrmse_tracer + nrmse_press) / 4

        # Last timestep for plotting
        last = -1
        res_data = {
            'gt_vx': target[0, last, :, :, CH_VX].float().cpu().numpy(),
            'pred_vx': output[0, last, :, :, CH_VX].float().cpu().numpy(),
            'gt_vy': target[0, last, :, :, CH_VY].float().cpu().numpy(),
            'pred_vy': output[0, last, :, :, CH_VY].float().cpu().numpy(),
            'gt_tracer': target[0, last, :, :, CH_TRACER].float().cpu().numpy(),
            'pred_tracer': output[0, last, :, :, CH_TRACER].float().cpu().numpy(),
            'gt_press': target[0, last, :, :, CH_PRESS].float().cpu().numpy(),
            'pred_press': output[0, last, :, :, CH_PRESS].float().cpu().numpy(),
            'sample_idx': sample_idx, 'start_t': start_t,
        }
        results.append(res_data)

        all_metrics['rmse_vx'].append(rmse_vx)
        all_metrics['rmse_vy'].append(rmse_vy)
        all_metrics['rmse_tracer'].append(rmse_tracer)
        all_metrics['rmse_press'].append(rmse_press)
        all_metrics['vrmse_vx'].append(vrmse_vx)
        all_metrics['vrmse_vy'].append(vrmse_vy)
        all_metrics['vrmse_tracer'].append(vrmse_tracer)
        all_metrics['vrmse_press'].append(vrmse_press)
        all_metrics['nrmse_vx'].append(nrmse_vx)
        all_metrics['nrmse_vy'].append(nrmse_vy)
        all_metrics['nrmse_tracer'].append(nrmse_tracer)
        all_metrics['nrmse_press'].append(nrmse_press)
        all_metrics['vrmse_all'].append(vrmse_all_val)
        all_metrics['nrmse_all'].append(nrmse_all_val)
        all_metrics['pde'].append(pde_loss)

        print(f"  Sample {sample_idx}: t_start={start_t}, "
              f"RMSE_vx={rmse_vx:.6f}, RMSE_vy={rmse_vy:.6f}, "
              f"RMSE_s={rmse_tracer:.6f}, RMSE_p={rmse_press:.6f}, "
              f"nRMSE_vx={nrmse_vx:.6f}, nRMSE_vy={nrmse_vy:.6f}, "
              f"nRMSE_s={nrmse_tracer:.6f}, nRMSE_p={nrmse_press:.6f}, PDE={pde_loss:.6f}")

    # Plot velocity
    print("Plotting velocity visualization...")
    plot_velocity(results, str(output_dir / "visualization_shear_flow_velocity.png"))

    # Plot scalar fields
    print("Plotting scalar visualization...")
    plot_scalar(results, str(output_dir / "visualization_shear_flow_scalar.png"))

    print(f"\n{'='*60}")
    print("Visualization Complete (Shear Flow LoRA)")
    print(f"{'='*60}")
    print(f"\nAverage Metrics ({len(results)} samples):")
    print(f"  - RMSE (Vx):      {np.mean(all_metrics['rmse_vx']):.6f}")
    print(f"  - RMSE (Vy):      {np.mean(all_metrics['rmse_vy']):.6f}")
    print(f"  - RMSE (tracer):  {np.mean(all_metrics['rmse_tracer']):.6f}")
    print(f"  - RMSE (press):   {np.mean(all_metrics['rmse_press']):.6f}")
    print(f"  - VRMSE (Vx):     {np.mean(all_metrics['vrmse_vx']):.6f}")
    print(f"  - VRMSE (Vy):     {np.mean(all_metrics['vrmse_vy']):.6f}")
    print(f"  - VRMSE (tracer): {np.mean(all_metrics['vrmse_tracer']):.6f}")
    print(f"  - VRMSE (press):  {np.mean(all_metrics['vrmse_press']):.6f}")
    print(f"  - nRMSE (Vx):     {np.mean(all_metrics['nrmse_vx']):.6f}")
    print(f"  - nRMSE (Vy):     {np.mean(all_metrics['nrmse_vy']):.6f}")
    print(f"  - nRMSE (tracer): {np.mean(all_metrics['nrmse_tracer']):.6f}")
    print(f"  - nRMSE (press):  {np.mean(all_metrics['nrmse_press']):.6f}")
    print(f"  - VRMSE (all):    {np.mean(all_metrics['vrmse_all']):.6f}")
    print(f"  - nRMSE (all):    {np.mean(all_metrics['nrmse_all']):.6f}")
    print(f"  - PDE Loss:       {np.mean(all_metrics['pde']):.6f}")
    print(f"{'='*60}")


def _plot_field_row(axes, row: int, gt: np.ndarray, pred: np.ndarray,
                    field_name: str, sample_idx: int, start_t: int,
                    cmap: str = 'RdBu_r'):
    """Plot GT, Pred, Error for one field in a row."""
    err = pred - gt
    vmin = min(gt.min(), pred.min())
    vmax = max(gt.max(), pred.max())

    im0 = axes[row, 0].imshow(gt, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[row, 0].set_title(f'GT ({field_name})', fontsize=11)
    axes[row, 0].set_ylabel(f'Sample {sample_idx}\nt0={start_t}', fontsize=10)
    plt.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)

    im1 = axes[row, 1].imshow(pred, cmap=cmap, vmin=vmin, vmax=vmax)
    rmse = np.sqrt(np.mean((pred - gt)**2))
    vrmse = _vrmse_np(gt, pred)
    axes[row, 1].set_title(f'Pred {field_name} (RMSE={rmse:.4f} VRMSE={vrmse:.4f})', fontsize=11)
    plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)

    err_max = max(np.percentile(np.abs(err), 95), 1e-6)
    im2 = axes[row, 2].imshow(err, cmap='RdBu_r',
                                vmin=-err_max, vmax=err_max)
    axes[row, 2].set_title(f'Error {field_name} (MAE={np.mean(np.abs(err)):.4f})', fontsize=11)
    plt.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)


def plot_velocity(results: list, save_path: str):
    """Plot GT vs Prediction vs Error for Vx and Vy."""
    n_samples = len(results)
    fig, axes = plt.subplots(n_samples, 6, figsize=(30, 4 * n_samples))

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

        # Vx
        vmin_vx = min(gt_vx.min(), pred_vx.min())
        vmax_vx = max(gt_vx.max(), pred_vx.max())

        im0 = axes[row, 0].imshow(gt_vx, cmap='RdBu_r',
                                    vmin=vmin_vx, vmax=vmax_vx)
        axes[row, 0].set_title('GT (Vx)', fontsize=11)
        axes[row, 0].set_ylabel(f'Sample {sample_idx}\nt0={start_t}', fontsize=10)
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)

        im1 = axes[row, 1].imshow(pred_vx, cmap='RdBu_r',
                                    vmin=vmin_vx, vmax=vmax_vx)
        rmse_vx = np.sqrt(np.mean(err_vx**2))
        vrmse_vx = _vrmse_np(gt_vx, pred_vx)
        axes[row, 1].set_title(f'Pred Vx (RMSE={rmse_vx:.4f} VRMSE={vrmse_vx:.4f})', fontsize=11)
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)

        err_vx_max = max(np.percentile(np.abs(err_vx), 95), 1e-6)
        im2 = axes[row, 2].imshow(err_vx, cmap='RdBu_r',
                                    vmin=-err_vx_max, vmax=err_vx_max)
        axes[row, 2].set_title(f'Error Vx (MAE={np.mean(np.abs(err_vx)):.4f})', fontsize=11)
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)

        # Vy
        vmin_vy = min(gt_vy.min(), pred_vy.min())
        vmax_vy = max(gt_vy.max(), pred_vy.max())

        im3 = axes[row, 3].imshow(gt_vy, cmap='RdBu_r',
                                    vmin=vmin_vy, vmax=vmax_vy)
        axes[row, 3].set_title('GT (Vy)', fontsize=11)
        plt.colorbar(im3, ax=axes[row, 3], fraction=0.046, pad=0.04)

        im4 = axes[row, 4].imshow(pred_vy, cmap='RdBu_r',
                                    vmin=vmin_vy, vmax=vmax_vy)
        rmse_vy = np.sqrt(np.mean(err_vy**2))
        vrmse_vy = _vrmse_np(gt_vy, pred_vy)
        axes[row, 4].set_title(f'Pred Vy (RMSE={rmse_vy:.4f} VRMSE={vrmse_vy:.4f})', fontsize=11)
        plt.colorbar(im4, ax=axes[row, 4], fraction=0.046, pad=0.04)

        err_vy_max = max(np.percentile(np.abs(err_vy), 95), 1e-6)
        im5 = axes[row, 5].imshow(err_vy, cmap='RdBu_r',
                                    vmin=-err_vy_max, vmax=err_vy_max)
        axes[row, 5].set_title(f'Error Vy (MAE={np.mean(np.abs(err_vy)):.4f})', fontsize=11)
        plt.colorbar(im5, ax=axes[row, 5], fraction=0.046, pad=0.04)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_scalar(results: list, save_path: str):
    """Plot GT vs Prediction vs Error for tracer and pressure."""
    n_samples = len(results)
    fig, axes = plt.subplots(n_samples, 6, figsize=(30, 4 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for row, res in enumerate(results):
        gt_s = res['gt_tracer']
        pred_s = res['pred_tracer']
        err_s = pred_s - gt_s
        gt_p = res['gt_press']
        pred_p = res['pred_press']
        err_p = pred_p - gt_p
        sample_idx = res['sample_idx']
        start_t = res.get('start_t', '?')

        # Tracer
        vmin_s = min(gt_s.min(), pred_s.min())
        vmax_s = max(gt_s.max(), pred_s.max())

        im0 = axes[row, 0].imshow(gt_s, cmap='RdBu_r',
                                    vmin=vmin_s, vmax=vmax_s)
        axes[row, 0].set_title('GT (tracer)', fontsize=11)
        axes[row, 0].set_ylabel(f'Sample {sample_idx}\nt0={start_t}', fontsize=10)
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)

        im1 = axes[row, 1].imshow(pred_s, cmap='RdBu_r',
                                    vmin=vmin_s, vmax=vmax_s)
        rmse_s = np.sqrt(np.mean(err_s**2))
        vrmse_s = _vrmse_np(gt_s, pred_s)
        axes[row, 1].set_title(f'Pred tracer (RMSE={rmse_s:.4f} VRMSE={vrmse_s:.4f})', fontsize=11)
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)

        err_s_max = max(np.percentile(np.abs(err_s), 95), 1e-6)
        im2 = axes[row, 2].imshow(err_s, cmap='RdBu_r',
                                    vmin=-err_s_max, vmax=err_s_max)
        axes[row, 2].set_title(f'Error tracer (MAE={np.mean(np.abs(err_s)):.4f})', fontsize=11)
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)

        # Pressure
        vmin_p = min(gt_p.min(), pred_p.min())
        vmax_p = max(gt_p.max(), pred_p.max())

        im3 = axes[row, 3].imshow(gt_p, cmap='RdBu_r',
                                    vmin=vmin_p, vmax=vmax_p)
        axes[row, 3].set_title('GT (pressure)', fontsize=11)
        plt.colorbar(im3, ax=axes[row, 3], fraction=0.046, pad=0.04)

        im4 = axes[row, 4].imshow(pred_p, cmap='RdBu_r',
                                    vmin=vmin_p, vmax=vmax_p)
        rmse_p = np.sqrt(np.mean(err_p**2))
        vrmse_p = _vrmse_np(gt_p, pred_p)
        axes[row, 4].set_title(f'Pred press (RMSE={rmse_p:.4f} VRMSE={vrmse_p:.4f})', fontsize=11)
        plt.colorbar(im4, ax=axes[row, 4], fraction=0.046, pad=0.04)

        err_p_max = max(np.percentile(np.abs(err_p), 95), 1e-6)
        im5 = axes[row, 5].imshow(err_p, cmap='RdBu_r',
                                    vmin=-err_p_max, vmax=err_p_max)
        axes[row, 5].set_title(f'Error press (MAE={np.mean(np.abs(err_p)):.4f})', fontsize=11)
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
        print(f"Shear Flow LoRA V3 Evaluation")
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
            print(f"  RMSE (tracer): {results['rmse_tracer']:.6f}")
            print(f"  RMSE (press):  {results['rmse_press']:.6f}")
            print(f"  RMSE:          {results['rmse']:.6f}")
            print(f"  VRMSE (Vx):    {results['vrmse_vx']:.6f}")
            print(f"  VRMSE (Vy):    {results['vrmse_vy']:.6f}")
            print(f"  VRMSE (tracer):{results['vrmse_tracer']:.6f}")
            print(f"  VRMSE (press): {results['vrmse_press']:.6f}")
            print(f"  nRMSE (Vx):    {results['nrmse_vx']:.6f}")
            print(f"  nRMSE (Vy):    {results['nrmse_vy']:.6f}")
            print(f"  nRMSE (tracer):{results['nrmse_tracer']:.6f}")
            print(f"  nRMSE (press): {results['nrmse_press']:.6f}")
            print(f"  VRMSE (all):   {results['vrmse_all']:.6f}")
            print(f"  nRMSE (all):   {results['nrmse_all']:.6f}")
            print(f"{'='*60}")

            if results.get('per_clip'):
                ckpt_dir = Path(args.checkpoint).parent
                plot_loss_distribution(
                    results['per_clip'],
                    str(ckpt_dir / "loss_distribution_shear_flow.png"),
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
