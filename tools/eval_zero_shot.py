"""
Zero-shot evaluation: load a pretrained/full-finetuned PDEModelV3 (no LoRA)
and evaluate on a finetune dataset.

Reports per-channel and aggregate metrics matching visualize_*_lora.py output.
Supports PDE loss if config has physics section.

Usage:
    # Zero-shot NS-PwC fullft → NS-SVS
    CUDA_VISIBLE_DEVICES=0 python tools/eval_zero_shot.py \
        --checkpoint checkpoints_ns_pwc_fullft/best_tf.pt \
        --config configs/finetune_ns_svs_v3.yaml

    # Multi-GPU
    CUDA_VISIBLE_DEVICES=1,3,4,7 torchrun --nproc_per_node=4 tools/eval_zero_shot.py \
        --checkpoint checkpoints_ns_pwc_fullft/best_tf.pt \
        --config configs/finetune_ns_svs_v3.yaml
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from torch.utils.data import DataLoader
from accelerate import Accelerator

from finetune.dataset_finetune import (
    FinetuneDataset, FinetuneSampler, finetune_collate_fn,
)
from pretrain.model_v3 import PDEModelV3


def _vrmse_torch(gt: torch.Tensor, pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse = torch.mean((pred - gt) ** 2)
    var = torch.mean((gt - gt.mean()) ** 2)
    return torch.sqrt(mse / (var + eps))


def _nrmse_torch(gt: torch.Tensor, pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse_pred = torch.mean((pred - gt) ** 2)
    mse_zero = torch.mean(gt ** 2)
    return torch.sqrt(mse_pred / (mse_zero + eps))


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot evaluation (no LoRA)")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to pretrained/full-finetuned checkpoint (.pt)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to finetune config (for dataset + physics)')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--t_input', type=int, default=None)
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model(config: dict, checkpoint_path: str, is_main: bool = True) -> PDEModelV3:
    """Load PDEModelV3 from checkpoint, using checkpoint's saved config for architecture."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'config' in ckpt:
        ckpt_model_cfg = ckpt['config'].get('model', {})
        arch_keys = ['decoder', 'patch_smoother', 'encoder', 'intra_patch', 'na',
                     'in_channels', 'hidden_dim', 'patch_size', 'num_layers', 'num_heads',
                     'vector_channels', 'scalar_channels', 'enable_1d', 'enable_3d']
        model_cfg = dict(config.get('model', {}))
        for k in arch_keys:
            if k in ckpt_model_cfg:
                if is_main and k == 'decoder' and model_cfg.get(k) != ckpt_model_cfg[k]:
                    print(f"  [INFO] Overriding decoder config from checkpoint")
                model_cfg[k] = ckpt_model_cfg[k]
        eval_config = {**config, 'model': model_cfg}
    else:
        eval_config = config

    model = PDEModelV3(eval_config)

    state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    cleaned = {}
    for k, v in state_dict.items():
        k = k.removeprefix('module.').removeprefix('_orig_mod.')
        cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if is_main:
        print(f"  Loaded {len(cleaned) - len(unexpected)} keys")
        if missing:
            print(f"  Missing: {len(missing)} keys")
        if unexpected:
            print(f"  Unexpected: {len(unexpected)} keys")
        if 'val_rmse' in ckpt:
            print(f"  Checkpoint val_rmse: {ckpt['val_rmse']:.6f}")
        if 'global_step' in ckpt:
            print(f"  Global step: {ckpt['global_step']}")

    model = model.float()
    return model


def detect_channels(config: dict) -> Dict[str, int]:
    """Detect active channel names and indices from config's model_name or known patterns."""
    # Try to infer from the config's dataset path or model_name
    path = config.get('dataset', {}).get('path', '')
    model_name = config.get('model_name', '')

    name = (path + model_name).lower()

    if 'ns_svs' in name or 'ns_pwc' in name or 'shear_flow' in name:
        return {'Vx': 0, 'Vy': 1, 'tracer': 14}
    elif 'rayleigh' in name:
        return {'Vx': 0, 'Vy': 1, 'buoy': 3, 'press': 15}
    elif 'turbulent' in name:
        return {'Vx': 0, 'Vy': 1, 'density': 7, 'press': 15}
    elif 'gray_scott' in name:
        return {'A': 5, 'B': 6}
    elif 'active_matter' in name:
        return {'Vx': 0, 'Vy': 1, 'conc': 3, 'Dxx': 4, 'Dxy': 5, 'Dyy': 6}
    elif 'wave_gauss' in name:
        return {'u': 3, 'c': 4}
    else:
        return {}


def try_create_pde_loss(config: dict):
    """Try to create PDE loss function from config. Returns None if not possible."""
    physics = config.get('physics', {})
    if not physics:
        return None

    model_name = config.get('model_name', '')
    path = config.get('dataset', {}).get('path', '')
    name = (path + model_name).lower()

    eq_scales = physics.get('eq_scales', None)
    eq_weights = physics.get('eq_weights', None)

    try:
        if 'ns_svs' in name or 'ns_pwc' in name:
            from finetune.pde_loss_verified import NSPwCPDELoss
            return NSPwCPDELoss(
                nx=physics.get('nx', 128), ny=physics.get('ny', 128),
                Lx=physics.get('Lx', 1.0), Ly=physics.get('Ly', 1.0),
                dt=physics.get('dt', 0.05), nu=physics.get('nu', 4e-4),
                kappa=physics.get('kappa', 4e-4),
                eq_scales=eq_scales, eq_weights=eq_weights,
            ), 'ns'
        elif 'shear_flow' in name:
            from finetune.pde_loss_verified import ShearFlowPDELossNPINN
            return ShearFlowPDELossNPINN(
                nx=physics.get('nx', 256), ny=physics.get('ny', 512),
                Lx=physics.get('Lx', 1.0), Ly=physics.get('Ly', 2.0),
                dt=physics.get('dt', 0.1), nu=physics.get('nu', 1e-4),
                D=physics.get('D', 1e-3),
                eq_scales=eq_scales, eq_weights=eq_weights,
            ), 'shear_flow'
        elif 'active_matter' in name:
            from finetune.pde_loss_verified import ActiveMatterNPINNPDELoss
            return ActiveMatterNPINNPDELoss(
                nx=physics.get('nx', 256), ny=physics.get('ny', 256),
                Lx=physics.get('Lx', 10.0), Ly=physics.get('Ly', 10.0),
                dt=physics.get('dt', 0.25), d_T=physics.get('d_T', 0.05),
                eq_scales=eq_scales, eq_weights=eq_weights,
            ), 'active_matter'
    except Exception:
        pass
    return None, None


def compute_pde_loss_ns(output, input_data, pde_loss_fn, ch_vx, ch_vy, ch_tracer):
    """PDE loss for NS-PwC/SVS type."""
    with torch.autocast(device_type='cuda', enabled=False):
        t0_ux = input_data[:, 0:1, :, :, ch_vx].float()
        t0_uy = input_data[:, 0:1, :, :, ch_vy].float()
        t0_tr = input_data[:, 0:1, :, :, ch_tracer].float()
        ux = torch.cat([t0_ux, output[:, :, :, :, ch_vx].float()], dim=1)
        uy = torch.cat([t0_uy, output[:, :, :, :, ch_vy].float()], dim=1)
        tracer = torch.cat([t0_tr, output[:, :, :, :, ch_tracer].float()], dim=1)
        total_loss, _ = pde_loss_fn(ux, uy, tracer)
    return total_loss


@torch.no_grad()
def evaluate(accelerator, model, val_loader, t_input: int,
             channels: Dict[str, int], pde_loss_fn=None, pde_type: str = None):
    """Distributed evaluation matching visualize_*_lora.py output."""
    accelerator.wait_for_everyone()
    model.eval()

    ch_names = list(channels.keys())
    ch_indices = list(channels.values())

    max_batches = len(val_loader)

    # Per-channel metrics
    local_rmse_ch = {name: torch.full((max_batches,), float('nan'), device=accelerator.device)
                     for name in ch_names}
    local_vrmse_ch = {name: torch.full((max_batches,), float('nan'), device=accelerator.device)
                      for name in ch_names}
    local_nrmse_ch = {name: torch.full((max_batches,), float('nan'), device=accelerator.device)
                      for name in ch_names}
    # Aggregate
    local_rmse = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_vrmse_all = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_nrmse_all = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_pde = torch.full((max_batches,), float('nan'), device=accelerator.device)

    for i, batch in enumerate(val_loader):
        data = batch['data'].to(device=accelerator.device, dtype=torch.float32)
        channel_mask = batch['channel_mask'].to(device=accelerator.device)

        input_data = data[:, :t_input]
        target_data = data[:, 1:t_input + 1]

        output_norm, mean, std = model(input_data, return_normalized=True)
        output = output_norm * std + mean

        # Per-channel metrics
        for name, idx in channels.items():
            pred_ch = output[..., idx]
            gt_ch = target_data[..., idx]
            local_rmse_ch[name][i] = torch.sqrt(torch.mean((pred_ch - gt_ch) ** 2) + 1e-8).detach()
            local_vrmse_ch[name][i] = _vrmse_torch(gt_ch, pred_ch).detach()
            local_nrmse_ch[name][i] = _nrmse_torch(gt_ch, pred_ch).detach()

        # Aggregate over all valid channels
        valid_ch = (torch.where(channel_mask[0] > 0)[0] if channel_mask.dim() > 1
                    else torch.where(channel_mask > 0)[0])
        output_valid = output[..., valid_ch]
        target_valid = target_data[..., valid_ch]
        mse_all = torch.mean((output_valid - target_valid) ** 2)
        local_rmse[i] = torch.sqrt(mse_all + 1e-8).detach()

        var_all = torch.mean((target_valid - target_valid.mean()) ** 2)
        local_vrmse_all[i] = torch.sqrt(mse_all / (var_all + 1e-8)).detach()

        mse_zero_all = torch.mean(target_valid ** 2)
        local_nrmse_all[i] = torch.sqrt(mse_all / (mse_zero_all + 1e-8)).detach()

        # PDE loss
        if pde_loss_fn is not None and pde_type == 'ns':
            ch_tracer = channels.get('tracer', 14)
            pde_val = compute_pde_loss_ns(output, input_data, pde_loss_fn,
                                          channels['Vx'], channels['Vy'], ch_tracer)
            local_pde[i] = pde_val.detach()

    # Gather
    accelerator.wait_for_everyone()
    all_rmse = accelerator.gather(local_rmse)
    all_vrmse_all = accelerator.gather(local_vrmse_all)
    all_nrmse_all = accelerator.gather(local_nrmse_all)
    all_pde = accelerator.gather(local_pde)

    all_rmse_ch = {name: accelerator.gather(local_rmse_ch[name]) for name in ch_names}
    all_vrmse_ch = {name: accelerator.gather(local_vrmse_ch[name]) for name in ch_names}
    all_nrmse_ch = {name: accelerator.gather(local_nrmse_ch[name]) for name in ch_names}
    accelerator.wait_for_everyone()

    valid_mask = ~torch.isnan(all_rmse)
    n = valid_mask.sum().item()

    results = {
        'rmse': float(all_rmse[valid_mask].cpu().numpy().mean()) if n > 0 else 0,
        'vrmse_all': float(all_vrmse_all[valid_mask].cpu().numpy().mean()) if n > 0 else 0,
        'nrmse_all': float(all_nrmse_all[valid_mask].cpu().numpy().mean()) if n > 0 else 0,
        'num_batches': n,
    }

    pde_valid = ~torch.isnan(all_pde)
    if pde_valid.sum() > 0:
        results['pde'] = float(all_pde[pde_valid].cpu().numpy().mean())

    for name in ch_names:
        results[f'rmse_{name}'] = float(all_rmse_ch[name][valid_mask].cpu().numpy().mean()) if n > 0 else 0
        results[f'vrmse_{name}'] = float(all_vrmse_ch[name][valid_mask].cpu().numpy().mean()) if n > 0 else 0
        results[f'nrmse_{name}'] = float(all_nrmse_ch[name][valid_mask].cpu().numpy().mean()) if n > 0 else 0

    return results


def main():
    args = parse_args()
    config = load_config(args.config)

    t_input = args.t_input or config.get('dataset', {}).get('t_input', 8)
    batch_size = args.batch_size or config.get('dataloader', {}).get('batch_size', 4)
    temporal_length = t_input + 1

    accelerator = Accelerator()
    is_main = accelerator.is_main_process

    if is_main:
        print(f"{'='*60}")
        print(f"Zero-Shot Evaluation (no LoRA)")
        print(f"{'='*60}")
        print(f"  Checkpoint: {args.checkpoint}")
        print(f"  Config:     {args.config}")
        print(f"  Dataset:    {config['dataset']['path']}")
        print(f"  Devices:    {accelerator.num_processes}")
        print(f"  t_input:    {t_input}")
        print(f"  batch_size: {batch_size}")
        print(f"{'='*60}")

    model = load_model(config, args.checkpoint, is_main=is_main)
    channels = detect_channels(config)
    if is_main:
        print(f"  Channels: {channels}")

    pde_result = try_create_pde_loss(config)
    pde_loss_fn, pde_type = (pde_result if pde_result and pde_result[0] is not None
                              else (None, None))
    if is_main:
        print(f"  PDE loss: {'enabled (' + pde_type + ')' if pde_loss_fn else 'disabled'}")

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
        print(f"\nScanning {len(val_dataset)} clips "
              f"across {accelerator.num_processes} GPUs...")

    results = evaluate(accelerator, model, val_loader, t_input,
                       channels, pde_loss_fn, pde_type)

    if is_main:
        ch_names = list(channels.keys())
        print(f"\n{'='*60}")
        print(f"Results ({results['num_batches']} batches):")
        if 'pde' in results:
            print(f"  PDE Loss:      {results['pde']:.6f}")
        for name in ch_names:
            print(f"  RMSE ({name:>8s}):  {results[f'rmse_{name}']:.6f}")
        print(f"  RMSE:            {results['rmse']:.6f}")
        for name in ch_names:
            print(f"  VRMSE ({name:>7s}):  {results[f'vrmse_{name}']:.6f}")
        for name in ch_names:
            print(f"  nRMSE ({name:>7s}):  {results[f'nrmse_{name}']:.6f}")
        print(f"  VRMSE (all):     {results['vrmse_all']:.6f}")
        print(f"  nRMSE (all):     {results['nrmse_all']:.6f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
