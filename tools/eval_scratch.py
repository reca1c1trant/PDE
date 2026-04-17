"""
Evaluate from-scratch full-param checkpoint (PDEModelV3) on validation set.

Reports per-channel RMSE, VRMSE, nRMSE + overall + PDE loss.
Works for all 2D datasets by specifying pde_type in config.

Usage:
    torchrun --nproc_per_node=4 tools/eval_scratch.py \
        --config configs/finetune_taylor_green_2d_v3_scratch.yaml \
        --checkpoint checkpoints_taylor_green_2d_lora_v3_scratch/best_scratch.pt
"""

import os
import sys
import warnings

if os.environ.get('LOCAL_RANK', '0') != '0':
    warnings.filterwarnings('ignore')

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from accelerate import Accelerator
from pretrain.model_v3 import PDEModelV3
from finetune.dataset_finetune import (
    FinetuneDataset, FinetuneSampler, finetune_collate_fn,
    create_finetune_dataloaders,
)

torch.set_float32_matmul_precision('high')


# ============================================================
# Channel mappings per pde_type
# ============================================================
CHANNEL_MAPS = {
    'taylor_green_2d': {
        'names': ['Vx', 'Vy', 'pressure'],
        'indices': [0, 1, 15],  # in 18-ch representation
    },
    'wave_2d': {
        'names': ['u', 'w'],
        'indices': [3, 4],  # scalar[0]=ch3, scalar[1]=ch4
    },
    'advdiff_2d': {
        'names': ['u'],
        'indices': [3],  # scalar[0]=ch3
    },
    'burgers_2d_ch': {
        'names': ['Vx', 'Vy'],
        'indices': [0, 1],
    },
}


# ============================================================
# PDE Loss (reuse from train_scratch_fullparam)
# ============================================================
def create_pde_loss(config: dict, device: torch.device):
    import inspect
    from finetune.pde_loss_verified import (
        TaylorGreen2DPDELoss, KP2DPDELoss, Wave2DPDELoss,
        AdvDiff2DPDELoss, Burgers2DCHPDELoss,
    )
    registry = {
        'taylor_green_2d': TaylorGreen2DPDELoss,
        'wave_2d': Wave2DPDELoss,
        'advdiff_2d': AdvDiff2DPDELoss,
        'burgers_2d_ch': Burgers2DCHPDELoss,
    }
    pde_type = config.get('pde_type', '')
    physics = config.get('physics', {})
    cls = registry[pde_type]

    meta_keys = {'eq_scales', 'eq_weights', 'eq_scales_per_t_path', 'pde_type'}
    valid_params = set(inspect.signature(cls.__init__).parameters.keys()) - {'self'}
    kwargs = {k: v for k, v in physics.items() if k in valid_params and k not in meta_keys}
    # No eq_scales for eval (raw PDE residual)
    return cls(**kwargs).to(device)


def compute_pde_loss(output, input_data, pde_loss_fn, config, batch):
    pde_type = config.get('pde_type', '')
    with torch.autocast(device_type='cuda', enabled=False):
        if pde_type == 'taylor_green_2d':
            CH_VX, CH_VY, CH_PRESS = 0, 1, 15
            t0_u = input_data[:, 0:1, :, :, CH_VX].float()
            t0_v = input_data[:, 0:1, :, :, CH_VY].float()
            t0_p = input_data[:, 0:1, :, :, CH_PRESS].float()
            u = torch.cat([t0_u, output[:, :, :, :, CH_VX].float()], dim=1)
            v = torch.cat([t0_v, output[:, :, :, :, CH_VY].float()], dim=1)
            p = torch.cat([t0_p, output[:, :, :, :, CH_PRESS].float()], dim=1)
            nu = batch.get('nu', None)
            if nu is not None:
                nu = nu.to(output.device)
            total, _ = pde_loss_fn(u, v, p, nu=nu)

        elif pde_type in ('advdiff_2d',):
            CH_U = 3
            t0_u = input_data[:, 0:1, :, :, CH_U].float()
            u = torch.cat([t0_u, output[:, :, :, :, CH_U].float()], dim=1)
            total, _ = pde_loss_fn(u)

        elif pde_type == 'wave_2d':
            CH_U, CH_W = 3, 4
            t0_u = input_data[:, 0:1, :, :, CH_U].float()
            t0_w = input_data[:, 0:1, :, :, CH_W].float()
            u = torch.cat([t0_u, output[:, :, :, :, CH_U].float()], dim=1)
            w = torch.cat([t0_w, output[:, :, :, :, CH_W].float()], dim=1)
            c = batch['nu'].to(output.device)
            total, _ = pde_loss_fn(u, w, c)

        elif pde_type == 'burgers_2d_ch':
            CH_VX, CH_VY = 0, 1
            t0_u = input_data[:, 0:1, :, :, CH_VX].float()
            t0_v = input_data[:, 0:1, :, :, CH_VY].float()
            u = torch.cat([t0_u, output[:, :, :, :, CH_VX].float()], dim=1)
            v = torch.cat([t0_v, output[:, :, :, :, CH_VY].float()], dim=1)
            nu = batch['nu'].to(output.device)
            total, _ = pde_loss_fn(u, v, nu=nu)

        else:
            raise ValueError(f"Unknown pde_type: {pde_type}")

    return total


@torch.no_grad()
def evaluate(accelerator, model, val_loader, config, t_input):
    accelerator.wait_for_everyone()
    model.eval()

    pde_type = config.get('pde_type', '')
    ch_map = CHANNEL_MAPS[pde_type]
    ch_names = ch_map['names']
    ch_indices = ch_map['indices']
    n_ch = len(ch_names)

    pde_loss_fn = create_pde_loss(config, accelerator.device)

    max_batches = len(val_loader)
    local_pde = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_rmse = torch.full((max_batches, n_ch), float('nan'), device=accelerator.device)
    local_vrmse = torch.full((max_batches, n_ch), float('nan'), device=accelerator.device)
    local_nrmse = torch.full((max_batches, n_ch), float('nan'), device=accelerator.device)
    local_rmse_all = torch.full((max_batches,), float('nan'), device=accelerator.device)

    for i, batch in enumerate(val_loader):
        data = batch['data'].to(device=accelerator.device, dtype=torch.float32)
        input_data = data[:, :t_input]
        target_data = data[:, 1:t_input + 1]

        output_norm, mean, std = model(input_data, return_normalized=True)
        output = output_norm * std + mean

        # PDE loss
        pde = compute_pde_loss(output, input_data, pde_loss_fn, config, batch)
        local_pde[i] = pde.detach()

        # Per-channel metrics
        for c_idx, ch in enumerate(ch_indices):
            pred_c = output[..., ch]
            gt_c = target_data[..., ch]
            mse_c = torch.mean((pred_c - gt_c) ** 2)
            rmse_c = torch.sqrt(mse_c + 1e-8)
            var_c = torch.mean((gt_c - gt_c.mean()) ** 2)
            vrmse_c = torch.sqrt(mse_c / (var_c + 1e-8))
            range_c = gt_c.max() - gt_c.min()
            nrmse_c = rmse_c / (range_c + 1e-8)

            local_rmse[i, c_idx] = rmse_c
            local_vrmse[i, c_idx] = vrmse_c
            local_nrmse[i, c_idx] = nrmse_c

        # Overall RMSE (across valid channels)
        all_pred = torch.cat([output[..., ch:ch+1] for ch in ch_indices], dim=-1)
        all_gt = torch.cat([target_data[..., ch:ch+1] for ch in ch_indices], dim=-1)
        local_rmse_all[i] = torch.sqrt(torch.mean((all_pred - all_gt) ** 2) + 1e-8)

    # Gather across GPUs
    accelerator.wait_for_everyone()

    all_pde = accelerator.gather(local_pde)
    all_rmse = accelerator.gather(local_rmse)
    all_vrmse = accelerator.gather(local_vrmse)
    all_nrmse = accelerator.gather(local_nrmse)
    all_rmse_total = accelerator.gather(local_rmse_all)

    if accelerator.is_main_process:
        # Remove NaN padding
        valid_pde = all_pde[~torch.isnan(all_pde)]
        valid_rmse_all = all_rmse_total[~torch.isnan(all_rmse_total)]

        print(f"\n{'='*70}")
        print(f"  Evaluation Results: {pde_type}")
        print(f"{'='*70}")
        print(f"  Batches evaluated: {len(valid_pde)}")
        print(f"\n  Overall:")
        print(f"    RMSE (all ch): {valid_rmse_all.mean().item():.6f}")

        vrmse_per_ch = []
        nrmse_per_ch = []
        for c_idx, name in enumerate(ch_names):
            valid_rmse_c = all_rmse[:, c_idx][~torch.isnan(all_rmse[:, c_idx])]
            valid_vrmse_c = all_vrmse[:, c_idx][~torch.isnan(all_vrmse[:, c_idx])]
            valid_nrmse_c = all_nrmse[:, c_idx][~torch.isnan(all_nrmse[:, c_idx])]
            print(f"\n  Channel: {name}")
            print(f"    RMSE:  {valid_rmse_c.mean().item():.6f}")
            print(f"    VRMSE: {valid_vrmse_c.mean().item():.6f}")
            print(f"    nRMSE: {valid_nrmse_c.mean().item():.6f}")
            vrmse_per_ch.append(valid_vrmse_c.mean().item())
            nrmse_per_ch.append(valid_nrmse_c.mean().item())

        avg_vrmse = np.mean(vrmse_per_ch)
        avg_nrmse = np.mean(nrmse_per_ch)
        print(f"\n  Summary:")
        print(f"    VRMSE (avg): {avg_vrmse:.6f}")
        print(f"    nRMSE (avg): {avg_nrmse:.6f}")
        print(f"    RMSE  (all): {valid_rmse_all.mean().item():.6f}")
        print(f"    PDE loss:    {valid_pde.mean().item():.6f}")
        print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    t_input = config['dataset'].get('t_input', 8)
    temporal_length = t_input + 1

    accelerator = Accelerator(mixed_precision='no')
    is_main = accelerator.is_main_process

    # Load model
    model = PDEModelV3(config)

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=True)
    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Loaded scratch checkpoint: {args.checkpoint}")
        print(f"Model params: {n_params:,}")
        if 'global_step' in ckpt:
            print(f"Global step: {ckpt['global_step']}")
        if 'val_vrmse' in ckpt:
            print(f"Saved val_vrmse: {ckpt['val_vrmse']:.6f}")

    model = model.float()

    # Create val dataloader
    _, val_loader, _, _ = create_finetune_dataloaders(
        data_path=config['dataset']['path'],
        batch_size=config['dataloader']['batch_size'],
        num_workers=config['dataloader']['num_workers'],
        pin_memory=config['dataloader']['pin_memory'],
        seed=config['dataset']['seed'],
        temporal_length=temporal_length,
        train_ratio=config['dataset'].get('train_ratio', 0.9),
        clips_per_sample=config['dataset'].get('clips_per_sample'),
        vector_dim=config['dataset'].get('vector_dim', 2),
        val_time_interval=config['dataset'].get('val_time_interval', 20),
    )

    model = accelerator.prepare(model)
    evaluate(accelerator, model, val_loader, config, t_input)


if __name__ == '__main__':
    main()
