"""
Generic Split-NS / Muon-LoRA evaluation for all 2D datasets.
Supports: taylor_green_2d, burgers_2d_ch, advdiff_2d, wave_2d.

Usage:
    torchrun --nproc_per_node=4 tools/eval_splitns.py \
        --config configs/finetune_taylor_green_2d_splitns.yaml \
        --checkpoint checkpoints_taylor_green_2d_splitns/best_lora.pt --scan_all
"""

import argparse
import yaml
import torch
import inspect
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from accelerate import Accelerator

from finetune.dataset_finetune import (
    FinetuneDataset, FinetuneSampler, finetune_collate_fn,
)
from finetune.model_lora_v3 import PDELoRAModelV3 as PDELoRAModelMuon, load_lora_checkpoint


def _vrmse(gt: torch.Tensor, pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse = torch.mean((pred - gt) ** 2)
    var = torch.mean((gt - gt.mean()) ** 2)
    return torch.sqrt(mse / (var + eps))


def _nrmse(gt: torch.Tensor, pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse_pred = torch.mean((pred - gt) ** 2)
    mse_zero = torch.mean(gt ** 2)
    return torch.sqrt(mse_pred / (mse_zero + eps))


# ============================================================
# Dataset-specific channel definitions
# ============================================================

DATASET_CHANNELS = {
    'taylor_green_2d': {
        'channels': {'Vx': 0, 'Vy': 1, 'press': 15},
        'vector_dim': 2,
    },
    'burgers_2d_ch': {
        'channels': {'Vx': 0, 'Vy': 1},
        'vector_dim': 2,
    },
    'advdiff_2d': {
        'channels': {'u': 3},
        'vector_dim': 0,
    },
    'wave_2d': {
        'channels': {'u': 3, 'w': 4},
        'vector_dim': 0,
    },
}

# ============================================================
# PDE loss creation
# ============================================================

PDE_LOSS_REGISTRY = {}

def _lazy_import_pde_losses():
    global PDE_LOSS_REGISTRY
    if PDE_LOSS_REGISTRY:
        return
    from finetune.pde_loss_verified import (
        TaylorGreen2DPDELoss, KP2DPDELoss, Wave2DPDELoss,
        AdvDiff2DPDELoss, Burgers2DCHPDELoss,
    )
    PDE_LOSS_REGISTRY = {
        'taylor_green_2d': TaylorGreen2DPDELoss,
        'wave_2d': Wave2DPDELoss,
        'advdiff_2d': AdvDiff2DPDELoss,
        'burgers_2d_ch': Burgers2DCHPDELoss,
    }


def create_pde_loss(config: dict, device: torch.device):
    _lazy_import_pde_losses()
    pde_type = config.get('pde_type', '')
    physics = config.get('physics', {})
    cls = PDE_LOSS_REGISTRY[pde_type]
    meta_keys = {'eq_scales', 'eq_weights', 'eq_scales_per_t_path', 'pde_type'}
    valid_params = set(inspect.signature(cls.__init__).parameters.keys()) - {'self'}
    kwargs = {k: v for k, v in physics.items() if k in valid_params and k not in meta_keys}
    return cls(**kwargs).to(device)


def compute_pde_loss_generic(output, input_data, pde_loss_fn, pde_type, batch):
    with torch.autocast(device_type='cuda', enabled=False):
        if pde_type == 'taylor_green_2d':
            t0_u = input_data[:, 0:1, :, :, 0].float()
            t0_v = input_data[:, 0:1, :, :, 1].float()
            t0_p = input_data[:, 0:1, :, :, 15].float()
            u = torch.cat([t0_u, output[:, :, :, :, 0].float()], dim=1)
            v = torch.cat([t0_v, output[:, :, :, :, 1].float()], dim=1)
            p = torch.cat([t0_p, output[:, :, :, :, 15].float()], dim=1)
            nu = batch.get('nu', None)
            if nu is not None:
                nu = nu.to(output.device)
            total, _ = pde_loss_fn(u, v, p, nu=nu)
        elif pde_type in ('advdiff_2d',):
            t0 = input_data[:, 0:1, :, :, 3].float()
            u = torch.cat([t0, output[:, :, :, :, 3].float()], dim=1)
            total, _ = pde_loss_fn(u)
        elif pde_type == 'wave_2d':
            t0_u = input_data[:, 0:1, :, :, 3].float()
            t0_w = input_data[:, 0:1, :, :, 4].float()
            u = torch.cat([t0_u, output[:, :, :, :, 3].float()], dim=1)
            w = torch.cat([t0_w, output[:, :, :, :, 4].float()], dim=1)
            c = batch['nu'].to(output.device)
            total, _ = pde_loss_fn(u, w, c)
        elif pde_type == 'burgers_2d_ch':
            t0_u = input_data[:, 0:1, :, :, 0].float()
            t0_v = input_data[:, 0:1, :, :, 1].float()
            u = torch.cat([t0_u, output[:, :, :, :, 0].float()], dim=1)
            v = torch.cat([t0_v, output[:, :, :, :, 1].float()], dim=1)
            nu = batch['nu'].to(output.device)
            total, _ = pde_loss_fn(u, v, nu=nu)
        else:
            raise ValueError(f"Unknown pde_type: {pde_type}")
    return total


# ============================================================
# Args / Model loading
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Split-NS / Muon-LoRA Eval (all datasets)")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--scan_all', action='store_true')
    parser.add_argument('--t_input', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    return parser.parse_args()


def load_model(config: dict, checkpoint_path: str, is_main: bool = True):
    ckpt_probe = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'config' in ckpt_probe:
        ckpt_model_cfg = ckpt_probe['config'].get('model', {})
        arch_keys = ['decoder', 'patch_smoother', 'encoder', 'intra_patch', 'na',
                     'in_channels', 'hidden_dim', 'patch_size', 'num_layers', 'num_heads',
                     'vector_channels', 'scalar_channels', 'enable_1d', 'enable_3d']
        model_cfg = dict(config.get('model', {}))
        for k in arch_keys:
            if k in ckpt_model_cfg:
                model_cfg[k] = ckpt_model_cfg[k]
        config = {**config, 'model': model_cfg}
    del ckpt_probe

    model = PDELoRAModelMuon(
        config=config,
        pretrained_path=config.get('model', {}).get('pretrained_path'),
        freeze_encoder=config.get('model', {}).get('freeze_encoder', False),
        freeze_decoder=config.get('model', {}).get('freeze_decoder', False),
    )

    # Load enc/dec + LoRA from init_from (direct match for PEFT vanilla)
    init_from = config.get('model', {}).get('init_from', None)
    if init_from:
        init_ckpt = torch.load(init_from, map_location='cpu', weights_only=False)
        if 'trainable_state_dict' in init_ckpt:
            src_state = init_ckpt['trainable_state_dict']
            model_state = model.model.state_dict()
            loaded = 0
            for k, v in src_state.items():
                if k in model_state and model_state[k].shape == v.shape:
                    model_state[k] = v
                    loaded += 1
            model.model.load_state_dict(model_state)
            if is_main:
                print(f"  Loaded {loaded} keys from {init_from}")
        del init_ckpt

    checkpoint = load_lora_checkpoint(model, checkpoint_path)
    if is_main:
        if 'metrics' in checkpoint:
            print(f"  Checkpoint metrics: {checkpoint['metrics']}")
    model = model.float()
    return model


# ============================================================
# Distributed scan_all
# ============================================================

@torch.no_grad()
def scan_all_distributed(accelerator, model, val_loader, config, t_input):
    accelerator.wait_for_everyone()
    model.eval()

    pde_type = config.get('pde_type', '')
    ch_info = DATASET_CHANNELS[pde_type]
    ch_names = list(ch_info['channels'].keys())
    ch_indices = list(ch_info['channels'].values())

    pde_loss_fn = create_pde_loss(config, accelerator.device)

    max_batches = len(val_loader)
    local_pde = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_vrmse = {name: torch.full((max_batches,), float('nan'), device=accelerator.device)
                   for name in ch_names}
    local_nrmse = {name: torch.full((max_batches,), float('nan'), device=accelerator.device)
                   for name in ch_names}
    local_rmse = {name: torch.full((max_batches,), float('nan'), device=accelerator.device)
                  for name in ch_names}

    for i, batch in enumerate(val_loader):
        data = batch['data'].to(device=accelerator.device, dtype=torch.float32)
        channel_mask = batch['channel_mask'].to(device=accelerator.device)

        input_data = data[:, :t_input]
        target_data = data[:, 1:t_input + 1]

        output_norm, mean, std = model(input_data, return_normalized=True)
        output = output_norm * std + mean

        pde_loss = compute_pde_loss_generic(output, input_data, pde_loss_fn, pde_type, batch)
        local_pde[i] = pde_loss.detach()

        for name, ch_idx in zip(ch_names, ch_indices):
            pred_ch = output[..., ch_idx]
            gt_ch = target_data[..., ch_idx]
            local_rmse[name][i] = torch.sqrt(torch.mean((pred_ch - gt_ch) ** 2) + 1e-8).detach()
            local_vrmse[name][i] = _vrmse(gt_ch, pred_ch).detach()
            local_nrmse[name][i] = _nrmse(gt_ch, pred_ch).detach()

    accelerator.wait_for_everyone()
    all_pde = accelerator.gather(local_pde)
    all_vrmse = {n: accelerator.gather(v) for n, v in local_vrmse.items()}
    all_nrmse = {n: accelerator.gather(v) for n, v in local_nrmse.items()}
    all_rmse = {n: accelerator.gather(v) for n, v in local_rmse.items()}
    accelerator.wait_for_everyone()

    valid_mask = ~torch.isnan(all_pde)
    n = valid_mask.sum().item()
    if n == 0:
        return {}

    results = {
        'pde': float(all_pde[valid_mask].mean()),
        'num_batches': n,
    }

    vrmse_vals = []
    nrmse_vals = []
    for name in ch_names:
        v = float(all_vrmse[name][valid_mask].mean())
        nr = float(all_nrmse[name][valid_mask].mean())
        r = float(all_rmse[name][valid_mask].mean())
        results[f'vrmse_{name}'] = v
        results[f'nrmse_{name}'] = nr
        results[f'rmse_{name}'] = r
        vrmse_vals.append(v)
        nrmse_vals.append(nr)

    results['vrmse_all'] = sum(vrmse_vals) / len(vrmse_vals)
    results['nrmse_all'] = sum(nrmse_vals) / len(nrmse_vals)

    return results


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config))
    pde_type = config.get('pde_type', 'unknown')
    t_input = args.t_input or config.get('dataset', {}).get('t_input', 8)
    temporal_length = t_input + 1
    batch_size = args.batch_size or config.get('dataloader', {}).get('batch_size', 4)

    accelerator = Accelerator()
    is_main = accelerator.is_main_process

    if is_main:
        print(f"{'='*60}")
        print(f"Vanilla LoRA Eval — {pde_type}")
        print(f"{'='*60}")
        print(f"  Checkpoint: {args.checkpoint}")

    model = load_model(config, args.checkpoint, is_main)

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
        print(f"  Val clips: {len(val_dataset)}")

    if args.scan_all:
        val_sampler = FinetuneSampler(
            val_dataset, batch_size, shuffle=False,
            seed=config['dataset'].get('seed', 42),
        )
        val_loader = DataLoader(
            val_dataset, batch_sampler=val_sampler,
            collate_fn=finetune_collate_fn,
            num_workers=config.get('dataloader', {}).get('num_workers', 4),
            pin_memory=True,
        )

        model, val_loader = accelerator.prepare(model, val_loader)
        results = scan_all_distributed(accelerator, model, val_loader, config, t_input)

        if is_main and results:
            print(f"\n{'='*60}")
            print(f"Results ({results['num_batches']} batches) — {pde_type}:")
            print(f"  PDE Loss:      {results['pde']:.6f}")
            ch_names = list(DATASET_CHANNELS[pde_type]['channels'].keys())
            for name in ch_names:
                print(f"  VRMSE ({name:>5s}): {results[f'vrmse_{name}']:.6f}")
                print(f"  nRMSE ({name:>5s}): {results[f'nrmse_{name}']:.6f}")
                print(f"  RMSE  ({name:>5s}): {results[f'rmse_{name}']:.6f}")
            print(f"  VRMSE (all):   {results['vrmse_all']:.6f}")
            print(f"  nRMSE (all):   {results['nrmse_all']:.6f}")
            print(f"{'='*60}")
    else:
        print("Use --scan_all for evaluation")


if __name__ == "__main__":
    main()
