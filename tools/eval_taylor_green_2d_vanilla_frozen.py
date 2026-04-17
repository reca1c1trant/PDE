"""
Muon-LoRA Evaluation for Taylor-Green 2D.

Same as visualize_taylor_green_2d_lora.py but uses PDELoRAModelMuon.

Usage:
    torchrun --nproc_per_node=4 tools/eval_taylor_green_2d_muon.py \
        --config configs/finetune_taylor_green_2d_v3_muon.yaml \
        --checkpoint checkpoints_taylor_green_2d_muon_lora/best_lora.pt --scan_all
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

from torch.utils.data import DataLoader
from accelerate import Accelerator

from finetune.dataset_finetune import (
    FinetuneDataset, FinetuneSampler, finetune_collate_fn,
)
from finetune.model_lora_v3 import PDELoRAModelV3 as PDELoRAModelMuon, load_lora_checkpoint
from finetune.pde_loss_verified import TaylorGreen2DPDELoss

CH_VX = 0
CH_VY = 1
CH_PRESS = 15


def _vrmse_torch(gt: torch.Tensor, pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse = torch.mean((pred - gt) ** 2)
    var = torch.mean((gt - gt.mean()) ** 2)
    return torch.sqrt(mse / (var + eps))


def _nrmse_torch(gt: torch.Tensor, pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse_pred = torch.mean((pred - gt) ** 2)
    mse_zero = torch.mean(gt ** 2)
    return torch.sqrt(mse_pred / (mse_zero + eps))


def parse_args():
    parser = argparse.ArgumentParser(description="Muon-LoRA Eval for Taylor-Green 2D")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--scan_all', action='store_true')
    parser.add_argument('--t_input', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model(config: dict, checkpoint_path: str, is_main: bool = True):
    """Load Muon-LoRA model from checkpoint."""
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

    pretrained_path = config.get('model', {}).get('pretrained_path')

    model = PDELoRAModelMuon(
        config=config,
        pretrained_path=pretrained_path,
        freeze_encoder=config.get('model', {}).get('freeze_encoder', False),
        freeze_decoder=config.get('model', {}).get('freeze_decoder', False),
    )

    # Load encoder/decoder from init_from if specified (for frozen enc/dec setup)
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
                print(f"  Loaded {loaded} enc/dec keys from {init_from}")
        del init_ckpt

    checkpoint = load_lora_checkpoint(model, checkpoint_path)

    if is_main:
        if 'metrics' in checkpoint:
            print(f"  Checkpoint metrics: {checkpoint['metrics']}")
        if 'global_step' in checkpoint:
            print(f"  Global step: {checkpoint['global_step']}")

    model = model.float()
    return model


def create_pde_loss_fn(config: dict) -> TaylorGreen2DPDELoss:
    physics = config.get('physics', {})
    return TaylorGreen2DPDELoss(
        nx=physics.get('nx', 256),
        ny=physics.get('ny', 256),
        Lx=physics.get('Lx', 6.283185307179586),
        Ly=physics.get('Ly', 6.283185307179586),
        dt=physics.get('dt', 0.01),
    )


def compute_pde_loss_from_output(output, input_data, pde_loss_fn, nu):
    with torch.autocast(device_type='cuda', enabled=False):
        t0_u = input_data[:, 0:1, :, :, CH_VX].float()
        t0_v = input_data[:, 0:1, :, :, CH_VY].float()
        t0_p = input_data[:, 0:1, :, :, CH_PRESS].float()
        u = torch.cat([t0_u, output[:, :, :, :, CH_VX].float()], dim=1)
        v = torch.cat([t0_v, output[:, :, :, :, CH_VY].float()], dim=1)
        p = torch.cat([t0_p, output[:, :, :, :, CH_PRESS].float()], dim=1)
        total_loss, _ = pde_loss_fn(u, v, p, nu=nu.float())
    return total_loss


@torch.no_grad()
def scan_all_distributed(accelerator, model, val_loader, config, t_input):
    accelerator.wait_for_everyone()
    model.eval()
    pde_loss_fn = create_pde_loss_fn(config)

    max_batches = len(val_loader)
    local_pde = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_vrmse_vx = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_vrmse_vy = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_vrmse_press = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_rmse = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_vrmse_all = torch.full((max_batches,), float('nan'), device=accelerator.device)

    for i, batch in enumerate(val_loader):
        data = batch['data'].to(device=accelerator.device, dtype=torch.float32)
        channel_mask = batch['channel_mask'].to(device=accelerator.device)
        nu = batch['nu'].to(device=accelerator.device)

        input_data = data[:, :t_input]
        target_data = data[:, 1:t_input + 1]

        output_norm, mean, std = model(input_data, return_normalized=True)
        output = output_norm * std + mean

        pde_loss = compute_pde_loss_from_output(output, input_data, pde_loss_fn, nu)

        vrmse_vx = _vrmse_torch(target_data[..., CH_VX], output[..., CH_VX])
        vrmse_vy = _vrmse_torch(target_data[..., CH_VY], output[..., CH_VY])
        vrmse_press = _vrmse_torch(target_data[..., CH_PRESS], output[..., CH_PRESS])

        valid_ch = (torch.where(channel_mask[0] > 0)[0] if channel_mask.dim() > 1
                    else torch.where(channel_mask > 0)[0])
        mse = torch.mean((output[..., valid_ch] - target_data[..., valid_ch]) ** 2)
        rmse = torch.sqrt(mse + 1e-8)

        vrmse_all = (vrmse_vx + vrmse_vy + vrmse_press) / 3

        local_pde[i] = pde_loss.detach()
        local_vrmse_vx[i] = vrmse_vx.detach()
        local_vrmse_vy[i] = vrmse_vy.detach()
        local_vrmse_press[i] = vrmse_press.detach()
        local_rmse[i] = rmse.detach()
        local_vrmse_all[i] = vrmse_all.detach()

    accelerator.wait_for_everyone()
    all_pde = accelerator.gather(local_pde)
    all_vrmse_vx = accelerator.gather(local_vrmse_vx)
    all_vrmse_vy = accelerator.gather(local_vrmse_vy)
    all_vrmse_press = accelerator.gather(local_vrmse_press)
    all_rmse = accelerator.gather(local_rmse)
    all_vrmse_all = accelerator.gather(local_vrmse_all)
    accelerator.wait_for_everyone()

    valid_mask = ~torch.isnan(all_pde)
    n = valid_mask.sum().item()
    if n > 0:
        return {
            'pde': float(all_pde[valid_mask].mean()),
            'rmse': float(all_rmse[valid_mask].mean()),
            'vrmse_vx': float(all_vrmse_vx[valid_mask].mean()),
            'vrmse_vy': float(all_vrmse_vy[valid_mask].mean()),
            'vrmse_press': float(all_vrmse_press[valid_mask].mean()),
            'vrmse_all': float(all_vrmse_all[valid_mask].mean()),
            'num_batches': n,
        }
    return {'pde': 0, 'rmse': 0, 'vrmse_vx': 0, 'vrmse_vy': 0,
            'vrmse_press': 0, 'vrmse_all': 0, 'num_batches': 0}


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
        print(f"Taylor-Green 2D Muon-LoRA Evaluation")
        print(f"{'='*60}")
        print(f"  Devices: {accelerator.num_processes}")
        print(f"  Checkpoint: {args.checkpoint}")

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

        if is_main:
            print(f"\n{'='*60}")
            print(f"Results ({results['num_batches']} batches):")
            print(f"  PDE Loss:      {results['pde']:.6f}")
            print(f"  RMSE:          {results['rmse']:.6f}")
            print(f"  VRMSE (Vx):    {results['vrmse_vx']:.6f}")
            print(f"  VRMSE (Vy):    {results['vrmse_vy']:.6f}")
            print(f"  VRMSE (press): {results['vrmse_press']:.6f}")
            print(f"  VRMSE (all):   {results['vrmse_all']:.6f}")
            print(f"{'='*60}")
    else:
        print("Use --scan_all for evaluation")


if __name__ == "__main__":
    main()
