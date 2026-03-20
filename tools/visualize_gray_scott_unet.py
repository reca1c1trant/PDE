"""
UNet Baseline Evaluation for Gray-Scott.

Output format matches visualize_gray_scott_lora.py --scan_all.

Usage:
    torchrun --nproc_per_node=4 tools/visualize_gray_scott_unet.py \
        --config configs/finetune_gray_scott_unet.yaml \
        --checkpoint checkpoints_gray_scott_unet/best_unet.pt --scan_all
"""

import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

from torch.utils.data import DataLoader
from accelerate import Accelerator

from finetune.dataset_finetune import (
    FinetuneDataset, FinetuneSampler, finetune_collate_fn,
)
from finetune.unet_gray_scott import UNetGrayScott, CH_A, CH_B
from finetune.pde_loss_verified import GrayScottPDELoss


def _vrmse_torch(gt: torch.Tensor, pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse = torch.mean((pred - gt) ** 2)
    var = torch.mean((gt - gt.mean()) ** 2)
    return torch.sqrt(mse / (var + eps))


def _nrmse_torch(gt: torch.Tensor, pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse_pred = torch.mean((pred - gt) ** 2)
    mse_zero = torch.mean(gt ** 2)
    return torch.sqrt(mse_pred / (mse_zero + eps))


def _vrmse_np(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-8) -> float:
    mse = np.mean((pred - gt) ** 2)
    var = np.mean((gt - gt.mean()) ** 2)
    return float(np.sqrt(mse / (var + eps)))


def parse_args():
    parser = argparse.ArgumentParser(description="UNet Baseline Evaluation for Gray-Scott")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./vis_results')
    parser.add_argument('--num_samples', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--scan_all', action='store_true')
    parser.add_argument('--t_input', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model(config: dict, checkpoint_path: str, is_main: bool = True) -> UNetGrayScott:
    init_features = config.get('model', {}).get('init_features', 32)
    model = UNetGrayScott(init_features=init_features)

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))

    cleaned = {}
    for k, v in state_dict.items():
        k = k.removeprefix('module.').removeprefix('_orig_mod.')
        cleaned[k] = v

    model.load_state_dict(cleaned, strict=False)

    if is_main:
        if 'metrics' in ckpt:
            print(f"  Checkpoint metrics: {ckpt['metrics']}")
        if 'global_step' in ckpt:
            print(f"  Global step: {ckpt['global_step']}")
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  UNet params: {n_params:,}")

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
    with torch.autocast(device_type='cuda', enabled=False):
        t0_A = input_data[:, 0:1, :, :, CH_A].float()
        t0_B = input_data[:, 0:1, :, :, CH_B].float()
        out_A = output[:, :, :, :, CH_A].float()
        out_B = output[:, :, :, :, CH_B].float()
        A = torch.cat([t0_A, out_A], dim=1)
        B = torch.cat([t0_B, out_B], dim=1)
        total_loss, losses = pde_loss_fn(A, B)
    return total_loss


@torch.no_grad()
def scan_all_distributed(accelerator, model, val_loader, config, t_input):
    accelerator.wait_for_everyone()
    model.eval()

    pde_loss_fn = create_pde_loss_fn(config)

    max_batches = len(val_loader)
    local_pde = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_rmse_A = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_rmse_B = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_rmse = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_vrmse_A = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_vrmse_B = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_nrmse_A = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_nrmse_B = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_vrmse_all = torch.full((max_batches,), float('nan'), device=accelerator.device)
    local_nrmse_all = torch.full((max_batches,), float('nan'), device=accelerator.device)

    for i, batch in enumerate(val_loader):
        data = batch['data'].to(device=accelerator.device, dtype=torch.float32)
        channel_mask = batch['channel_mask'].to(device=accelerator.device)

        input_data = data[:, :t_input]
        target_data = data[:, 1:t_input + 1]

        # UNet forward: direct output (no normalization)
        output = model(input_data)

        pde_loss = compute_pde_loss_from_output(output, input_data, pde_loss_fn)

        rmse_A = torch.sqrt(torch.mean((output[..., CH_A] - target_data[..., CH_A]) ** 2) + 1e-8)
        rmse_B = torch.sqrt(torch.mean((output[..., CH_B] - target_data[..., CH_B]) ** 2) + 1e-8)

        vrmse_A = _vrmse_torch(target_data[..., CH_A], output[..., CH_A])
        vrmse_B = _vrmse_torch(target_data[..., CH_B], output[..., CH_B])

        nrmse_A = _nrmse_torch(target_data[..., CH_A], output[..., CH_A])
        nrmse_B = _nrmse_torch(target_data[..., CH_B], output[..., CH_B])

        valid_ch = (
            torch.where(channel_mask[0] > 0)[0]
            if channel_mask.dim() > 1
            else torch.where(channel_mask > 0)[0]
        )
        mse = torch.mean((output[..., valid_ch] - target_data[..., valid_ch]) ** 2)
        rmse = torch.sqrt(mse + 1e-8)

        output_valid = output[..., valid_ch]
        target_valid = target_data[..., valid_ch]
        mse_all = torch.mean((output_valid - target_valid) ** 2)
        var_all = torch.mean((target_valid - target_valid.mean()) ** 2)
        vrmse_all = torch.sqrt(mse_all / (var_all + 1e-8))
        mse_zero_all = torch.mean(target_valid ** 2)
        nrmse_all = torch.sqrt(mse_all / (mse_zero_all + 1e-8))

        local_pde[i] = pde_loss.detach()
        local_rmse_A[i] = rmse_A.detach()
        local_rmse_B[i] = rmse_B.detach()
        local_rmse[i] = rmse.detach()
        local_vrmse_A[i] = vrmse_A.detach()
        local_vrmse_B[i] = vrmse_B.detach()
        local_nrmse_A[i] = nrmse_A.detach()
        local_nrmse_B[i] = nrmse_B.detach()
        local_vrmse_all[i] = vrmse_all.detach()
        local_nrmse_all[i] = nrmse_all.detach()

    accelerator.wait_for_everyone()
    all_pde = accelerator.gather(local_pde)
    all_rmse_A = accelerator.gather(local_rmse_A)
    all_rmse_B = accelerator.gather(local_rmse_B)
    all_rmse = accelerator.gather(local_rmse)
    all_vrmse_A = accelerator.gather(local_vrmse_A)
    all_vrmse_B = accelerator.gather(local_vrmse_B)
    all_nrmse_A = accelerator.gather(local_nrmse_A)
    all_nrmse_B = accelerator.gather(local_nrmse_B)
    all_vrmse_all = accelerator.gather(local_vrmse_all)
    all_nrmse_all = accelerator.gather(local_nrmse_all)
    accelerator.wait_for_everyone()

    valid_mask = ~torch.isnan(all_pde)
    n = valid_mask.sum().item()

    if n > 0:
        return {
            'pde': float(all_pde[valid_mask].cpu().numpy().mean()),
            'rmse_A': float(all_rmse_A[valid_mask].cpu().numpy().mean()),
            'rmse_B': float(all_rmse_B[valid_mask].cpu().numpy().mean()),
            'rmse': float(all_rmse[valid_mask].cpu().numpy().mean()),
            'vrmse_A': float(all_vrmse_A[valid_mask].cpu().numpy().mean()),
            'vrmse_B': float(all_vrmse_B[valid_mask].cpu().numpy().mean()),
            'nrmse_A': float(all_nrmse_A[valid_mask].cpu().numpy().mean()),
            'nrmse_B': float(all_nrmse_B[valid_mask].cpu().numpy().mean()),
            'vrmse_all': float(all_vrmse_all[valid_mask].cpu().numpy().mean()),
            'nrmse_all': float(all_nrmse_all[valid_mask].cpu().numpy().mean()),
            'num_batches': n,
        }
    return {'pde': 0, 'rmse_A': 0, 'rmse_B': 0, 'rmse': 0,
            'vrmse_A': 0, 'vrmse_B': 0, 'nrmse_A': 0, 'nrmse_B': 0,
            'vrmse_all': 0, 'nrmse_all': 0, 'num_batches': 0}


@torch.no_grad()
def run_visualization(model, dataset, config, device, t_input, num_samples, seed, output_dir):
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
    clip_offsets = (np.linspace(0, clips_per_sample - 1, n_vis, dtype=int)
                    if clips_per_sample > 1 and n_vis > 1
                    else np.zeros(n_vis, dtype=int))

    print(f"Visualizing {n_vis} samples...")

    results = []
    all_metrics: Dict[str, list] = {
        'rmse_A': [], 'rmse_B': [],
        'vrmse_A': [], 'vrmse_B': [],
        'nrmse_A': [], 'nrmse_B': [],
        'vrmse_all': [], 'nrmse_all': [],
        'pde': [],
    }

    for i, sample_idx in enumerate(sample_indices):
        idx = min(sample_idx * clips_per_sample + clip_offsets[i], total_clips - 1)

        sample = dataset[idx]
        batch = finetune_collate_fn([sample])
        start_t = dataset.clips[idx][2]

        data = batch['data'].to(device=device, dtype=torch.float32)
        input_data = data[:, :t_input]
        target = data[:, 1:t_input + 1]

        output = model(input_data)

        pde_loss = compute_pde_loss_from_output(output, input_data, pde_loss_fn).item()

        rmse_A = torch.sqrt(torch.mean((output[..., CH_A] - target[..., CH_A]) ** 2) + 1e-8).item()
        rmse_B = torch.sqrt(torch.mean((output[..., CH_B] - target[..., CH_B]) ** 2) + 1e-8).item()
        vrmse_A = _vrmse_torch(target[..., CH_A], output[..., CH_A]).item()
        vrmse_B = _vrmse_torch(target[..., CH_B], output[..., CH_B]).item()
        nrmse_A = _nrmse_torch(target[..., CH_A], output[..., CH_A]).item()
        nrmse_B = _nrmse_torch(target[..., CH_B], output[..., CH_B]).item()

        valid_chs = [CH_A, CH_B]
        out_v = output[..., valid_chs]
        tgt_v = target[..., valid_chs]
        mse_all = torch.mean((out_v - tgt_v) ** 2)
        var_all = torch.mean((tgt_v - tgt_v.mean()) ** 2)
        vrmse_all = torch.sqrt(mse_all / (var_all + 1e-8)).item()
        nrmse_all = torch.sqrt(mse_all / (torch.mean(tgt_v ** 2) + 1e-8)).item()

        last = -1
        results.append({
            'gt_A': target[0, last, :, :, CH_A].float().cpu().numpy(),
            'pred_A': output[0, last, :, :, CH_A].float().cpu().numpy(),
            'gt_B': target[0, last, :, :, CH_B].float().cpu().numpy(),
            'pred_B': output[0, last, :, :, CH_B].float().cpu().numpy(),
            'sample_idx': sample_idx, 'start_t': start_t,
        })

        for k, v in [('rmse_A', rmse_A), ('rmse_B', rmse_B),
                      ('vrmse_A', vrmse_A), ('vrmse_B', vrmse_B),
                      ('nrmse_A', nrmse_A), ('nrmse_B', nrmse_B),
                      ('vrmse_all', vrmse_all), ('nrmse_all', nrmse_all),
                      ('pde', pde_loss)]:
            all_metrics[k].append(v)

        print(f"  Sample {sample_idx}: t_start={start_t}, "
              f"nRMSE_A={nrmse_A:.6f}, nRMSE_B={nrmse_B:.6f}, PDE={pde_loss:.6f}")

    # Plot
    n_s = len(results)
    fig, axes = plt.subplots(n_s, 6, figsize=(30, 4 * n_s))
    if n_s == 1:
        axes = axes.reshape(1, -1)

    for row, res in enumerate(results):
        for col_off, (field, cmap) in enumerate([(('gt_A', 'pred_A'), 'viridis'),
                                                    (('gt_B', 'pred_B'), 'magma')]):
            gt = res[field[0]]
            pred = res[field[1]]
            err = pred - gt
            vmin, vmax = min(gt.min(), pred.min()), max(gt.max(), pred.max())
            c = col_off * 3

            im0 = axes[row, c].imshow(gt, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[row, c].set_title(f'GT ({field[0][-1]})', fontsize=11)
            if col_off == 0:
                axes[row, c].set_ylabel(f'Sample {res["sample_idx"]}\nt0={res["start_t"]}', fontsize=10)
            plt.colorbar(im0, ax=axes[row, c], fraction=0.046, pad=0.04)

            im1 = axes[row, c+1].imshow(pred, cmap=cmap, vmin=vmin, vmax=vmax)
            rmse_val = np.sqrt(np.mean(err**2))
            vrmse_val = _vrmse_np(gt, pred)
            axes[row, c+1].set_title(f'Pred {field[0][-1]} (RMSE={rmse_val:.4f})', fontsize=11)
            plt.colorbar(im1, ax=axes[row, c+1], fraction=0.046, pad=0.04)

            err_max = max(np.percentile(np.abs(err), 95), 1e-6)
            im2 = axes[row, c+2].imshow(err, cmap='RdBu_r', vmin=-err_max, vmax=err_max)
            axes[row, c+2].set_title(f'Error {field[0][-1]} (MAE={np.mean(np.abs(err)):.4f})', fontsize=11)
            plt.colorbar(im2, ax=axes[row, c+2], fraction=0.046, pad=0.04)

    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    save_path = str(output_dir / "visualization_gray_scott_unet.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

    print(f"\n{'='*60}")
    print("Visualization Complete (Gray-Scott UNet)")
    print(f"{'='*60}")
    print(f"\nAverage Metrics ({len(results)} samples):")
    for k in ['rmse_A', 'rmse_B', 'vrmse_A', 'vrmse_B',
              'nrmse_A', 'nrmse_B', 'vrmse_all', 'nrmse_all', 'pde']:
        print(f"  - {k:>15s}: {np.mean(all_metrics[k]):.6f}")
    print(f"{'='*60}")


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
        print(f"Gray-Scott UNet Baseline Evaluation")
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
        val_time_interval=config['dataset'].get('val_time_interval', 2),
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
            val_dataset, batch_sampler=val_sampler,
            collate_fn=finetune_collate_fn,
            num_workers=config.get('dataloader', {}).get('num_workers', 4),
            pin_memory=True,
        )

        model, val_loader = accelerator.prepare(model, val_loader)

        if is_main:
            print(f"\nScanning {len(val_dataset)} clips "
                  f"across {accelerator.num_processes} GPUs...")

        results = scan_all_distributed(accelerator, model, val_loader, config, t_input)

        if is_main:
            print(f"\n{'='*60}")
            print(f"Results ({results['num_batches']} batches):")
            print(f"  PDE Loss:      {results['pde']:.6f}")
            print(f"  RMSE (A):      {results['rmse_A']:.6f}")
            print(f"  RMSE (B):      {results['rmse_B']:.6f}")
            print(f"  RMSE:          {results['rmse']:.6f}")
            print(f"  VRMSE (A):     {results['vrmse_A']:.6f}")
            print(f"  VRMSE (B):     {results['vrmse_B']:.6f}")
            print(f"  nRMSE (A):     {results['nrmse_A']:.6f}")
            print(f"  nRMSE (B):     {results['nrmse_B']:.6f}")
            print(f"  VRMSE (all):   {results['vrmse_all']:.6f}")
            print(f"  nRMSE (all):   {results['nrmse_all']:.6f}")
            print(f"{'='*60}")

    else:
        model = model.to(accelerator.device)
        if is_main:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            run_visualization(model, val_dataset, config, accelerator.device,
                              t_input, args.num_samples, args.seed, output_dir)


if __name__ == "__main__":
    main()
