"""
Diagnose BC width effect on boundary vs internal patches.

Compares bc_width=1 vs bc_width=16 checkpoints:
- Boundary patches: outermost ring of patches (have GT in bc_width=16)
- Internal patches: all non-boundary patches (NO GT even in bc_width=16)

If internal patch metrics improve with bc_width=16, it proves that
boundary token accuracy propagates inward through transformer NA.

Usage:
    python tools/diagnose_bc_width_effect.py \
        --config_bc1 configs/finetune_gray_scott_v3.yaml \
        --ckpt_bc1 checkpoints_gray_scott_lora_v4_bc1/best_lora.pt \
        --config_bc16 configs/finetune_gray_scott_v3_dec_v3.yaml \
        --ckpt_bc16 checkpoints_gray_scott_lora_v4_dec_v3/best_lora.pt
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

from finetune.dataset_finetune import FinetuneDataset, finetune_collate_fn
from finetune.model_lora_v3 import PDELoRAModelV3, load_lora_checkpoint


def compute_region_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
) -> dict:
    """Compute RMSE, VRMSE, nRMSE on masked region.

    Args:
        pred, gt: [B, T, H, W] or [B, T, H, W, C]
        mask: [H, W] bool, True = include
    """
    if pred.dim() == 5:
        # [B, T, H, W, C] → apply mask on H, W
        p = pred[:, :, mask, :]
        g = gt[:, :, mask, :]
    else:
        p = pred[:, :, mask]
        g = gt[:, :, mask]

    mse = torch.mean((p - g) ** 2).item()
    rmse = np.sqrt(mse + 1e-10)

    gt_var = torch.mean((g - g.mean()) ** 2).item()
    vrmse = np.sqrt(mse / (gt_var + 1e-10))

    gt_mse_zero = torch.mean(g ** 2).item()
    nrmse = np.sqrt(mse / (gt_mse_zero + 1e-10))

    return {'rmse': rmse, 'vrmse': vrmse, 'nrmse': nrmse}


def create_patch_masks(H: int, W: int, patch_size: int = 16):
    """Create boolean masks for boundary patches vs internal patches.

    Boundary patches = outermost ring of patches.
    Internal patches = everything else.

    Returns:
        boundary_mask: [H, W] bool
        internal_mask: [H, W] bool
    """
    n_h = H // patch_size
    n_w = W // patch_size

    boundary_mask = np.zeros((H, W), dtype=bool)
    internal_mask = np.zeros((H, W), dtype=bool)

    for ih in range(n_h):
        for iw in range(n_w):
            y0 = ih * patch_size
            y1 = (ih + 1) * patch_size
            x0 = iw * patch_size
            x1 = (iw + 1) * patch_size

            is_boundary = (ih == 0 or ih == n_h - 1 or iw == 0 or iw == n_w - 1)
            if is_boundary:
                boundary_mask[y0:y1, x0:x1] = True
            else:
                internal_mask[y0:y1, x0:x1] = True

    return boundary_mask, internal_mask


def evaluate_model(config_path: str, ckpt_path: str, device: torch.device, max_clips: int = 50):
    """Load model, run inference, return (pred, gt, H, W, channel_info)."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    t_input = config['dataset'].get('t_input', 8)
    patch_size = config['model'].get('patch_size', 16)

    model = PDELoRAModelV3(
        config=config,
        pretrained_path=config['model'].get('pretrained_path'),
        freeze_encoder=False, freeze_decoder=False,
    )
    load_lora_checkpoint(model, ckpt_path)
    model = model.float().to(device).eval()

    dataset = FinetuneDataset(
        data_path=config['dataset']['path'],
        temporal_length=t_input + 1,
        split='val',
        train_ratio=config['dataset'].get('train_ratio', 0.9),
        seed=config['dataset'].get('seed', 42),
        clips_per_sample=None,
        vector_dim=config['dataset'].get('vector_dim', 0),
        val_time_interval=config['dataset'].get('val_time_interval', 2),
    )

    n_clips = min(max_clips, len(dataset))

    all_pred = []
    all_gt = []

    with torch.no_grad():
        for idx in range(n_clips):
            sample = dataset[idx]
            batch = finetune_collate_fn([sample])
            data = batch['data'].to(device=device, dtype=torch.float32)
            channel_mask = batch['channel_mask'].to(device=device)

            input_data = data[:, :t_input]
            target_data = data[:, 1:t_input + 1]

            output_norm, mean, std = model(input_data, return_normalized=True)
            output = output_norm * std + mean

            # Get valid channels
            valid_ch = torch.where(channel_mask[0] > 0)[0]

            all_pred.append(output[..., valid_ch].cpu())
            all_gt.append(target_data[..., valid_ch].cpu())

    pred = torch.cat(all_pred, dim=0)  # [N, T, H, W, C_valid]
    gt = torch.cat(all_gt, dim=0)

    H, W = pred.shape[2], pred.shape[3]

    return pred, gt, H, W, patch_size, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_bc1', type=str, required=True)
    parser.add_argument('--ckpt_bc1', type=str, required=True)
    parser.add_argument('--config_bc16', type=str, required=True)
    parser.add_argument('--ckpt_bc16', type=str, required=True)
    parser.add_argument('--max_clips', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device)

    print("=" * 70)
    print("BC Width Effect: Boundary vs Internal Patch Analysis")
    print("=" * 70)

    # Evaluate bc_width=1
    print(f"\nLoading bc_width=1: {args.ckpt_bc1}")
    pred_bc1, gt_bc1, H, W, P, config = evaluate_model(
        args.config_bc1, args.ckpt_bc1, device, args.max_clips)
    print(f"  Shape: pred={pred_bc1.shape}, H={H}, W={W}")

    # Evaluate bc_width=16
    print(f"\nLoading bc_width=16: {args.ckpt_bc16}")
    pred_bc16, gt_bc16, _, _, _, _ = evaluate_model(
        args.config_bc16, args.ckpt_bc16, device, args.max_clips)
    print(f"  Shape: pred={pred_bc16.shape}")

    # Create masks
    boundary_mask, internal_mask = create_patch_masks(H, W, P)
    boundary_mask_t = torch.from_numpy(boundary_mask)
    internal_mask_t = torch.from_numpy(internal_mask)

    n_h, n_w = H // P, W // P
    n_boundary = int(boundary_mask.sum()) // (P * P)
    n_internal = int(internal_mask.sum()) // (P * P)
    print(f"\n  Grid: {n_h}×{n_w} patches ({n_h * n_w} total)")
    print(f"  Boundary patches: {n_boundary} ({100*n_boundary/(n_h*n_w):.1f}%)")
    print(f"  Internal patches: {n_internal} ({100*n_internal/(n_h*n_w):.1f}%)")

    # Compute metrics
    regions = [
        ("All", torch.ones(H, W, dtype=torch.bool)),
        ("Boundary patches", boundary_mask_t),
        ("Internal patches", internal_mask_t),
    ]

    print(f"\n{'='*70}")
    print(f"{'Region':<22} | {'Metric':<8} | {'bc_width=1':>12} | {'bc_width=16':>12} | {'Improvement':>12}")
    print(f"{'-'*70}")

    for region_name, mask in regions:
        m1 = compute_region_metrics(pred_bc1, gt_bc1, mask)
        m16 = compute_region_metrics(pred_bc16, gt_bc16, mask)

        for metric in ['rmse', 'vrmse', 'nrmse']:
            v1 = m1[metric]
            v16 = m16[metric]
            imp = (v1 - v16) / v1 * 100 if v1 > 0 else 0
            label = region_name if metric == 'rmse' else ''
            print(f"{label:<22} | {metric:<8} | {v1:>12.6f} | {v16:>12.6f} | {imp:>+11.1f}%")
        print(f"{'-'*70}")

    # Per-channel analysis (if multi-channel)
    C = pred_bc1.shape[-1]
    if C > 1:
        print(f"\n{'='*70}")
        print(f"Per-Channel Internal Patch Analysis")
        print(f"{'='*70}")
        print(f"{'Channel':<10} | {'Metric':<8} | {'bc_width=1':>12} | {'bc_width=16':>12} | {'Improvement':>12}")
        print(f"{'-'*70}")

        for ch in range(C):
            p1_ch = pred_bc1[..., ch]
            g1_ch = gt_bc1[..., ch]
            p16_ch = pred_bc16[..., ch]
            g16_ch = gt_bc16[..., ch]

            m1 = compute_region_metrics(p1_ch, g1_ch, internal_mask_t)
            m16 = compute_region_metrics(p16_ch, g16_ch, internal_mask_t)

            for metric in ['vrmse', 'nrmse']:
                v1 = m1[metric]
                v16 = m16[metric]
                imp = (v1 - v16) / v1 * 100 if v1 > 0 else 0
                label = f"Ch {ch}" if metric == 'vrmse' else ''
                print(f"{label:<10} | {metric:<8} | {v1:>12.6f} | {v16:>12.6f} | {imp:>+11.1f}%")
        print(f"{'-'*70}")

    print(f"\n{'='*70}")
    print("Key Question: Did internal patches improve with bc_width=16?")
    m1_int = compute_region_metrics(pred_bc1, gt_bc1, internal_mask_t)
    m16_int = compute_region_metrics(pred_bc16, gt_bc16, internal_mask_t)
    imp_vrmse = (m1_int['vrmse'] - m16_int['vrmse']) / m1_int['vrmse'] * 100
    imp_nrmse = (m1_int['nrmse'] - m16_int['nrmse']) / m1_int['nrmse'] * 100
    print(f"  Internal VRMSE: {m1_int['vrmse']:.6f} → {m16_int['vrmse']:.6f} ({imp_vrmse:+.1f}%)")
    print(f"  Internal nRMSE: {m1_int['nrmse']:.6f} → {m16_int['nrmse']:.6f} ({imp_nrmse:+.1f}%)")
    if imp_vrmse > 5:
        print("  → YES: Boundary token accuracy propagates to internal tokens via transformer NA")
    elif imp_vrmse > 0:
        print("  → MARGINAL: Small improvement, boundary info has limited propagation")
    else:
        print("  → NO: Internal patches did not benefit from boundary accuracy")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
