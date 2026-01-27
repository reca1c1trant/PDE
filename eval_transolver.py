"""
Evaluate models on 2D Diffusion Reaction dataset.

Flexible evaluation - only evaluates models you specify.

Usage:
    # Evaluate single model
    python eval_transolver.py --model Base:base:./checkpoints_e2e_medium/best.pt

    # Evaluate multiple models
    python eval_transolver.py \
        --model Base:base:./checkpoints_e2e_medium/best.pt \
        --model Transolver:transolver:./checkpoints_transolver/best.pt

    # Model types: base, base_v2, transolver

    # Custom data path
    python eval_transolver.py \
        --model MyModel:base:./my_checkpoint.pt \
        --data_path ./data/2D_diff-react_NA_NA.hdf5
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import PDEDataset, DimensionGroupedSampler, collate_fn
from pipeline import PDECausalModel
from pipeline_v2 import PDECausalModelV2
from model_transolver import PDETransolver
from metrics import metric_func


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate models on Diffusion Reaction")
    parser.add_argument('--model', action='append', required=True,
                        help='Model specification: NAME:TYPE:PATH (e.g., Base:base:./ckpt.pt). '
                             'TYPE can be: base, base_v2, transolver')
    parser.add_argument('--data_path', type=str,
                        default='/home/msai/song0304/code/PDE/data/2D_diff-react_NA_NA.hdf5',
                        help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


def parse_model_spec(spec: str):
    """Parse model specification string: NAME:TYPE:PATH"""
    parts = spec.split(':')
    if len(parts) != 3:
        raise ValueError(f"Invalid model spec: {spec}. Expected NAME:TYPE:PATH")
    name, model_type, path = parts
    if model_type not in ['base', 'base_v2', 'transolver']:
        raise ValueError(f"Unknown model type: {model_type}. Expected: base, base_v2, transolver")
    return name, model_type, path


# ============ Model Configs ============

BASE_CONFIG = {
    'model': {
        'in_channels': 6,
        'noise_level': 0.0,
        'use_flash_attention': False,
        'gradient_checkpointing': False,
        'encoder': {
            'version': 'v2',
            'channels': [64, 128, 256],
            'use_resblock': True,
        },
        'transformer': {
            'hidden_size': 768,
            'num_hidden_layers': 10,
            'num_attention_heads': 16,
            'num_key_value_heads': 4,
            'intermediate_size': 3072,
            'hidden_act': 'silu',
            'max_position_embeddings': 4096,
            'rms_norm_eps': 1e-5,
            'rope_theta': 500000.0,
            'attention_dropout': 0.0,
        },
    },
}

TRANSOLVER_CONFIG = {
    'model': {
        'in_channels': 6,
        'transolver': {
            'hidden_dim': 256,
            'num_layers': 8,
            'num_heads': 8,
            'num_slices': 64,
            'patch_size': 4,
            'mlp_ratio': 4.0,
            'dropout': 0.0,
            'use_temporal_residual': True,
        }
    }
}


# ============ Model Loading ============

def load_model(name: str, model_type: str, checkpoint_path: str, device: str):
    """Load model based on type."""
    print(f"\nLoading {name} ({model_type}) from: {checkpoint_path}")

    if not Path(checkpoint_path).exists():
        print(f"  ERROR: Checkpoint not found!")
        return None

    # Create model
    if model_type == 'base':
        model = PDECausalModel(BASE_CONFIG)
    elif model_type == 'base_v2':
        model = PDECausalModelV2(BASE_CONFIG)
    elif model_type == 'transolver':
        model = PDETransolver(TRANSOLVER_CONFIG)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    print(f"  Loaded checkpoint (step={checkpoint.get('global_step', 'N/A')})")

    # Use fp32 for precision
    model = model.float().to(device=device)
    model.eval()
    return model


# ============ Evaluation ============

def compute_nrmse(pred: torch.Tensor, target: torch.Tensor, channel_mask: torch.Tensor):
    """Compute nRMSE."""
    eps = 1e-8

    if channel_mask.dim() == 2:
        valid_mask = channel_mask[0].bool()
    else:
        valid_mask = channel_mask.bool()

    pred_valid = pred[..., valid_mask]
    target_valid = target[..., valid_mask]

    B, T, H, W, C = pred_valid.shape

    pred_flat = pred_valid.permute(0, 4, 1, 2, 3).reshape(B, C, -1)
    target_flat = target_valid.permute(0, 4, 1, 2, 3).reshape(B, C, -1)

    mse_per_bc = ((pred_flat - target_flat) ** 2).mean(dim=2)
    rmse_per_bc = torch.sqrt(mse_per_bc + eps)

    rms_per_bc = torch.sqrt((target_flat ** 2).mean(dim=2) + eps)

    nrmse_per_bc = rmse_per_bc / rms_per_bc

    nrmse = nrmse_per_bc.mean()
    rmse = rmse_per_bc.mean()

    return nrmse.item(), rmse.item()


@torch.no_grad()
def evaluate_model(model, val_loader, device, model_name="Model"):
    """Evaluate a model on validation set."""
    print(f"  Evaluating...")

    all_preds = []
    all_targets = []
    total_nrmse = 0.0
    total_rmse = 0.0
    num_batches = 0

    for batch in tqdm(val_loader, desc=f"  {model_name}", leave=False):
        data = batch['data'].to(device=device, dtype=torch.float32)
        channel_mask = batch['channel_mask'].to(device=device)

        input_data = data[:, :-1]
        target_data = data[:, 1:]

        output = model(input_data)

        nrmse, rmse = compute_nrmse(output.float(), target_data.float(), channel_mask)
        total_nrmse += nrmse
        total_rmse += rmse
        num_batches += 1

        if channel_mask.dim() == 2:
            valid_mask = channel_mask[0].bool()
        else:
            valid_mask = channel_mask.bool()

        pred_last = output[:, -1, :, :, :][:, :, :, valid_mask].float()
        target_last = target_data[:, -1, :, :, :][:, :, :, valid_mask].float()

        all_preds.append(pred_last.cpu())
        all_targets.append(target_last.cpu())

    avg_nrmse = total_nrmse / num_batches
    avg_rmse = total_rmse / num_batches

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute additional metrics
    all_preds_metric = all_preds.unsqueeze(-2)
    all_targets_metric = all_targets.unsqueeze(-2)

    err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F = metric_func(
        all_preds_metric.float(),
        all_targets_metric.float(),
        if_mean=True,
        Lx=1.0, Ly=1.0, Lz=1.0,
        iLow=4, iHigh=12,
        initial_step=0
    )

    results = {
        'nrmse': avg_nrmse,
        'rmse': avg_rmse,
        'max_error': err_Max.item(),
        'boundary_rmse': err_BD.item(),
        'fourier_low': err_F[0].item(),
        'fourier_mid': err_F[1].item(),
        'fourier_high': err_F[2].item(),
    }

    return results


def count_parameters(model):
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    return total


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Parse model specifications
    model_specs = []
    for spec in args.model:
        try:
            name, model_type, path = parse_model_spec(spec)
            model_specs.append((name, model_type, path))
        except ValueError as e:
            print(f"ERROR: {e}")
            return

    print(f"\nModels to evaluate: {len(model_specs)}")
    for name, model_type, path in model_specs:
        print(f"  - {name} ({model_type}): {path}")

    # Check data path
    if not Path(args.data_path).exists():
        print(f"\nERROR: Data not found: {args.data_path}")
        return

    # Create validation dataloader
    print(f"\nLoading dataset: {args.data_path}")
    val_dataset = PDEDataset(
        data_dir=args.data_path,
        temporal_length=16,
        split='val',
        train_ratio=0.9,
        seed=42,
        clips_per_sample=None,
    )

    val_sampler = DimensionGroupedSampler(val_dataset, args.batch_size, shuffle=False, seed=42)
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    print(f"Validation set: {len(val_dataset)} clips, {len(val_sampler)} batches")

    # Evaluate models
    all_results = {}
    param_counts = {}

    for name, model_type, path in model_specs:
        print("\n" + "="*60)
        model = load_model(name, model_type, path, device)
        if model is None:
            continue

        param_counts[name] = count_parameters(model)
        all_results[name] = evaluate_model(model, val_loader, device, name)

        del model
        torch.cuda.empty_cache()

    if len(all_results) == 0:
        print("\nNo models evaluated!")
        return

    # Print results
    print("\n" + "="*90)
    print("RESULTS")
    print("="*90)

    models = list(all_results.keys())

    # Parameter counts
    print("\n[Parameters]")
    for name in models:
        print(f"  {name}: {param_counts[name]:,}")

    # Main metrics
    print("\n[Metrics]")
    print("-"*90)

    # Header
    header = f"{'Metric':<20}" + "".join([f"{m:>18}" for m in models])
    print(header)
    print("-"*90)

    metrics_to_show = ['nrmse', 'rmse', 'max_error', 'boundary_rmse']
    for metric in metrics_to_show:
        row = f"{metric.upper():<20}"
        for m in models:
            val = all_results[m].get(metric, 0)
            row += f"{val:>18.6f}"
        print(row)

    print("-"*90)

    # Fourier
    print("\n[Fourier RMSE]")
    for freq, key in [('Low', 'fourier_low'), ('Mid', 'fourier_mid'), ('High', 'fourier_high')]:
        row = f"  {freq:<8}"
        for m in models:
            val = all_results[m].get(key, 0)
            row += f"  {m}={val:.6f}"
        print(row)

    # Comparison if multiple models
    if len(models) >= 2:
        print("\n[Comparison]")
        baseline = models[0]
        baseline_nrmse = all_results[baseline]['nrmse']
        for m in models[1:]:
            m_nrmse = all_results[m]['nrmse']
            diff = (m_nrmse - baseline_nrmse) / baseline_nrmse * 100
            better_worse = "worse" if diff > 0 else "better"
            print(f"  {m} vs {baseline}: {diff:+.2f}% nRMSE ({better_worse})")

    print("="*90)


if __name__ == "__main__":
    main()
