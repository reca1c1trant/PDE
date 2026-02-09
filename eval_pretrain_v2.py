"""
Simple evaluation script for PDEModelV2 on pretrain validation set.

Uses metrics.py for evaluation (RMSE, nRMSE, etc.)

Usage:
    # Single GPU
    python eval_pretrain_v2.py --config configs/pretrain_v2.yaml --checkpoint checkpoints_v2/best.pt

    # Multi-GPU
    torchrun --nproc_per_node=8 eval_pretrain_v2.py --config configs/pretrain_v2.yaml --checkpoint checkpoints_v2/best.pt
"""

import argparse
import yaml
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Dict

from accelerate import Accelerator
from torch.utils.data import DataLoader

from model_v2 import PDEModelV2
from dataset_pretrain import create_pretrain_dataloaders, DEFAULT_TEMPORAL_LENGTH
import metrics
from metrics import metric_func


def load_config(config_path: str) -> dict:
    """Load config from yaml file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path: str, config: dict, device: torch.device) -> PDEModelV2:
    """Load PDEModelV2 from checkpoint."""
    model = PDEModelV2(config)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Handle DDP wrapped state dict
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model = model.float().to(device)
    model.eval()

    return model


@torch.no_grad()
def evaluate_model(
    model: PDEModelV2,
    val_loader: DataLoader,
    accelerator: Accelerator,
    t_input: int = 8,
) -> Dict:
    """
    Evaluate model using metrics.py's metric_func.

    Args:
        model: PDEModelV2
        val_loader: Validation dataloader
        accelerator: Accelerator
        t_input: Number of input timesteps

    Returns:
        results: dict with metrics
    """
    model.eval()

    all_preds = []
    all_targets = []
    num_batches = 0

    iterator = tqdm(
        val_loader,
        desc="Evaluating",
        disable=not accelerator.is_main_process
    )

    for batch in iterator:
        data = batch['data'].to(device=accelerator.device, dtype=torch.float32)
        channel_mask = batch['channel_mask'].to(device=accelerator.device)

        B, T_total, H, W, C = data.shape

        # Use t_input timesteps for input
        input_data = data[:, :t_input]      # [B, t_input, H, W, C]
        target_data = data[:, 1:t_input+1]  # [B, t_input, H, W, C]

        # Forward pass
        output = model(input_data)  # [B, t_input, H, W, C]

        # Apply channel mask to prediction and target
        mask = channel_mask[:, None, None, None, :].float()  # [B, 1, 1, 1, C]
        output_masked = output * mask
        target_masked = target_data * mask

        # Collect predictions and targets
        # Take last frame: [B, H, W, C]
        pred_last = output_masked[:, -1].float()
        target_last = target_masked[:, -1].float()

        all_preds.append(pred_last.cpu())
        all_targets.append(target_last.cpu())
        num_batches += 1

    # Concatenate all predictions
    all_preds = torch.cat(all_preds, dim=0)      # [N_local, H, W, C]
    all_targets = torch.cat(all_targets, dim=0)  # [N_local, H, W, C]

    # Gather across all processes
    accelerator.wait_for_everyone()
    all_preds_gathered = accelerator.gather(all_preds.to(accelerator.device))
    all_targets_gathered = accelerator.gather(all_targets.to(accelerator.device))

    # Move to CPU immediately to avoid GPU OOM
    all_preds_gathered = all_preds_gathered.cpu()
    all_targets_gathered = all_targets_gathered.cpu()

    results = {}

    if accelerator.is_main_process:
        # Add time dimension for metric_func: [N, H, W, T=1, C]
        all_preds_gathered = all_preds_gathered.unsqueeze(-2)
        all_targets_gathered = all_targets_gathered.unsqueeze(-2)

        # Compute metrics using metric_func (on CPU to avoid OOM)
        # Temporarily override metrics.device to CPU
        original_device = metrics.device
        metrics.device = torch.device('cpu')

        (
            err_RMSE,
            err_nRMSE,
            err_CSV,
            err_Max,
            err_BD,
            err_F,
            err_MPPnRMSE,
        ) = metric_func(
            all_preds_gathered.float(),
            all_targets_gathered.float(),
            if_mean=True,
            Lx=1.0, Ly=1.0, Lz=1.0,
            iLow=4, iHigh=12,
            initial_step=0
        )

        # Restore original device
        metrics.device = original_device

        results = {
            'rmse': err_RMSE.item() if hasattr(err_RMSE, 'item') else float(err_RMSE),
            'nrmse': err_nRMSE.item() if hasattr(err_nRMSE, 'item') else float(err_nRMSE),
            'mpp_nrmse': err_MPPnRMSE.item() if hasattr(err_MPPnRMSE, 'item') else float(err_MPPnRMSE),
            'max_error': err_Max.item() if hasattr(err_Max, 'item') else float(err_Max),
            'boundary_rmse': err_BD.item() if hasattr(err_BD, 'item') else float(err_BD),
            'csv_rmse': err_CSV.item() if hasattr(err_CSV, 'item') else float(err_CSV),
            'fourier_low': err_F[0].item() if hasattr(err_F[0], 'item') else float(err_F[0]),
            'fourier_mid': err_F[1].item() if hasattr(err_F[1], 'item') else float(err_F[1]),
            'fourier_high': err_F[2].item() if hasattr(err_F[2], 'item') else float(err_F[2]),
            'num_samples': all_preds_gathered.shape[0],
        }

    return results


def print_results(results: Dict):
    """Print evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\n{'Metric':<20} {'Value':>15}")
    print("-" * 40)

    metrics_order = [
        ('rmse', 'RMSE'),
        ('nrmse', 'nRMSE'),
        ('mpp_nrmse', 'MPP-nRMSE'),
        ('max_error', 'Max Error'),
        ('boundary_rmse', 'Boundary RMSE'),
        ('csv_rmse', 'CSV RMSE'),
    ]

    for key, name in metrics_order:
        if key in results:
            print(f"{name:<20} {results[key]:>15.6f}")

    print("-" * 40)
    print("\nFourier Space RMSE:")
    print(f"  Low freq:  {results.get('fourier_low', 0):.6f}")
    print(f"  Mid freq:  {results.get('fourier_mid', 0):.6f}")
    print(f"  High freq: {results.get('fourier_high', 0):.6f}")

    print(f"\nTotal samples: {results.get('num_samples', 0)}")
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PDEModelV2")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Accelerator
    accelerator = Accelerator()

    if accelerator.is_main_process:
        print(f"\n{'=' * 60}")
        print("PDEModelV2 Evaluation (metrics.py)")
        print(f"{'=' * 60}")
        print(f"Config: {args.config}")
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Device: {accelerator.device}")
        print(f"Num processes: {accelerator.num_processes}")

    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        if accelerator.is_main_process:
            print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        return

    # Load model
    if accelerator.is_main_process:
        print("\nLoading model...")

    model = load_model(args.checkpoint, config, accelerator.device)

    if accelerator.is_main_process:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,}")

    # Create validation dataloader
    t_input = config['dataset'].get('t_input', 8)
    multi_step_n = config['training'].get('multi_step_loss', {}).get('num_steps', 3)
    temporal_length = t_input + multi_step_n
    clips_ratio = config['dataset'].get('clips_ratio', 0.25)

    data_dir = config['dataset']['path']
    seed = config['dataset']['seed']
    dataset_overrides = config.get('dataset', {}).get('overrides', {})

    _, val_loader, _, val_sampler = create_pretrain_dataloaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        seed=seed,
        dataset_overrides=dataset_overrides,
        temporal_length=temporal_length,
        clips_ratio=clips_ratio,
    )

    if accelerator.is_main_process:
        print(f"Validation batches: {len(val_sampler)}")
        print(f"t_input: {t_input}")
        print(f"temporal_length: {temporal_length}")

    # Evaluate
    if accelerator.is_main_process:
        print("\nRunning evaluation...")

    results = evaluate_model(model, val_loader, accelerator, t_input=t_input)

    # Print results
    if accelerator.is_main_process:
        print_results(results)


if __name__ == "__main__":
    main()
