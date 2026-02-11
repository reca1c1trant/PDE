"""
Evaluate PDEModelV2 on pretrain validation set.

Metrics:
- RMSE (per-channel and overall)
- nRMSE (normalized RMSE)
- Per-dataset breakdown

Usage:
    # Single GPU
    python eval_model_v2.py --config configs/pretrain_v2.yaml --checkpoint checkpoints_v2/best.pt

    # Multi-GPU
    torchrun --nproc_per_node=8 eval_model_v2.py --config configs/pretrain_v2.yaml --checkpoint checkpoints_v2/best.pt
"""

import os
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Tuple

from accelerate import Accelerator
from torch.utils.data import DataLoader
from rich.console import Console
from rich.table import Table

from pretrain.model_v2 import PDEModelV2
from pretrain.dataset_pretrain import (
    PretrainDataset, PretrainSampler, pretrain_collate_fn,
    create_pretrain_dataloaders, DEFAULT_TEMPORAL_LENGTH,
    NUM_VECTOR_CHANNELS, NUM_SCALAR_CHANNELS, TOTAL_CHANNELS,
)


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


def compute_rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    channel_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute RMSE and nRMSE with channel masking.

    Args:
        pred: [B, T, H, W, C]
        target: [B, T, H, W, C]
        channel_mask: [B, C]

    Returns:
        rmse: scalar
        nrmse: scalar
    """
    # Expand mask to match data shape
    mask = channel_mask[:, None, None, None, :].float()  # [B, 1, 1, 1, C]

    # Squared error
    squared_error = (pred - target) ** 2
    masked_error = squared_error * mask

    # RMSE
    total_error = masked_error.sum()
    num_valid = mask.sum() * pred.shape[1] * pred.shape[2] * pred.shape[3]
    mse = total_error / (num_valid + 1e-8)
    rmse = torch.sqrt(mse + 1e-8)

    # nRMSE (normalized by target std)
    target_masked = target * mask
    target_std = target_masked.std() + 1e-8
    nrmse = rmse / target_std

    return rmse, nrmse


def compute_per_channel_rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    channel_mask: torch.Tensor,
) -> Dict[str, float]:
    """Compute per-channel RMSE."""
    B, T, H, W, C = pred.shape
    results = {}

    # Vector channels (0, 1, 2)
    vector_names = ['Vx', 'Vy', 'Vz']
    for i in range(NUM_VECTOR_CHANNELS):
        if channel_mask[:, i].sum() > 0:
            err = (pred[..., i] - target[..., i]) ** 2
            valid_mask = channel_mask[:, i:i+1, None, None].float()
            rmse = torch.sqrt(err.mean() + 1e-8)
            results[vector_names[i]] = rmse.item()

    # Scalar channels (3-17)
    scalar_names = [
        'buoyancy', 'concentration_rho', 'concentration_u', 'concentration_v',
        'density', 'electron_fraction', 'energy', 'entropy', 'geometry',
        'gravitational_potential', 'height', 'passive_tracer', 'pressure',
        'speed_of_sound', 'temperature'
    ]
    for i, name in enumerate(scalar_names):
        idx = NUM_VECTOR_CHANNELS + i
        if channel_mask[:, idx].sum() > 0:
            err = (pred[..., idx] - target[..., idx]) ** 2
            rmse = torch.sqrt(err.mean() + 1e-8)
            results[name] = rmse.item()

    return results


@torch.no_grad()
def evaluate_model(
    model: PDEModelV2,
    val_loader: DataLoader,
    accelerator: Accelerator,
    t_input: int = 8,
) -> Dict:
    """
    Evaluate model on validation set.

    Returns:
        results: dict with overall and per-dataset metrics
    """
    model.eval()

    # Accumulators
    total_rmse = torch.zeros(1, device=accelerator.device)
    total_nrmse = torch.zeros(1, device=accelerator.device)
    num_batches = torch.zeros(1, device=accelerator.device)

    # Per-dataset accumulators
    all_dataset_names = ['diffusion_reaction', '2d_cfd', 'swe']
    dataset_rmse = {name: torch.zeros(1, device=accelerator.device) for name in all_dataset_names}
    dataset_nrmse = {name: torch.zeros(1, device=accelerator.device) for name in all_dataset_names}
    dataset_count = {name: torch.zeros(1, device=accelerator.device) for name in all_dataset_names}

    # Progress bar
    iterator = tqdm(
        val_loader,
        desc="Evaluating",
        disable=not accelerator.is_main_process
    )

    for batch in iterator:
        data = batch['data'].to(device=accelerator.device, dtype=torch.float32)
        channel_mask = batch['channel_mask'].to(device=accelerator.device)
        dataset_names = batch['dataset_names']

        B, T_total, H, W, C = data.shape

        # Use t_input timesteps for input
        input_data = data[:, :t_input]      # [B, t_input, H, W, C]
        target_data = data[:, 1:t_input+1]  # [B, t_input, H, W, C]

        # Forward pass
        output = model(input_data)  # [B, t_input, H, W, C]

        # Compute metrics
        rmse, nrmse = compute_rmse(output, target_data, channel_mask)

        total_rmse += rmse.detach()
        total_nrmse += nrmse.detach()
        num_batches += 1

        # Per-dataset metrics
        ds_name = dataset_names[0]  # Assume batch is same dataset
        if ds_name in dataset_rmse:
            dataset_rmse[ds_name] += rmse.detach()
            dataset_nrmse[ds_name] += nrmse.detach()
            dataset_count[ds_name] += 1

    # Reduce across GPUs
    accelerator.wait_for_everyone()

    total_rmse = accelerator.reduce(total_rmse, reduction='sum')
    total_nrmse = accelerator.reduce(total_nrmse, reduction='sum')
    num_batches = accelerator.reduce(num_batches, reduction='sum')

    for ds_name in all_dataset_names:
        dataset_rmse[ds_name] = accelerator.reduce(dataset_rmse[ds_name], reduction='sum')
        dataset_nrmse[ds_name] = accelerator.reduce(dataset_nrmse[ds_name], reduction='sum')
        dataset_count[ds_name] = accelerator.reduce(dataset_count[ds_name], reduction='sum')

    # Compute averages
    results = {
        'overall': {
            'rmse': (total_rmse / num_batches).item() if num_batches.item() > 0 else 0,
            'nrmse': (total_nrmse / num_batches).item() if num_batches.item() > 0 else 0,
            'num_batches': int(num_batches.item()),
        },
        'per_dataset': {},
    }

    for ds_name in all_dataset_names:
        count = dataset_count[ds_name].item()
        if count > 0:
            results['per_dataset'][ds_name] = {
                'rmse': dataset_rmse[ds_name].item() / count,
                'nrmse': dataset_nrmse[ds_name].item() / count,
                'num_batches': int(count),
            }

    return results


@torch.no_grad()
def evaluate_autoregressive(
    model: PDEModelV2,
    val_loader: DataLoader,
    accelerator: Accelerator,
    t_input: int = 8,
    rollout_steps: int = 3,
) -> Dict:
    """
    Evaluate model with autoregressive rollout.

    Args:
        model: The model
        val_loader: Validation dataloader
        accelerator: Accelerator
        t_input: Number of input timesteps
        rollout_steps: Number of AR rollout steps

    Returns:
        results: dict with per-step metrics
    """
    model.eval()

    step_rmse = {i: torch.zeros(1, device=accelerator.device) for i in range(rollout_steps)}
    step_count = torch.zeros(1, device=accelerator.device)

    iterator = tqdm(
        val_loader,
        desc="AR Evaluation",
        disable=not accelerator.is_main_process
    )

    for batch in iterator:
        data = batch['data'].to(device=accelerator.device, dtype=torch.float32)
        channel_mask = batch['channel_mask'].to(device=accelerator.device)

        B, T_total, H, W, C = data.shape
        required_T = t_input + rollout_steps

        if T_total < required_T:
            continue

        # Initial input
        current_input = data[:, :t_input]  # [B, t_input, H, W, C]

        for step in range(rollout_steps):
            # Target for this step
            target_idx = step + 1
            target = data[:, target_idx:target_idx + t_input]  # [B, t_input, H, W, C]

            # Forward pass
            output = model(current_input)  # [B, t_input, H, W, C]

            # Compute RMSE for last frame prediction
            pred_last = output[:, -1]   # [B, H, W, C]
            target_last = target[:, -1]  # [B, H, W, C]

            mask = channel_mask[:, None, None, :].float()
            err = ((pred_last - target_last) ** 2 * mask).sum() / (mask.sum() * H * W + 1e-8)
            rmse = torch.sqrt(err + 1e-8)

            step_rmse[step] += rmse.detach()

            # Build next input: GT[step+1:step+t_input] + pred[-1]
            if step < rollout_steps - 1:
                gt_part = data[:, step + 1:step + t_input]  # [B, t_input-1, H, W, C]
                pred_part = output[:, -1:]  # [B, 1, H, W, C]
                current_input = torch.cat([gt_part, pred_part], dim=1)

        step_count += 1

    # Reduce
    accelerator.wait_for_everyone()
    step_count = accelerator.reduce(step_count, reduction='sum')

    results = {'ar_steps': {}}
    for step in range(rollout_steps):
        step_rmse[step] = accelerator.reduce(step_rmse[step], reduction='sum')
        if step_count.item() > 0:
            results['ar_steps'][f'step_{step+1}'] = step_rmse[step].item() / step_count.item()

    results['num_samples'] = int(step_count.item())

    return results


def print_results(results: Dict, ar_results: Dict, console: Console):
    """Print evaluation results in a nice table."""
    # Overall metrics
    table = Table(title="Overall Metrics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("RMSE", f"{results['overall']['rmse']:.6f}")
    table.add_row("nRMSE", f"{results['overall']['nrmse']:.6f}")
    table.add_row("Num Batches", str(results['overall']['num_batches']))

    console.print(table)

    # Per-dataset metrics
    if results['per_dataset']:
        table = Table(title="Per-Dataset Metrics", show_header=True)
        table.add_column("Dataset", style="cyan")
        table.add_column("RMSE", style="green")
        table.add_column("nRMSE", style="green")
        table.add_column("Batches", style="dim")

        for ds_name, metrics in results['per_dataset'].items():
            table.add_row(
                ds_name,
                f"{metrics['rmse']:.6f}",
                f"{metrics['nrmse']:.6f}",
                str(metrics['num_batches'])
            )

        console.print(table)

    # AR metrics
    if ar_results and ar_results.get('ar_steps'):
        table = Table(title="Autoregressive Rollout RMSE", show_header=True)
        table.add_column("Step", style="cyan")
        table.add_column("RMSE", style="green")

        for step_name, rmse in ar_results['ar_steps'].items():
            table.add_row(step_name, f"{rmse:.6f}")

        console.print(table)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PDEModelV2")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--ar_steps', type=int, default=3, help='AR rollout steps')
    parser.add_argument('--skip_ar', action='store_true', help='Skip AR evaluation')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Accelerator
    accelerator = Accelerator()
    console = Console()

    if accelerator.is_main_process:
        console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        console.print("[bold cyan]PDEModelV2 Evaluation[/bold cyan]")
        console.print(f"[bold cyan]{'='*60}[/bold cyan]")
        console.print(f"Config: {args.config}")
        console.print(f"Checkpoint: {args.checkpoint}")
        console.print(f"Device: {accelerator.device}")
        console.print(f"Num processes: {accelerator.num_processes}")

    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        if accelerator.is_main_process:
            console.print(f"[red]ERROR: Checkpoint not found: {args.checkpoint}[/red]")
        return

    # Load model
    if accelerator.is_main_process:
        console.print("\n[yellow]Loading model...[/yellow]")

    model = load_model(args.checkpoint, config, accelerator.device)

    if accelerator.is_main_process:
        num_params = sum(p.numel() for p in model.parameters())
        console.print(f"Model parameters: {num_params:,}")

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
        console.print(f"Validation batches: {len(val_sampler)}")

    # Evaluate
    if accelerator.is_main_process:
        console.print("\n[yellow]Running evaluation...[/yellow]")

    results = evaluate_model(model, val_loader, accelerator, t_input=t_input)

    # AR evaluation
    ar_results = {}
    if not args.skip_ar:
        if accelerator.is_main_process:
            console.print("\n[yellow]Running AR evaluation...[/yellow]")
        ar_results = evaluate_autoregressive(
            model, val_loader, accelerator,
            t_input=t_input, rollout_steps=args.ar_steps
        )

    # Print results
    if accelerator.is_main_process:
        console.print()
        print_results(results, ar_results, console)
        console.print(f"\n[bold green]{'='*60}[/bold green]")
        console.print("[bold green]Evaluation Complete[/bold green]")
        console.print(f"[bold green]{'='*60}[/bold green]")


if __name__ == "__main__":
    main()
