"""
Baseline Training Script - Flow Mixing Equation.

Train UNet or MLP model on Flow Mixing equation.
Supports both central difference (v2) and 2nd order upwind (v1) PDE loss.

Usage:
    # Single GPU
    python train_fno_baseline_flow.py --config configs/unet_baseline_flow_v2.yaml

    # Multi-GPU
    torchrun --nproc_per_node=8 train_fno_baseline_flow.py --config configs/unet_baseline_flow_v2.yaml

    torchrun --nproc_per_node=8 train_fno_baseline_flow.py --config configs/mlp_baseline_flow.yaml

    python train_fno_baseline_flow.py --config configs/mlp_baseline_flow.yaml
    """

import os
import sys
import warnings

def _is_main_process():
    return os.environ.get('LOCAL_RANK', '0') == '0'

IS_MAIN_PROCESS = _is_main_process()

if not IS_MAIN_PROCESS:
    warnings.filterwarnings('ignore')

triton_cache = '/tmp/triton_cache'
os.makedirs(triton_cache, exist_ok=True)
os.environ.setdefault('TRITON_CACHE_DIR', triton_cache)

import argparse
import yaml
import logging
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    TaskProgressColumn, MofNCompleteColumn, TimeRemainingColumn
)
from rich.table import Table

torch.set_float32_matmul_precision('high')

if IS_MAIN_PROCESS:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
else:
    logging.disable(logging.CRITICAL)

from model_unet_baseline import create_unet_baseline
from model_mlp_baseline import create_mlp_baseline
from dataset_flow import FlowMixingDataset, FlowMixingSampler, flow_mixing_collate_fn
from pde_loss_flow import flow_mixing_pde_loss
from pde_loss_flow_v2 import flow_mixing_pde_loss_v2

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_dataloaders(config: dict):
    """Create train and validation dataloaders."""
    clips_per_sample = config['dataset'].get('clips_per_sample', None)

    train_dataset = FlowMixingDataset(
        data_path=config['dataset']['path'],
        temporal_length=config['dataset']['temporal_length'],
        split='train',
        train_ratio=config['dataset']['train_ratio'],
        seed=config['dataset']['seed'],
        clips_per_sample=clips_per_sample,
    )

    val_dataset = FlowMixingDataset(
        data_path=config['dataset']['path'],
        temporal_length=config['dataset']['temporal_length'],
        split='val',
        train_ratio=config['dataset']['train_ratio'],
        seed=config['dataset']['seed'],
        clips_per_sample=None,
    )

    batch_size = config['dataloader']['batch_size']
    seed = config['dataset']['seed']

    train_sampler = FlowMixingSampler(train_dataset, batch_size, shuffle=True, seed=seed)
    val_sampler = FlowMixingSampler(val_dataset, batch_size, shuffle=False, seed=seed)

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=flow_mixing_collate_fn,
        num_workers=config['dataloader']['num_workers'],
        pin_memory=config['dataloader']['pin_memory']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=flow_mixing_collate_fn,
        num_workers=config['dataloader']['num_workers'],
        pin_memory=config['dataloader']['pin_memory']
    )

    return train_loader, val_loader, train_sampler, val_sampler


def get_lr_scheduler(optimizer, config: dict, total_steps: int):
    """Create learning rate scheduler with warmup and cosine decay."""
    from torch.optim.lr_scheduler import LambdaLR
    import math

    warmup_steps = config['training'].get('warmup_steps', 500)
    min_lr = config['training'].get('min_lr', 1e-6)
    base_lr = config['training']['learning_rate']
    min_lr_ratio = min_lr / base_lr

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def compute_pde_loss(output, input_data, batch, config, accelerator, pde_version="v1"):
    """Compute PDE residual loss for Flow Mixing equation.

    Flow Mixing: ∂u/∂t + a·∂u/∂x + b·∂u/∂y = 0
    Only uses first channel (u).

    IMPORTANT: PDE loss must be computed in float32 for numerical stability!
    The time derivative du/dt = (u[t] - u[t-1]) / dt is very sensitive to precision
    because u[t] ≈ u[t-1] when the model is well-trained, causing catastrophic
    cancellation in lower precision formats like bf16.

    Args:
        pde_version: "v1" for 2nd order upwind, "v2" for central difference
    """
    # CRITICAL: Disable autocast to ensure all PDE computations are in float32
    # accelerator.accumulate() enables autocast which can affect intermediate computations
    with torch.autocast(device_type='cuda', enabled=False):
        # Use only first channel for PDE loss
        # Convert to float32 BEFORE any computation to avoid precision loss
        t0_frame = input_data[:, 0:1, ..., :1].float()  # [B, 1, H, W, 1]
        output_u = output[..., :1].float()  # [B, T-1, H, W, 1]
        pred_with_t0 = torch.cat([t0_frame, output_u], dim=1)
        pred_u = pred_with_t0  # Already float32

        # Boundaries only have 1 channel
        boundary_left = batch['boundary_left'].to(accelerator.device).float()
        boundary_right = batch['boundary_right'].to(accelerator.device).float()
        boundary_bottom = batch['boundary_bottom'].to(accelerator.device).float()
        boundary_top = batch['boundary_top'].to(accelerator.device).float()
        vtmax = batch['vtmax'].to(accelerator.device).float()

        dt = config.get('physics', {}).get('dt', 1/999)
        vtmax_mean = vtmax.mean().item()

        # Select PDE loss version
        if pde_version == "v2":
            # V2: Central difference (train_PINN_transient style)
            pde_loss, loss_time, loss_advection, _ = flow_mixing_pde_loss_v2(
                pred=pred_u,
                boundary_left=boundary_left,
                boundary_right=boundary_right,
                boundary_bottom=boundary_bottom,
                boundary_top=boundary_top,
                vtmax=vtmax_mean,
                dt=dt
            )
        else:
            # V1: 2nd order upwind (n-PINN style)
            pde_loss, loss_time, loss_advection, _ = flow_mixing_pde_loss(
                pred=pred_u,
                boundary_left=boundary_left,
                boundary_right=boundary_right,
                boundary_bottom=boundary_bottom,
                boundary_top=boundary_top,
                vtmax=vtmax_mean,
                dt=dt
            )

    return pde_loss, loss_time, loss_advection


def compute_rmse_loss(output, target):
    """
    Compute RMSE loss between output and ground truth.

    Args:
        output: [B, T, H, W, C] (C=1 for Flow Mixing)
        target: [B, T, H, W, C]
    """
    # Disable autocast for numerical stability
    with torch.autocast(device_type='cuda', enabled=False):
        output_f32 = output.float()
        target_f32 = target.float()
        mse = torch.mean((output_f32 - target_f32) ** 2)
        rmse = torch.sqrt(mse + 1e-8)
    return rmse


def compute_boundary_loss(output, target):
    """
    Compute boundary RMSE loss on 4 edges (excluding corners).

    Args:
        output: [B, T, H, W, C] (C=1 for Flow Mixing)
        target: [B, T, H, W, C]
    """
    # Disable autocast for numerical stability
    with torch.autocast(device_type='cuda', enabled=False):
        output = output.float()
        target = target.float()

        # Left edge (excluding corners)
        left_pred = output[:, :, 1:-1, 0, :]
        left_target = target[:, :, 1:-1, 0, :]

        # Right edge (excluding corners)
        right_pred = output[:, :, 1:-1, -1, :]
        right_target = target[:, :, 1:-1, -1, :]

        # Bottom edge (excluding corners)
        bottom_pred = output[:, :, 0, 1:-1, :]
        bottom_target = target[:, :, 0, 1:-1, :]

        # Top edge (excluding corners)
        top_pred = output[:, :, -1, 1:-1, :]
        top_target = target[:, :, -1, 1:-1, :]

        # Concatenate all boundary points
        bc_pred = torch.cat([
            left_pred.reshape(-1),
            right_pred.reshape(-1),
            bottom_pred.reshape(-1),
            top_pred.reshape(-1),
        ])
        bc_target = torch.cat([
            left_target.reshape(-1),
            right_target.reshape(-1),
            bottom_target.reshape(-1),
            top_target.reshape(-1),
        ])

        # RMSE
        mse = torch.mean((bc_pred - bc_target) ** 2)
        rmse = torch.sqrt(mse + 1e-8)
    return rmse


@torch.no_grad()
def validate(model, val_loader, config, accelerator, pde_version="v1"):
    """Run validation and compute metrics."""
    model.eval()

    total_pde_loss = torch.zeros(1, device=accelerator.device)
    total_rmse_loss = torch.zeros(1, device=accelerator.device)
    total_bc_loss = torch.zeros(1, device=accelerator.device)
    num_batches = torch.zeros(1, device=accelerator.device)

    for batch in val_loader:
        # Only use first channel (Flow Mixing has 1 real channel)
        data = batch['data'][..., :1].to(device=accelerator.device, dtype=torch.float32)
        input_data = data[:, :-1]
        target = data[:, 1:]

        output = model(data)

        pde_loss, _, _ = compute_pde_loss(output, input_data, batch, config, accelerator, pde_version)
        rmse_loss = compute_rmse_loss(output, target)
        bc_loss = compute_boundary_loss(output, target)

        total_pde_loss += pde_loss.detach()
        total_rmse_loss += rmse_loss.detach()
        total_bc_loss += bc_loss.detach()
        num_batches += 1

    accelerator.wait_for_everyone()

    total_pde_loss = accelerator.reduce(total_pde_loss, reduction='sum')
    total_rmse_loss = accelerator.reduce(total_rmse_loss, reduction='sum')
    total_bc_loss = accelerator.reduce(total_bc_loss, reduction='sum')
    num_batches = accelerator.reduce(num_batches, reduction='sum')

    model.train()

    n = num_batches.item()
    return (
        (total_pde_loss / n).item() if n > 0 else 0,
        (total_rmse_loss / n).item() if n > 0 else 0,
        (total_bc_loss / n).item() if n > 0 else 0,
    )


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    global_step: int,
    metrics: Dict,
    save_path: str,
    config: Dict,
):
    """Save model checkpoint."""
    checkpoint = {
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'config': config,
    }
    torch.save(checkpoint, save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/unet_baseline_flow_v2.yaml')
    args = parser.parse_args()

    # Set NCCL timeout to avoid false timeouts
    os.environ.setdefault('NCCL_TIMEOUT', '1800')  # 30 minutes
    os.environ.setdefault('NCCL_BLOCKING_WAIT', '0')

    config = load_config(args.config)
    set_seed(config['dataset']['seed'])

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)

    accelerator = Accelerator(
        mixed_precision=config['training'].get('mixed_precision', 'no'),
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1),
        log_with="wandb",
        kwargs_handlers=[ddp_kwargs]
    )

    # Training params
    max_epochs = config['training'].get('max_epochs', 50)
    warmup_steps = config['training'].get('warmup_steps', 200)
    lambda_pde = config['training'].get('lambda_pde', 1.0)
    lambda_rmse = config['training'].get('lambda_rmse', 1.0)
    lambda_bc = config['training'].get('lambda_bc', 1.0)
    grad_clip = config['training'].get('grad_clip', 1.0)
    eval_interval = config['training'].get('eval_interval', 100)
    early_stopping_patience = config['training'].get('early_stopping_patience', 30)
    log_interval = config['logging'].get('log_interval', 10)
    pde_version = config['training'].get('pde_version', 'v1')  # v1=2nd upwind, v2=central diff

    # Get model type for logging
    model_type = config.get('model', {}).get('type', 'fno')

    if accelerator.is_main_process:
        logger.info(f"{'='*60}")
        logger.info(f"Baseline Training - Flow Mixing Equation")
        logger.info(f"{'='*60}")
        logger.info(f"Model Type: {model_type.upper()}")
        logger.info(f"PDE Version: {pde_version} ({'central diff' if pde_version == 'v2' else '2nd upwind'})")
        logger.info(f"Config: {args.config}")
        logger.info(f"Max Epochs: {max_epochs}")
        logger.info(f"Warmup Steps: {warmup_steps}")
        logger.info(f"Learning Rate: {config['training']['learning_rate']}")
        logger.info(f"Loss: lambda_pde={lambda_pde}, lambda_rmse={lambda_rmse}, lambda_bc={lambda_bc}")
        logger.info(f"{'='*60}")

    # Create dataloaders
    train_loader, val_loader, train_sampler, val_sampler = create_dataloaders(config)

    # Calculate total steps
    steps_per_epoch = len(train_loader)
    total_steps = max_epochs * steps_per_epoch

    # Create model (UNet or MLP baseline)
    model_type = config.get('model', {}).get('type', 'unet')
    if model_type == 'mlp':
        model = create_mlp_baseline(config)
        model_name = "MLP Baseline (SIREN)"
    elif model_type == 'unet':
        model = create_unet_baseline(config)
        model_name = "UNet Baseline (Tanh)"
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported: 'unet', 'mlp'")

    # Log parameter counts
    if accelerator.is_main_process:
        params = model.count_parameters()
        print(f"\n{'='*60}")
        print(f"{model_name} Parameters")
        print(f"{'='*60}")
        for key, value in params.items():
            if key != 'total':
                print(f"{key:>15}: {value:>10,}")
        print(f"{'-'*60}")
        print(f"{'Total':>15}: {params['total']:>10,}")
        print(f"{'='*60}\n")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 1e-4),
        betas=tuple(config['training'].get('betas', [0.9, 0.999]))
    )

    scheduler = get_lr_scheduler(optimizer, config, total_steps)

    model, optimizer = accelerator.prepare(model, optimizer)

    # Init WandB
    if accelerator.is_main_process:
        run_name = f"{model_type}-baseline-flow-pde{lambda_pde}-rmse{lambda_rmse}"
        accelerator.init_trackers(
            project_name=config['logging']['project'],
            config=config,
            init_kwargs={"wandb": {
                "entity": config['logging'].get('entity'),
                "name": run_name,
                "tags": [f"{model_type}-baseline", "flow-mixing", "2nd-order-upwind"],
            }}
        )

    # Save directory
    save_dir = Path(config['logging']['save_dir'])
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Training state
    global_step = 0
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop = False

    console = Console()

    if accelerator.is_main_process:
        logger.info(f"Steps per epoch: {steps_per_epoch}")
        logger.info(f"Total steps: {total_steps}")

    model.train()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=not accelerator.is_main_process,
    ) as progress:
        train_task = progress.add_task("Flow Baseline", total=total_steps)

        for epoch in range(max_epochs):
            if early_stop:
                break

            train_sampler.set_epoch(epoch)

            epoch_pde_loss = 0.0
            epoch_rmse_loss = 0.0
            epoch_bc_loss = 0.0
            epoch_steps = 0

            for batch in train_loader:
                if early_stop:
                    break

                # Only use first channel (Flow Mixing has 1 real channel)
                data = batch['data'][..., :1].to(device=accelerator.device, dtype=torch.float32)
                input_data = data[:, :-1]
                target = data[:, 1:]

                in_warmup = global_step < warmup_steps

                with accelerator.accumulate(model):
                    output = model(data)

                    # PDE loss
                    pde_loss, loss_time, loss_advection = compute_pde_loss(
                        output, input_data, batch, config, accelerator, pde_version
                    )

                    # RMSE loss
                    rmse_loss = compute_rmse_loss(output, target)

                    # Boundary loss
                    bc_loss = compute_boundary_loss(output, target)

                    # Total loss
                    train_loss = lambda_pde * pde_loss + lambda_bc * bc_loss

                    # NaN check - all ranks must agree to skip
                    has_nan = torch.isnan(train_loss) or torch.isinf(train_loss)
                    has_nan_tensor = torch.tensor([1.0 if has_nan else 0.0], device=accelerator.device)
                    has_nan_any = accelerator.reduce(has_nan_tensor, reduction='sum')

                    if has_nan_any.item() > 0:
                        if accelerator.is_main_process:
                            console.print(f"[red]NaN/Inf detected at step {global_step}, skipping batch[/red]")
                        optimizer.zero_grad()
                        scheduler.step()
                        global_step += 1
                        continue

                    accelerator.backward(train_loss)

                    if grad_clip > 0:
                        accelerator.clip_grad_norm_(model.parameters(), grad_clip)

                    optimizer.step()
                    optimizer.zero_grad()

                scheduler.step()
                global_step += 1
                epoch_steps += 1

                # Reduce for logging
                pde_loss_reduced = accelerator.reduce(pde_loss.detach(), reduction='mean')
                rmse_loss_reduced = accelerator.reduce(rmse_loss.detach(), reduction='mean')
                bc_loss_reduced = accelerator.reduce(bc_loss.detach(), reduction='mean')

                epoch_pde_loss += pde_loss_reduced.item()
                epoch_rmse_loss += rmse_loss_reduced.item()
                epoch_bc_loss += bc_loss_reduced.item()

                # Update progress
                phase_str = "[warmup]" if in_warmup else f"[E{epoch+1}]"
                desc = f"{phase_str} PDE={pde_loss_reduced.item():.4f} RMSE={rmse_loss_reduced.item():.4f} BC={bc_loss_reduced.item():.4f}"
                desc += f" lr={scheduler.get_last_lr()[0]:.2e}"
                progress.update(train_task, advance=1, description=desc)

                # Log to wandb
                if global_step % log_interval == 0:
                    total_loss = lambda_pde * pde_loss_reduced + lambda_bc * bc_loss_reduced
                    log_dict = {
                        'train/pde_loss': pde_loss_reduced.item(),
                        'train/rmse_loss': rmse_loss_reduced.item(),  # For monitoring only
                        'train/bc_loss': bc_loss_reduced.item(),
                        'train/loss_time': accelerator.reduce(loss_time.detach(), reduction='mean').item(),
                        'train/loss_advection': accelerator.reduce(loss_advection.detach(), reduction='mean').item(),
                        'train/total_loss': total_loss.item(),
                        'train/lr': scheduler.get_last_lr()[0],
                        'train/epoch': epoch + 1,
                    }
                    accelerator.log(log_dict, step=global_step)

                # Validation
                if global_step % eval_interval == 0:
                    accelerator.wait_for_everyone()
                    val_pde, val_rmse, val_bc = validate(model, val_loader, config, accelerator, pde_version)

                    val_loss = lambda_pde * val_pde + lambda_bc * val_bc
                    accelerator.log({
                        'val/pde_loss': val_pde,
                        'val/rmse_loss': val_rmse,
                        'val/bc_loss': val_bc,
                        'val/total_loss': val_loss,
                    }, step=global_step)

                    if accelerator.is_main_process:
                        console.print(
                            f"\n[green]Step {global_step}/{total_steps} {phase_str}:[/green] "
                            f"val_pde={val_pde:.6f}, val_rmse={val_rmse:.6f}, val_bc={val_bc:.6f}"
                        )

                    if not in_warmup:
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            patience_counter = 0

                            if accelerator.is_main_process:
                                unwrapped_model = accelerator.unwrap_model(model)
                                save_checkpoint(
                                    model=unwrapped_model,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    global_step=global_step,
                                    metrics={'pde_loss': val_pde, 'rmse_loss': val_rmse, 'bc_loss': val_bc},
                                    save_path=str(save_dir / 'best.pt'),
                                    config=config,
                                )
                                console.print(f"[yellow]Saved best model[/yellow] (val_loss: {val_loss:.6f})")
                        else:
                            patience_counter += 1
                            if accelerator.is_main_process:
                                console.print(f"[dim]Patience: {patience_counter}/{early_stopping_patience}[/dim]")

                            if patience_counter >= early_stopping_patience:
                                if accelerator.is_main_process:
                                    console.print(f"[red]Early stopping triggered![/red]")
                                early_stop = True
                                break

                    model.train()

            # End of epoch
            if epoch_steps > 0:
                avg_pde = epoch_pde_loss / epoch_steps
                avg_rmse = epoch_rmse_loss / epoch_steps
                avg_bc = epoch_bc_loss / epoch_steps
                accelerator.log({
                    'epoch/train_pde_loss': avg_pde,
                    'epoch/train_rmse_loss': avg_rmse,
                    'epoch/train_bc_loss': avg_bc,
                    'epoch': epoch + 1
                }, step=global_step)

                if accelerator.is_main_process:
                    console.print(
                        f"\n[blue]Epoch {epoch+1}/{max_epochs} completed:[/blue] "
                        f"avg_pde={avg_pde:.6f}, avg_rmse={avg_rmse:.6f}, avg_bc={avg_bc:.6f}"
                    )

    accelerator.end_training()

    if accelerator.is_main_process:
        table = Table(title=f"{model_type.upper()} Baseline Training Complete (Flow)", show_header=False, border_style="green")
        table.add_row("Model Type", model_type.upper())
        table.add_row("Total Epochs", str(epoch + 1))
        table.add_row("Total Steps", str(global_step))
        table.add_row("Best Val Loss", f"{best_val_loss:.6f}")
        table.add_row("Early Stopped", "Yes" if early_stop else "No")
        table.add_row("Checkpoint", str(save_dir / "best.pt"))
        console.print(table)


if __name__ == "__main__":
    main()
