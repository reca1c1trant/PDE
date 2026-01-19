"""
FNO Baseline Training Script V2 - Flow Mixing Equation.

V2 improvements:
- Model: Residual skip, Positional embedding, ChannelMLP (PINO-aligned)
- Training: Progressive PDE weight

Loss = lambda_rmse * RMSE + lambda_pde(t) * PDE

Key difference from Burgers: in_channels=1, out_channels=1

Usage:
    torchrun --nproc_per_node=8 train_fno_baseline_flow_v2.py --config configs/fno_baseline_flow_v2.yaml
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

from model_fno_baseline import create_fno_baseline
from dataset_flow import FlowMixingDataset, FlowMixingSampler, flow_mixing_collate_fn
from pde_loss_flow import flow_mixing_pde_loss
from pde_loss_fourier_v2 import boundary_loss

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

    warmup_steps = config['training'].get('warmup_steps', 200)
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


def get_progressive_pde_weight(
    step: int,
    pde_warmup_steps: int,
    lambda_pde_max: float,
) -> float:
    """Progressive PDE weight: 0 → lambda_pde_max over pde_warmup_steps."""
    if step < pde_warmup_steps:
        return lambda_pde_max * (step / pde_warmup_steps)
    else:
        return lambda_pde_max


def compute_pde_loss(output, input_data, batch, config, accelerator):
    """Compute PDE loss for Flow Mixing equation.

    Returns:
        pde_loss: scalar PDE loss
        loss_time: time derivative component
        loss_advection: advection component
    """
    # Prepend t0 frame
    t0_frame = input_data[:, 0:1]  # [B, 1, H, W, C]
    pred_with_t0 = torch.cat([t0_frame, output], dim=1)  # [B, T, H, W, C]

    # Extract u channel (only channel for flow)
    pred_u = pred_with_t0[..., :1].float()  # [B, T, H, W, 1]

    # Get boundaries
    boundary_left = batch['boundary_left'].to(accelerator.device).float()
    boundary_right = batch['boundary_right'].to(accelerator.device).float()
    boundary_bottom = batch['boundary_bottom'].to(accelerator.device).float()
    boundary_top = batch['boundary_top'].to(accelerator.device).float()
    vtmax = batch['vtmax'].to(accelerator.device).float()

    dt = config.get('physics', {}).get('dt', 1/999)
    vtmax_mean = vtmax.mean().item()

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
    """Compute RMSE loss."""
    # Only channel 0 (u) for flow
    pred_u = output[..., 0]
    target_u = target[..., 0]
    mse = torch.mean((pred_u - target_u) ** 2)
    rmse = torch.sqrt(mse + 1e-8)
    return rmse


def compute_boundary_loss(output, target):
    """Compute boundary loss (same as burgers)."""
    return boundary_loss(output, target)


@torch.no_grad()
def validate(model, val_loader, config, accelerator, lambda_pde, lambda_bc):
    """Run validation and compute metrics."""
    model.eval()

    total_pde_loss = torch.zeros(1, device=accelerator.device)
    total_rmse_loss = torch.zeros(1, device=accelerator.device)
    total_bc_loss = torch.zeros(1, device=accelerator.device)
    num_batches = torch.zeros(1, device=accelerator.device)

    for batch in val_loader:
        data = batch['data'].to(device=accelerator.device, dtype=torch.float32)
        # Flow data is padded to 6 channels, extract only first channel
        data = data[..., :1]  # [B, T, H, W, 1]
        input_data = data[:, :-1]
        target = data[:, 1:]

        output = model(data)

        pde_loss, _, _ = compute_pde_loss(output, input_data, batch, config, accelerator)
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
    parser.add_argument('--config', type=str, default='configs/fno_baseline_flow_v2.yaml')
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config['dataset']['seed'])

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)

    accelerator = Accelerator(
        mixed_precision=config['training'].get('mixed_precision', 'bf16'),
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1),
        log_with="wandb",
        kwargs_handlers=[ddp_kwargs]
    )

    # Training params
    max_epochs = config['training'].get('max_epochs', 50)
    warmup_steps = config['training'].get('warmup_steps', 200)
    pde_warmup_steps = config['training'].get('pde_warmup_steps', 1000)
    lambda_pde_max = config['training'].get('lambda_pde', 1.0)
    lambda_rmse = config['training'].get('lambda_rmse', 1.0)
    lambda_bc = config['training'].get('lambda_bc', 1.0)
    grad_clip = config['training'].get('grad_clip', 1.0)
    eval_interval = config['training'].get('eval_interval', 100)
    early_stopping_patience = config['training'].get('early_stopping_patience', 30)
    log_interval = config['logging'].get('log_interval', 10)

    # Model options
    use_pos_embed = config['model'].get('use_positional_embedding', True)
    use_channel_mlp = config['model'].get('use_channel_mlp', True)

    if accelerator.is_main_process:
        logger.info(f"{'='*60}")
        logger.info(f"FNO Baseline V2 - Flow Mixing (PINO-aligned)")
        logger.info(f"{'='*60}")
        logger.info(f"Config: {args.config}")
        logger.info(f"Model: in_channels=1, out_channels=1")
        logger.info(f"Model: Residual + PosEmbed({use_pos_embed}) + ChannelMLP({use_channel_mlp})")
        logger.info(f"PDE Loss: Flow Mixing (advection)")
        logger.info(f"PDE Warmup: {pde_warmup_steps} steps (0 → {lambda_pde_max})")
        logger.info(f"RMSE Loss: lambda_rmse={lambda_rmse}")
        logger.info(f"Boundary Loss: lambda_bc={lambda_bc}")
        logger.info(f"{'='*60}")

    # Create dataloaders
    train_loader, val_loader, train_sampler, val_sampler = create_dataloaders(config)

    steps_per_epoch = len(train_loader)
    total_steps = max_epochs * steps_per_epoch

    # Create model
    model = create_fno_baseline(config)

    if accelerator.is_main_process:
        params = model.count_parameters()
        print(f"\n{'='*60}")
        print(f"FNO Baseline V2 Model Parameters (Flow)")
        print(f"{'='*60}")
        print(f"Pos Embed:  {params['positional_embedding']:>10,}")
        print(f"Lifting:    {params['lifting']:>10,}")
        print(f"FNO Blocks: {params['fno_blocks']:>10,}")
        print(f"Projection: {params['projection']:>10,}")
        print(f"{'-'*60}")
        print(f"Total:      {params['total']:>10,}")
        print(f"{'='*60}\n")

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
        run_name = f"fno-v2-flow-residual-posembed-mlp"
        accelerator.init_trackers(
            project_name=config['logging']['project'],
            config=config,
            init_kwargs={"wandb": {
                "entity": config['logging'].get('entity'),
                "name": run_name,
                "tags": ["fno-v2", "flow-mixing", "pino-aligned"],
            }}
        )

    save_dir = Path(config['logging']['save_dir'])
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop = False

    console = Console()

    if accelerator.is_main_process:
        logger.info(f"Steps per epoch: {steps_per_epoch}")
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"PDE warmup steps: {pde_warmup_steps}")

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
        train_task = progress.add_task("FNO V2 Flow", total=total_steps)

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

                data = batch['data'].to(device=accelerator.device, dtype=torch.float32)
                # Flow data is padded to 6 channels, extract only first channel
                data = data[..., :1]  # [B, T, H, W, 1]
                input_data = data[:, :-1]
                target = data[:, 1:]

                in_warmup = global_step < warmup_steps

                # Progressive PDE weight
                lambda_pde = get_progressive_pde_weight(
                    global_step, pde_warmup_steps, lambda_pde_max
                )

                with accelerator.accumulate(model):
                    output = model(data)

                    # Losses
                    pde_loss, loss_time, loss_advection = compute_pde_loss(
                        output, input_data, batch, config, accelerator
                    )
                    rmse_loss = compute_rmse_loss(output, target)
                    bc_loss = compute_boundary_loss(output, target)

                    # Total loss
                    train_loss = (
                        lambda_rmse * rmse_loss +
                        lambda_pde * pde_loss +
                        lambda_bc * bc_loss
                    )

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
                desc = f"{phase_str} PDE={pde_loss_reduced.item():.4f}(w={lambda_pde:.2f}) RMSE={rmse_loss_reduced.item():.4f} BC={bc_loss_reduced.item():.4f}"
                progress.update(train_task, advance=1, description=desc)

                # Log to wandb
                if global_step % log_interval == 0:
                    log_dict = {
                        'train/pde_loss': pde_loss_reduced.item(),
                        'train/rmse_loss': rmse_loss_reduced.item(),
                        'train/bc_loss': bc_loss_reduced.item(),
                        'train/lambda_pde': lambda_pde,
                        'train/loss_time': accelerator.reduce(loss_time.detach(), reduction='mean').item(),
                        'train/loss_advection': accelerator.reduce(loss_advection.detach(), reduction='mean').item(),
                        'train/total_loss': (
                            lambda_rmse * rmse_loss_reduced +
                            lambda_pde * pde_loss_reduced +
                            lambda_bc * bc_loss_reduced
                        ).item(),
                        'train/lr': scheduler.get_last_lr()[0],
                        'train/epoch': epoch + 1,
                    }
                    accelerator.log(log_dict, step=global_step)

                # Validation
                if global_step % eval_interval == 0:
                    accelerator.wait_for_everyone()
                    val_pde, val_rmse, val_bc = validate(
                        model, val_loader, config, accelerator, lambda_pde, lambda_bc
                    )

                    accelerator.log({
                        'val/pde_loss': val_pde,
                        'val/rmse_loss': val_rmse,
                        'val/bc_loss': val_bc,
                        'val/total_loss': lambda_rmse * val_rmse + lambda_pde * val_pde + lambda_bc * val_bc,
                    }, step=global_step)

                    if accelerator.is_main_process:
                        console.print(
                            f"\n[green]Step {global_step}/{total_steps} {phase_str}:[/green] "
                            f"val_pde={val_pde:.6f}, val_rmse={val_rmse:.6f}, val_bc={val_bc:.6f}"
                        )

                    val_loss = lambda_rmse * val_rmse + lambda_pde * val_pde + lambda_bc * val_bc

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
                                    metrics={
                                        'pde_loss': val_pde,
                                        'rmse_loss': val_rmse,
                                        'bc_loss': val_bc,
                                    },
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

                    # Sync early_stop across all ranks to prevent NCCL timeout
                    early_stop_tensor = torch.tensor([1 if early_stop else 0], device=accelerator.device)
                    early_stop_tensor = accelerator.reduce(early_stop_tensor, reduction='max')
                    early_stop = early_stop_tensor.item() > 0

                    if early_stop:
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
        table = Table(title="FNO V2 Flow Training Complete", show_header=False, border_style="green")
        table.add_row("Total Epochs", str(epoch + 1))
        table.add_row("Total Steps", str(global_step))
        table.add_row("Best Val Loss", f"{best_val_loss:.6f}")
        table.add_row("Early Stopped", "Yes" if early_stop else "No")
        table.add_row("Checkpoint", str(save_dir / "best.pt"))
        console.print(table)


if __name__ == "__main__":
    main()
