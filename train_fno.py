"""
FNO (Fourier Neural Operator) Training for PDE.

Train FNO model from scratch on Burgers or Flow Mixing equation with:
- PDE residual loss (physics-informed)
- RMSE loss (supervised, optional)

Key advantages over Llama-based model:
1. Operates in frequency domain (native to PDE solving)
2. No downsampling (preserves spatial resolution)
3. Much fewer parameters (~2-5M vs 80M)

Usage:
    # Single GPU
    python train_fno.py --config configs/fno_burgers.yaml

    # Multi-GPU
    torchrun --nproc_per_node=8 train_fno.py --config configs/fno_burgers.yaml

    # Flow Mixing
    python train_fno.py --config configs/fno_flow.yaml
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
from typing import Dict, Optional

import torch
import torch.nn as nn
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

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_dataloaders(config: dict, task: str = "burgers"):
    """Create train and validation dataloaders based on task."""
    clips_per_sample = config['dataset'].get('clips_per_sample', None)

    if task == "burgers":
        from dataset_burgers import BurgersDataset, BurgersSampler, burgers_collate_fn

        train_dataset = BurgersDataset(
            data_path=config['dataset']['path'],
            temporal_length=config['dataset']['temporal_length'],
            split='train',
            train_ratio=config['dataset']['train_ratio'],
            seed=config['dataset']['seed'],
            clips_per_sample=clips_per_sample,
        )

        val_dataset = BurgersDataset(
            data_path=config['dataset']['path'],
            temporal_length=config['dataset']['temporal_length'],
            split='val',
            train_ratio=config['dataset']['train_ratio'],
            seed=config['dataset']['seed'],
            clips_per_sample=None,
        )

        batch_size = config['dataloader']['batch_size']
        seed = config['dataset']['seed']

        train_sampler = BurgersSampler(train_dataset, batch_size, shuffle=True, seed=seed)
        val_sampler = BurgersSampler(val_dataset, batch_size, shuffle=False, seed=seed)
        collate_fn = burgers_collate_fn

    elif task in ["flow_mixing", "flow"]:
        from dataset_flow import FlowMixingDataset, FlowMixingSampler, flow_collate_fn

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
        collate_fn = flow_collate_fn
    else:
        raise ValueError(f"Unknown task: {task}")

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=config['dataloader']['num_workers'],
        pin_memory=config['dataloader']['pin_memory']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=config['dataloader']['num_workers'],
        pin_memory=config['dataloader']['pin_memory']
    )

    return train_loader, val_loader, train_sampler, val_sampler


def create_model(config: dict, task: str = "burgers") -> nn.Module:
    """Create FNO model from config."""
    from fno_model import get_fno_model

    model_cfg = config.get('model', {})

    model = get_fno_model(
        task=task,
        modes=model_cfg.get('modes', 12),
        width=model_cfg.get('width', 32),
        n_layers=model_cfg.get('n_layers', 4),
        activation=model_cfg.get('activation', 'gelu'),
    )

    # Log parameter counts
    if IS_MAIN_PROCESS:
        total = model.count_parameters()
        print(f"\n{'='*60}")
        print(f"FNO Model for {task.upper()}")
        print(f"{'='*60}")
        print(f"Modes:       {model_cfg.get('modes', 12)}")
        print(f"Width:       {model_cfg.get('width', 32)}")
        print(f"Layers:      {model_cfg.get('n_layers', 4)}")
        print(f"Activation:  {model_cfg.get('activation', 'gelu')}")
        print(f"{'-'*60}")
        print(f"Total Params: {total:>12,}")
        print(f"{'='*60}\n")

    return model


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


def compute_pde_loss_burgers(output, input_data, batch, config, accelerator):
    """Compute PDE residual loss for Burgers equation."""
    from pde_loss import burgers_pde_loss_upwind

    # FNO output is [B, T-1, H, W, 6], add t0 frame for PDE computation
    t0_frame = input_data[:, 0:1]  # [B, 1, H, W, 6]
    pred_with_t0 = torch.cat([t0_frame, output], dim=1)  # [B, T, H, W, 6]
    pred_uv = pred_with_t0[..., :2].float()

    boundary_left = batch['boundary_left'].to(accelerator.device).float()
    boundary_right = batch['boundary_right'].to(accelerator.device).float()
    boundary_bottom = batch['boundary_bottom'].to(accelerator.device).float()
    boundary_top = batch['boundary_top'].to(accelerator.device).float()
    nu = batch['nu'].to(accelerator.device).float()

    dt = config.get('physics', {}).get('dt', 1/999)
    nu_mean = nu.mean().item()

    pde_loss, loss_u, loss_v, _, _ = burgers_pde_loss_upwind(
        pred=pred_uv,
        boundary_left=boundary_left,
        boundary_right=boundary_right,
        boundary_bottom=boundary_bottom,
        boundary_top=boundary_top,
        nu=nu_mean,
        dt=dt
    )

    return pde_loss, loss_u, loss_v


def compute_pde_loss_flow(output, input_data, batch, config, accelerator):
    """Compute PDE residual loss for Flow Mixing equation."""
    from pde_loss_flow import flow_mixing_pde_loss

    # FNO output is [B, T-1, H, W, 6], add t0 frame
    t0_frame = input_data[:, 0:1]
    pred_with_t0 = torch.cat([t0_frame, output], dim=1)
    pred_u = pred_with_t0[..., :1].float()

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


def compute_rmse_loss(output, target, task: str = "burgers"):
    """
    Compute RMSE loss between output and ground truth.

    Args:
        output: Model output [B, T-1, H, W, 6]
        target: Ground truth [B, T-1, H, W, 6]
        task: "burgers" (2 channels) or "flow" (1 channel)

    Returns:
        rmse_loss: RMSE loss on valid channels
    """
    if task == "burgers":
        pred = output[..., :2]
        gt = target[..., :2]
    else:
        pred = output[..., :1]
        gt = target[..., :1]

    mse = torch.mean((pred - gt) ** 2)
    rmse = torch.sqrt(mse + 1e-8)
    return rmse


def compute_constraint_error_burgers(output, input_data):
    """Compute u + v = 1.5 constraint error (Burgers only)."""
    t0_frame = input_data[:, 0:1]
    pred_with_t0 = torch.cat([t0_frame, output], dim=1)

    u = pred_with_t0[..., 0]
    v = pred_with_t0[..., 1]

    constraint = u + v - 1.5
    return torch.mean(torch.abs(constraint))


@torch.no_grad()
def validate(model, val_loader, config, accelerator, task: str = "burgers"):
    """Run validation and compute metrics."""
    model.eval()

    total_pde_loss = torch.zeros(1, device=accelerator.device)
    total_rmse_loss = torch.zeros(1, device=accelerator.device)
    total_constraint_error = torch.zeros(1, device=accelerator.device)
    num_batches = torch.zeros(1, device=accelerator.device)

    for batch in val_loader:
        data = batch['data'].to(device=accelerator.device, dtype=torch.float32)
        input_data = data[:, :-1]  # [B, T-1, H, W, 6] for input
        target = data[:, 1:]       # [B, T-1, H, W, 6] for target

        output = model(input_data)  # [B, T-2, H, W, 6]

        # Align target with output (output predicts from t[1] to t[T-1])
        target_aligned = target[:, :-1]  # [B, T-2, H, W, 6]

        if task == "burgers":
            pde_loss, _, _ = compute_pde_loss_burgers(
                output, input_data, batch, config, accelerator
            )
            constraint_error = compute_constraint_error_burgers(output, input_data)
        else:
            pde_loss, _, _ = compute_pde_loss_flow(
                output, input_data, batch, config, accelerator
            )
            constraint_error = torch.tensor(0.0, device=accelerator.device)

        rmse_loss = compute_rmse_loss(output, target_aligned, task)

        total_pde_loss += pde_loss.detach()
        total_rmse_loss += rmse_loss.detach()
        total_constraint_error += constraint_error.detach()
        num_batches += 1

    accelerator.wait_for_everyone()

    total_pde_loss = accelerator.reduce(total_pde_loss, reduction='sum')
    total_rmse_loss = accelerator.reduce(total_rmse_loss, reduction='sum')
    total_constraint_error = accelerator.reduce(total_constraint_error, reduction='sum')
    num_batches = accelerator.reduce(num_batches, reduction='sum')

    model.train()

    n = num_batches.item()
    return (
        (total_pde_loss / n).item() if n > 0 else 0,
        (total_rmse_loss / n).item() if n > 0 else 0,
        (total_constraint_error / n).item() if n > 0 else 0,
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
    """Save full model checkpoint."""
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
    parser.add_argument('--config', type=str, default='configs/fno_burgers.yaml')
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config['dataset']['seed'])

    # Determine task from config
    task = config.get('task', 'burgers')

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)

    accelerator = Accelerator(
        mixed_precision=config['training'].get('mixed_precision', 'no'),  # FNO works better with fp32
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1),
        log_with="wandb",
        kwargs_handlers=[ddp_kwargs]
    )

    # Training params
    max_epochs = config['training'].get('max_epochs', 100)
    warmup_steps = config['training'].get('warmup_steps', 200)
    lambda_pde = config['training'].get('lambda_pde', 1.0)
    lambda_rmse = config['training'].get('lambda_rmse', 0.0)
    grad_clip = config['training'].get('grad_clip', 1.0)
    eval_interval = config['training'].get('eval_interval', 100)
    early_stopping_patience = config['training'].get('early_stopping_patience', 20)
    log_interval = config['logging'].get('log_interval', 10)

    if accelerator.is_main_process:
        logger.info(f"{'='*60}")
        logger.info(f"FNO Training - {task.upper()}")
        logger.info(f"{'='*60}")
        logger.info(f"Config: {args.config}")
        logger.info(f"Task: {task}")
        logger.info(f"Max Epochs: {max_epochs}")
        logger.info(f"Warmup Steps: {warmup_steps}")
        logger.info(f"Learning Rate: {config['training']['learning_rate']}")
        logger.info(f"Loss: lambda_pde={lambda_pde}, lambda_rmse={lambda_rmse}")
        logger.info(f"Eval Interval: {eval_interval} steps")
        logger.info(f"Grad Clip: {grad_clip}")
        logger.info(f"{'='*60}")

    # Create dataloaders
    train_loader, val_loader, train_sampler, val_sampler = create_dataloaders(config, task)

    # Log dataset info
    if accelerator.is_main_process:
        clips_per_sample = config['dataset'].get('clips_per_sample', None)
        if clips_per_sample is None:
            logger.info(f"Training: ALL clips per sample (full traversal)")
        else:
            logger.info(f"Training: {clips_per_sample} clips per sample per epoch")

    # Calculate total steps
    steps_per_epoch = len(train_loader)
    total_steps = max_epochs * steps_per_epoch

    # Create FNO model
    model = create_model(config, task)

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
        run_name = f"fno-{task}-pde{lambda_pde}-rmse{lambda_rmse}"
        accelerator.init_trackers(
            project_name=config['logging']['project'],
            config=config,
            init_kwargs={"wandb": {
                "entity": config['logging'].get('entity'),
                "name": run_name,
                "tags": [task, "fno", "pde-loss"],
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
        train_task = progress.add_task("Training", total=total_steps)

        for epoch in range(max_epochs):
            if early_stop:
                break

            train_sampler.set_epoch(epoch)

            epoch_pde_loss = 0.0
            epoch_rmse_loss = 0.0
            epoch_steps = 0

            for batch in train_loader:
                if early_stop:
                    break

                # Use float32 for FNO (FFT operations)
                data = batch['data'].to(device=accelerator.device, dtype=torch.float32)
                input_data = data[:, :-1]  # [B, T-1, H, W, 6]
                target = data[:, 1:]       # [B, T-1, H, W, 6]

                in_warmup = global_step < warmup_steps

                with accelerator.accumulate(model):
                    output = model(input_data)  # [B, T-2, H, W, 6]

                    # Align target with output
                    target_aligned = target[:, :-1]

                    # PDE loss
                    if task == "burgers":
                        pde_loss, loss_u, loss_v = compute_pde_loss_burgers(
                            output, input_data, batch, config, accelerator
                        )
                    else:
                        pde_loss, loss_time, loss_advection = compute_pde_loss_flow(
                            output, input_data, batch, config, accelerator
                        )

                    # RMSE loss (optional)
                    rmse_loss = torch.tensor(0.0, device=accelerator.device)
                    if lambda_rmse > 0:
                        rmse_loss = compute_rmse_loss(output, target_aligned, task)

                    # Total loss
                    train_loss = lambda_pde * pde_loss + lambda_rmse * rmse_loss

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
                epoch_pde_loss += pde_loss_reduced.item()

                if lambda_rmse > 0:
                    rmse_loss_reduced = accelerator.reduce(rmse_loss.detach(), reduction='mean')
                    epoch_rmse_loss += rmse_loss_reduced.item()

                # Update progress
                phase_str = "[warmup]" if in_warmup else f"[E{epoch+1}]"
                desc = f"{phase_str} PDE={pde_loss_reduced.item():.6f}"
                if lambda_rmse > 0:
                    desc += f" RMSE={rmse_loss_reduced.item():.4f}"
                desc += f" lr={scheduler.get_last_lr()[0]:.2e}"
                progress.update(train_task, advance=1, description=desc)

                # Log to wandb
                if global_step % log_interval == 0:
                    log_dict = {
                        'train/pde_loss': pde_loss_reduced.item(),
                        'train/lr': scheduler.get_last_lr()[0],
                        'train/epoch': epoch + 1,
                    }
                    if task == "burgers":
                        log_dict['train/loss_u'] = accelerator.reduce(loss_u.detach(), reduction='mean').item()
                        log_dict['train/loss_v'] = accelerator.reduce(loss_v.detach(), reduction='mean').item()
                    else:
                        log_dict['train/loss_time'] = accelerator.reduce(loss_time.detach(), reduction='mean').item()
                        log_dict['train/loss_advection'] = accelerator.reduce(loss_advection.detach(), reduction='mean').item()

                    if lambda_rmse > 0:
                        log_dict['train/rmse_loss'] = rmse_loss_reduced.item()
                    accelerator.log(log_dict, step=global_step)

                # Step-based validation
                if global_step % eval_interval == 0:
                    accelerator.wait_for_everyone()
                    val_pde, val_rmse, val_constraint = validate(
                        model, val_loader, config, accelerator, task
                    )

                    log_dict = {
                        'val/pde_loss': val_pde,
                        'val/rmse_loss': val_rmse,
                    }
                    if task == "burgers":
                        log_dict['val/constraint_error'] = val_constraint
                    accelerator.log(log_dict, step=global_step)

                    if accelerator.is_main_process:
                        msg = f"\n[green]Step {global_step}/{total_steps} {phase_str}:[/green] "
                        msg += f"val_pde={val_pde:.6e}, val_rmse={val_rmse:.6f}"
                        if task == "burgers":
                            msg += f", constraint={val_constraint:.6f}"
                        console.print(msg)

                    # Use combined loss for best model selection
                    val_loss = lambda_pde * val_pde + lambda_rmse * val_rmse

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
                                        'constraint': val_constraint,
                                    },
                                    save_path=str(save_dir / 'best.pt'),
                                    config=config,
                                )
                                console.print(f"[yellow]Saved best model[/yellow] (val_loss: {val_loss:.6e})")
                        else:
                            patience_counter += 1
                            if accelerator.is_main_process:
                                console.print(f"[dim]Patience: {patience_counter}/{early_stopping_patience}[/dim]")

                            if patience_counter >= early_stopping_patience:
                                if accelerator.is_main_process:
                                    console.print(f"[red]Early stopping triggered![/red]")
                                early_stop = True
                                break
                    else:
                        if accelerator.is_main_process:
                            console.print(f"[dim](warmup phase - no model saving)[/dim]")

                    model.train()

            # End of epoch summary
            if epoch_steps > 0:
                avg_pde = epoch_pde_loss / epoch_steps
                log_dict = {'epoch/train_pde_loss': avg_pde, 'epoch': epoch + 1}
                if lambda_rmse > 0:
                    log_dict['epoch/train_rmse_loss'] = epoch_rmse_loss / epoch_steps
                accelerator.log(log_dict, step=global_step)

                if accelerator.is_main_process:
                    console.print(f"\n[blue]Epoch {epoch+1}/{max_epochs} completed:[/blue] avg_pde={avg_pde:.6e}")

    accelerator.end_training()

    if accelerator.is_main_process:
        table = Table(title="FNO Training Complete", show_header=False, border_style="green")
        table.add_row("Task", task)
        table.add_row("Total Epochs", str(epoch + 1))
        table.add_row("Total Steps", str(global_step))
        table.add_row("Best Val Loss", f"{best_val_loss:.6e}")
        table.add_row("Early Stopped", "Yes" if early_stop else "No")
        table.add_row("Checkpoint", str(save_dir / "best.pt"))
        console.print(table)


if __name__ == "__main__":
    main()
