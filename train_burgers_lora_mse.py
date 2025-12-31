"""
LoRA Finetuning for 2D Burgers Equation with MSE Loss.

Features:
- MSE loss for training (pred vs GT)
- PDE loss for logging only (not used in backprop)
- LoRA applied to Transformer layers only
- Encoder and Decoder frozen
- Validation every 100 steps
- Resume from checkpoint support

Usage:
    # Fresh start
    torchrun --nproc_per_node=8 train_burgers_lora_mse.py --config configs/finetune_burgers.yaml

    # Resume from checkpoint
    torchrun --nproc_per_node=8 train_burgers_lora_mse.py --config configs/finetune_burgers_resume.yaml
"""

import os
import sys
import warnings

def _is_main_process():
    return os.environ.get('LOCAL_RANK', '0') == '0'

IS_MAIN_PROCESS = _is_main_process()

if not IS_MAIN_PROCESS:
    warnings.filterwarnings('ignore')

# Set triton cache
triton_cache = '/tmp/triton_cache'
os.makedirs(triton_cache, exist_ok=True)
os.environ.setdefault('TRITON_CACHE_DIR', triton_cache)

import argparse
import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.table import Table
import logging

torch.set_float32_matmul_precision('high')

if IS_MAIN_PROCESS:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
else:
    logging.disable(logging.CRITICAL)

from dataset_burgers import BurgersDataset, BurgersSampler, burgers_collate_fn
from model_lora import PDELoRAModel, save_lora_checkpoint, load_lora_checkpoint
from pde_loss import burgers_pde_loss_upwind

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA Finetuning for Burgers with MSE Loss")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_dataloaders(config: dict):
    """Create train and validation dataloaders."""
    clips_per_sample = config['dataset'].get('clips_per_sample', 100)

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
        clips_per_sample=clips_per_sample,
    )

    batch_size = config['dataloader']['batch_size']
    seed = config['dataset']['seed']

    train_sampler = BurgersSampler(train_dataset, batch_size, shuffle=True, seed=seed)
    val_sampler = BurgersSampler(val_dataset, batch_size, shuffle=False, seed=seed)

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=burgers_collate_fn,
        num_workers=config['dataloader']['num_workers'],
        pin_memory=config['dataloader']['pin_memory']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=burgers_collate_fn,
        num_workers=config['dataloader']['num_workers'],
        pin_memory=config['dataloader']['pin_memory']
    )

    return train_loader, val_loader, train_sampler, val_sampler


def get_lr_scheduler(optimizer, config, total_steps: int):
    """Create learning rate scheduler with warmup and cosine decay."""
    from torch.optim.lr_scheduler import LambdaLR
    import math

    warmup_steps = config['training'].get('warmup_steps', 100)
    min_lr_ratio = config['training'].get('min_lr', 1e-6) / config['training']['learning_rate']

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def compute_mse_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute MSE loss between pred and GT for u, v channels only.

    Args:
        output: [B, 16, H, W, 6] model prediction
        target: [B, 16, H, W, 6] ground truth (data[:, 1:])

    Returns:
        mse_loss: scalar MSE loss
    """
    # Extract u, v channels only
    pred_uv = output[..., :2].float()    # [B, 16, H, W, 2]
    target_uv = target[..., :2].float()  # [B, 16, H, W, 2]

    mse_loss = torch.mean((pred_uv - target_uv) ** 2)
    return mse_loss*1000


def compute_pde_loss(output, input_data, batch, config, accelerator):
    """
    Compute PDE residual loss for Burgers equation (for logging only).
    """
    t0_frame = input_data[:, 0:1]
    pred_with_t0 = torch.cat([t0_frame, output], dim=1)
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


def compute_constraint_error(output, input_data):
    """Compute u + v = 1.5 constraint error."""
    t0_frame = input_data[:, 0:1]
    pred_with_t0 = torch.cat([t0_frame, output], dim=1)

    u = pred_with_t0[..., 0]
    v = pred_with_t0[..., 1]

    constraint_error = torch.abs(u + v - 1.5).mean()
    return constraint_error


@torch.no_grad()
def validate(model, val_loader, config, accelerator):
    """
    Validate model on validation set.

    Returns:
        avg_mse_loss: Average MSE loss (main metric)
        avg_pde_loss: Average PDE loss (for logging)
        avg_constraint_error: Average constraint error
    """
    model.eval()

    total_mse_loss = torch.zeros(1, device=accelerator.device)
    total_pde_loss = torch.zeros(1, device=accelerator.device)
    total_constraint_error = torch.zeros(1, device=accelerator.device)
    num_batches = torch.zeros(1, device=accelerator.device)

    for batch in val_loader:
        data = batch['data'].to(device=accelerator.device, dtype=torch.bfloat16)

        input_data = data[:, :-1]   # [B, 16, H, W, 6]
        target_data = data[:, 1:]   # [B, 16, H, W, 6]

        output = model(input_data)

        # MSE loss
        mse_loss = compute_mse_loss(output, target_data)

        # PDE loss (for logging)
        pde_loss, _, _ = compute_pde_loss(output, input_data, batch, config, accelerator)

        # Constraint error
        constraint_error = compute_constraint_error(output, input_data)

        total_mse_loss += mse_loss.detach()
        total_pde_loss += pde_loss.detach()
        total_constraint_error += constraint_error.detach()
        num_batches += 1

    accelerator.wait_for_everyone()

    total_mse_loss = accelerator.reduce(total_mse_loss, reduction='sum')
    total_pde_loss = accelerator.reduce(total_pde_loss, reduction='sum')
    total_constraint_error = accelerator.reduce(total_constraint_error, reduction='sum')
    num_batches = accelerator.reduce(num_batches, reduction='sum')

    model.train()

    n = num_batches.item()
    avg_mse_loss = (total_mse_loss / n).item() if n > 0 else 0
    avg_pde_loss = (total_pde_loss / n).item() if n > 0 else 0
    avg_constraint_error = (total_constraint_error / n).item() if n > 0 else 0

    return avg_mse_loss, avg_pde_loss, avg_constraint_error


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config['dataset']['seed'])

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        mixed_precision=config['training'].get('mixed_precision', 'bf16'),
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1),
        log_with="wandb",
        kwargs_handlers=[ddp_kwargs]
    )

    # Training params
    max_epochs = config['training'].get('max_epochs', 10)
    warmup_steps = config['training'].get('warmup_steps', 100)
    log_interval = config['logging']['log_interval']
    eval_interval = config['training'].get('eval_interval', 100)  # Validate every 100 steps
    grad_clip = config['training'].get('grad_clip', 1.0)
    clips_per_sample = config['dataset'].get('clips_per_sample', 100)

    if accelerator.is_main_process:
        logger.info(f"{'='*60}")
        logger.info(f"Burgers2D LoRA Finetuning with MSE Loss")
        logger.info(f"{'='*60}")
        logger.info(f"Config: {args.config}")
        logger.info(f"Max Epochs: {max_epochs}")
        logger.info(f"Clips per Sample: {clips_per_sample}")
        logger.info(f"Warmup Steps: {warmup_steps}")
        logger.info(f"Eval Interval: {eval_interval} steps")
        logger.info(f"Loss: MSE (pred vs GT)")
        logger.info(f"Grad Clip: {grad_clip}")
        logger.info(f"{'='*60}")

    # Create dataloaders
    train_loader, val_loader, train_sampler, val_sampler = create_dataloaders(config)

    # Calculate total steps
    steps_per_epoch = len(train_loader)
    total_steps = max_epochs * steps_per_epoch

    # Create model with LoRA
    pretrained_path = config['model'].get('pretrained_path', None)
    model = PDELoRAModel(config, pretrained_path=pretrained_path)

    # Optimizer (only LoRA params)
    trainable_params = model.get_trainable_params()
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        betas=tuple(config['training'].get('betas', [0.9, 0.999]))
    )

    scheduler = get_lr_scheduler(optimizer, config, total_steps)

    # Prepare with accelerator
    model, optimizer = accelerator.prepare(model, optimizer)

    # Init WandB
    if accelerator.is_main_process:
        lora_r = config['model'].get('lora', {}).get('r', 16)
        run_name = f"burgers-lora-mse-r{lora_r}"
        accelerator.init_trackers(
            project_name=config['logging']['project'],
            config=config,
            init_kwargs={"wandb": {
                "entity": config['logging'].get('entity'),
                "name": run_name,
                "tags": ["burgers", "lora", "mse-loss"],
            }}
        )

    # Save directory
    save_dir = Path(config['logging']['save_dir'])
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Training state
    global_step = 0
    best_mse_loss = float('inf')
    start_epoch = 0

    # Resume from checkpoint if specified
    resume_path = config.get('resume', {}).get('checkpoint_path', None)
    reset_scheduler = config.get('resume', {}).get('reset_scheduler', False)

    if resume_path is not None and Path(resume_path).exists():
        if accelerator.is_main_process:
            logger.info(f"{'='*60}")
            logger.info(f"Resuming from checkpoint: {resume_path}")
            logger.info(f"Reset scheduler: {reset_scheduler}")
            logger.info(f"{'='*60}")

        # Load checkpoint (need to unwrap model first)
        unwrapped_model = accelerator.unwrap_model(model)

        # If reset_scheduler=True, don't restore scheduler state
        checkpoint = load_lora_checkpoint(
            model=unwrapped_model.model,
            checkpoint_path=resume_path,
            optimizer=optimizer,
            scheduler=None if reset_scheduler else scheduler,
        )

        # Restore training state
        global_step = checkpoint.get('global_step', 0)
        if 'metrics' in checkpoint and 'mse_loss' in checkpoint['metrics']:
            best_mse_loss = checkpoint['metrics']['mse_loss']

        # Calculate start epoch
        start_epoch = global_step // steps_per_epoch

        # If reset_scheduler, create new scheduler with new total_steps
        if reset_scheduler:
            remaining_steps = total_steps - global_step
            scheduler = get_lr_scheduler(optimizer, config, total_steps=remaining_steps)
            if accelerator.is_main_process:
                logger.info(f"Created new scheduler: warmup + cosine decay for {remaining_steps} steps")
                logger.info(f"New LR: {config['training']['learning_rate']} -> {config['training'].get('min_lr', 1e-6)}")

        if accelerator.is_main_process:
            logger.info(f"Resumed from step {global_step}, epoch {start_epoch}")
            logger.info(f"Best MSE loss so far: {best_mse_loss:.6f}")
            logger.info(f"{'='*60}")

    console = Console()

    if accelerator.is_main_process:
        logger.info(f"Steps per epoch: {steps_per_epoch}")
        logger.info(f"Total steps: {total_steps}")
        if resume_path:
            logger.info(f"Starting from epoch {start_epoch}, step {global_step}")

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
        train_task = progress.add_task("Training", total=total_steps, completed=global_step)

        for epoch in range(start_epoch, max_epochs):
            train_sampler.set_epoch(epoch)

            # Calculate how many steps to skip in this epoch (for resume)
            steps_done_in_epoch = global_step - epoch * steps_per_epoch if epoch == start_epoch else 0
            batch_idx = 0

            for batch in train_loader:
                # Skip already processed batches (for resume)
                if batch_idx < steps_done_in_epoch:
                    batch_idx += 1
                    continue
                batch_idx += 1

                data = batch['data'].to(device=accelerator.device, dtype=torch.bfloat16)

                input_data = data[:, :-1]   # [B, 16, H, W, 6]
                target_data = data[:, 1:]   # [B, 16, H, W, 6]

                with accelerator.accumulate(model):
                    output = model(input_data)

                    # MSE loss for training
                    mse_loss = compute_mse_loss(output, target_data)

                    # PDE loss for logging only (no grad)
                    with torch.no_grad():
                        pde_loss, loss_u, loss_v = compute_pde_loss(
                            output, input_data, batch, config, accelerator
                        )

                    accelerator.backward(mse_loss)

                    if grad_clip > 0:
                        accelerator.clip_grad_norm_(trainable_params, grad_clip)

                    optimizer.step()
                    optimizer.zero_grad()

                scheduler.step()
                global_step += 1

                # Reduce losses for logging
                mse_loss_reduced = accelerator.reduce(mse_loss.detach(), reduction='mean')
                pde_loss_reduced = accelerator.reduce(pde_loss.detach(), reduction='mean')

                # Update progress bar
                in_warmup = global_step < warmup_steps
                phase_str = "[warmup]" if in_warmup else f"[E{epoch+1}]"
                progress.update(
                    train_task, advance=1,
                    description=f"{phase_str} MSE={mse_loss_reduced.item():.6f} PDE={pde_loss_reduced.item():.2f}"
                )

                # Log to wandb
                if global_step % log_interval == 0:
                    accelerator.log({
                        'train/mse_loss': mse_loss_reduced.item(),
                        'train/pde_loss': pde_loss_reduced.item(),
                        'train/loss_u': accelerator.reduce(loss_u.detach(), reduction='mean').item(),
                        'train/loss_v': accelerator.reduce(loss_v.detach(), reduction='mean').item(),
                        'train/lr': scheduler.get_last_lr()[0],
                        'train/epoch': epoch + 1,
                    }, step=global_step)

                # Validate every eval_interval steps
                if global_step % eval_interval == 0:
                    accelerator.wait_for_everyone()

                    val_mse, val_pde, val_constraint = validate(
                        model, val_loader, config, accelerator
                    )

                    accelerator.log({
                        'val/mse_loss': val_mse,
                        'val/pde_loss': val_pde,
                        'val/constraint_error': val_constraint,
                        'val/step': global_step,
                    }, step=global_step)

                    if accelerator.is_main_process:
                        console.print(
                            f"\n[cyan]Step {global_step}:[/cyan] "
                            f"val_mse={val_mse:.6f}, val_pde={val_pde:.2f}, "
                            f"constraint={val_constraint:.6f}"
                        )

                    # Save best model (by val_mse, after warmup)
                    if global_step >= warmup_steps and val_mse < best_mse_loss:
                        best_mse_loss = val_mse

                        if accelerator.is_main_process:
                            unwrapped_model = accelerator.unwrap_model(model)
                            save_lora_checkpoint(
                                model=unwrapped_model.model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                global_step=global_step,
                                metrics={
                                    'mse_loss': val_mse,
                                    'pde_loss': val_pde,
                                    'constraint_error': val_constraint
                                },
                                save_path=str(save_dir / 'best_lora.pt'),
                                config=config,
                            )
                            console.print(f"[yellow]Saved best model[/yellow] (mse_loss: {val_mse:.6f})")

    accelerator.end_training()

    if accelerator.is_main_process:
        table = Table(title="Training Complete", show_header=False, border_style="green")
        table.add_row("Total Steps", str(global_step))
        table.add_row("Best Val MSE Loss", f"{best_mse_loss:.6f}")
        table.add_row("Checkpoint", str(save_dir / "best_lora.pt"))
        console.print(table)


if __name__ == "__main__":
    main()
