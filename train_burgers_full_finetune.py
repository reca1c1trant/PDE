"""
Burgers2D Full Finetune Training Script (No LoRA).

For comparison with LoRA finetuning to determine if poor performance
is due to LoRA limitations or model capacity.

Differences from LoRA version:
- No PEFT/LoRA, directly finetune transformer parameters
- Encoder/Decoder frozen (same as LoRA for fair comparison)

Usage:
    # Single GPU
    python train_burgers_full_finetune.py --config configs/finetune_burgers_full.yaml

    # Multi-GPU
    torchrun --nproc_per_node=8 train_burgers_full_finetune.py --config configs/finetune_burgers_full.yaml
"""

import os
import argparse
import yaml
import logging
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from pipeline import PDECausalModel
from dataset_burgers import BurgersDataset, BurgersSampler, burgers_collate_fn
from pde_loss import burgers_pde_loss_upwind

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_dataloaders(config: dict):
    """Create train and validation dataloaders."""
    clips_per_sample = config['dataset'].get('clips_per_sample', 120)

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
        clips_per_sample=clips_per_sample // 10,
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


def create_model_full_finetune(config: dict) -> nn.Module:
    """
    Create PDECausalModel for full transformer finetuning.

    - Load pretrained weights
    - Freeze encoder and decoder
    - Keep transformer trainable
    """
    model = PDECausalModel(config)

    pretrained_path = config['model'].get('pretrained_path')
    if pretrained_path is not None:
        checkpoint = torch.load(pretrained_path, map_location='cpu')

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        if os.environ.get('LOCAL_RANK', '0') == '0':
            print(f"\n{'='*60}")
            print(f"Loaded pretrained weights from: {pretrained_path}")
            if missing:
                print(f"Missing keys: {len(missing)}")
            if unexpected:
                print(f"Unexpected keys: {len(unexpected)}")
            print(f"{'='*60}\n")

    # Freeze encoder and decoder (same as LoRA for fair comparison)
    for param in model.encoder_2d.parameters():
        param.requires_grad = False

    for param in model.decoder_2d.parameters():
        param.requires_grad = False

    # Keep transformer trainable
    for param in model.transformer.parameters():
        param.requires_grad = True

    # Enable gradient checkpointing if configured
    if config.get('model', {}).get('gradient_checkpointing', False):
        model.transformer.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # Log parameter counts
    if os.environ.get('LOCAL_RANK', '0') == '0':
        enc_total = sum(p.numel() for p in model.encoder_2d.parameters())
        dec_total = sum(p.numel() for p in model.decoder_2d.parameters())
        trans_total = sum(p.numel() for p in model.transformer.parameters())
        trans_train = sum(p.numel() for p in model.transformer.parameters() if p.requires_grad)

        total = enc_total + dec_total + trans_total
        trainable = trans_train

        print(f"\n{'='*60}")
        print(f"Full Finetune Model Parameters")
        print(f"{'='*60}")
        print(f"Encoder (frozen):     {enc_total:>12,}")
        print(f"Decoder (frozen):     {dec_total:>12,}")
        print(f"Transformer (train):  {trans_train:>12,}")
        print(f"{'-'*60}")
        print(f"Total:                {total:>12,}")
        print(f"Trainable:            {trainable:>12,}")
        print(f"Trainable ratio:      {100 * trainable / total:.2f}%")
        print(f"{'='*60}\n")

    return model


def get_trainable_params(model: nn.Module) -> List[nn.Parameter]:
    """Get trainable parameters (transformer only)."""
    return [p for p in model.parameters() if p.requires_grad]


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


def compute_pde_loss(output, input_data, batch, config, accelerator):
    """Compute PDE residual loss for Burgers equation."""
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
    """Compute u + v = 1.5 constraint error (for evaluation only)."""
    t0_frame = input_data[:, 0:1]
    pred_with_t0 = torch.cat([t0_frame, output], dim=1)

    u = pred_with_t0[..., 0]
    v = pred_with_t0[..., 1]

    constraint = u + v - 1.5
    return torch.mean(torch.abs(constraint))


@torch.no_grad()
def validate(model, val_loader, config, accelerator):
    """Run validation and compute metrics."""
    model.eval()

    total_pde_loss = 0.0
    total_constraint_error = 0.0
    num_batches = 0

    for batch in val_loader:
        data = batch['data'].to(device=accelerator.device, dtype=torch.bfloat16)
        input_data = data[:, :-1]

        output = model(input_data)

        pde_loss, _, _ = compute_pde_loss(output, input_data, batch, config, accelerator)
        constraint_error = compute_constraint_error(output, input_data)

        total_pde_loss += pde_loss.item()
        total_constraint_error += constraint_error.item()
        num_batches += 1

    model.train()

    avg_pde_loss = total_pde_loss / max(num_batches, 1)
    avg_constraint_error = total_constraint_error / max(num_batches, 1)

    return avg_pde_loss, avg_constraint_error


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
    parser.add_argument('--config', type=str, default='configs/finetune_burgers_full.yaml')
    args = parser.parse_args()

    config = load_config(args.config)

    accelerator = Accelerator(
        mixed_precision=config['training'].get('mixed_precision', 'bf16'),
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1),
        log_with="wandb",
    )

    max_epochs = config['training'].get('max_epochs', 10)
    lambda_pde = config['training'].get('lambda_pde', 1.0)
    grad_clip = config['training'].get('grad_clip', 1.0)
    warmup_steps = config['training'].get('warmup_steps', 100)

    if accelerator.is_main_process:
        logger.info(f"\n{'='*60}")
        logger.info(f"Burgers2D Full Finetune (No LoRA)")
        logger.info(f"{'='*60}")
        logger.info(f"Config: {args.config}")
        logger.info(f"Max Epochs: {max_epochs}")
        logger.info(f"Learning Rate: {config['training']['learning_rate']}")
        logger.info(f"Grad Clip: {grad_clip}")
        logger.info(f"{'='*60}")

    # Create dataloaders
    train_loader, val_loader, train_sampler, val_sampler = create_dataloaders(config)

    # Calculate total steps
    steps_per_epoch = len(train_loader)
    total_steps = max_epochs * steps_per_epoch

    # Create model
    model = create_model_full_finetune(config)

    # Optimizer (transformer params only)
    trainable_params = get_trainable_params(model)
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
        run_name = "burgers-full-finetune"
        accelerator.init_trackers(
            project_name=config['logging']['project'],
            config=config,
            init_kwargs={"wandb": {
                "entity": config['logging'].get('entity'),
                "name": run_name,
                "tags": ["burgers", "full-finetune", "pde-loss"],
            }}
        )

    # Save directory
    save_dir = Path(config['logging']['save_dir'])
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Training state
    global_step = 0
    best_pde_loss = float('inf')

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
        console=console,
        disable=not accelerator.is_main_process
    ) as progress:

        task = progress.add_task("Training", total=total_steps)

        for epoch in range(max_epochs):
            train_sampler.set_epoch(epoch)

            epoch_pde_loss = 0.0
            epoch_steps = 0

            for batch in train_loader:
                data = batch['data'].to(device=accelerator.device, dtype=torch.bfloat16)
                input_data = data[:, :-1]

                with accelerator.accumulate(model):
                    output = model(input_data)

                    pde_loss, loss_u, loss_v = compute_pde_loss(
                        output, input_data, batch, config, accelerator
                    )

                    train_loss = lambda_pde * pde_loss

                    accelerator.backward(train_loss)

                    if grad_clip > 0:
                        accelerator.clip_grad_norm_(trainable_params, grad_clip)

                    optimizer.step()
                    optimizer.zero_grad()

                scheduler.step()
                global_step += 1
                epoch_steps += 1

                pde_loss_reduced = accelerator.reduce(pde_loss.detach(), reduction='mean')
                epoch_pde_loss += pde_loss_reduced.item()

                if global_step % config['logging'].get('log_interval', 10) == 0:
                    current_lr = scheduler.get_last_lr()[0]

                    accelerator.log({
                        'train/pde_loss': pde_loss_reduced.item(),
                        'train/loss_u': accelerator.reduce(loss_u.detach(), reduction='mean').item(),
                        'train/loss_v': accelerator.reduce(loss_v.detach(), reduction='mean').item(),
                        'train/lr': current_lr,
                        'train/epoch': epoch + 1,
                    }, step=global_step)

                progress.update(task, completed=global_step,
                              description=f"[bold blue]Epoch {epoch+1}/{max_epochs} | PDE: {pde_loss_reduced.item():.4f}")

            # Epoch end - validation
            avg_epoch_loss = epoch_pde_loss / epoch_steps

            val_pde_loss, val_constraint_error = validate(
                model, val_loader, config, accelerator
            )

            accelerator.log({
                'val/pde_loss': val_pde_loss,
                'val/constraint_error': val_constraint_error,
                'train/epoch_loss': avg_epoch_loss,
                'epoch': epoch + 1,
            }, step=global_step)

            if accelerator.is_main_process:
                console.print(
                    f"\n[green]Epoch {epoch+1}/{max_epochs}:[/green] "
                    f"train_pde={avg_epoch_loss:.6f}, val_pde={val_pde_loss:.6f}, "
                    f"constraint_err={val_constraint_error:.6f}"
                )

            # Save best model
            if global_step >= warmup_steps and val_pde_loss < best_pde_loss:
                best_pde_loss = val_pde_loss

                if accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(model)
                    save_checkpoint(
                        model=unwrapped_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        global_step=global_step,
                        metrics={'pde_loss': val_pde_loss, 'constraint_error': val_constraint_error},
                        save_path=str(save_dir / 'best.pt'),
                        config=config,
                    )
                    console.print(f"[yellow]Saved best model[/yellow] (pde_loss: {val_pde_loss:.6f})")

    accelerator.end_training()

    if accelerator.is_main_process:
        table = Table(title="Training Complete", show_header=False, border_style="green")
        table.add_row("Total Steps", str(global_step))
        table.add_row("Best Val PDE Loss", f"{best_pde_loss:.6f}")
        table.add_row("Checkpoint", str(save_dir / "best.pt"))
        console.print(table)


if __name__ == "__main__":
    main()
