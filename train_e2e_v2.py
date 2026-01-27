"""
End-to-End Training V2 for PDE Causal Model with Temporal Residual.

Key difference from V1:
- Uses PDECausalModelV2 with temporal residual (output = input + delta)
- Zero-init residual projection ensures stable training start
- Model learns time evolution Δu instead of full field u

Physical insight:
- For dt ≈ 0.001, |Δu| ≈ 0.001 while |u| ≈ 1.0
- Learning small delta is 1000x easier than learning full field
- Residual provides strong inductive bias matching PDE structure

Usage:
    OMP_NUM_THREADS=6 torchrun --nproc_per_node=8 train_e2e_v2.py --config configs/e2e_medium_v2.yaml
"""

import os
import sys
import warnings

def _is_main_process():
    return os.environ.get('LOCAL_RANK', '0') == '0'

IS_MAIN_PROCESS = _is_main_process()

if not IS_MAIN_PROCESS:
    warnings.filterwarnings('ignore')
else:
    warnings.filterwarnings('ignore', message='.*FSDP upcast.*')

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

from dataset import PDEDataset, DimensionGroupedSampler, collate_fn
from pipeline_v2 import PDECausalModelV2  # V2 model with temporal residual

logger = logging.getLogger(__name__)


def compute_nrmse_loss(pred: torch.Tensor, target: torch.Tensor, channel_mask: torch.Tensor):
    """
    Compute nRMSE following metrics.py style (per-sample, per-channel).

    For each sample i, channel c:
        RMSE_i_c = sqrt(mean((pred - target)², dim=spatial))
        RMS_i_c = sqrt(mean(target², dim=spatial))
        nRMSE_i_c = RMSE_i_c / RMS_i_c

    nRMSE = mean(nRMSE_i_c) over all samples and channels

    Args:
        pred: [B, T, H, W, C] predictions
        target: [B, T, H, W, C] ground truth
        channel_mask: [B, C] or [C] valid channel mask

    Returns:
        nrmse: scalar nRMSE loss for backprop
        rmse: scalar RMSE for logging
    """
    eps = 1e-8

    # Handle channel_mask shape
    if channel_mask.dim() == 2:
        valid_mask = channel_mask[0].bool()  # [C]
    else:
        valid_mask = channel_mask.bool()  # [C]

    # Filter valid channels
    pred_valid = pred[..., valid_mask]  # [B, T, H, W, C_valid]
    target_valid = target[..., valid_mask]  # [B, T, H, W, C_valid]

    B, T, H, W, C = pred_valid.shape

    # Reshape to [B, C, spatial] where spatial = T * H * W
    pred_flat = pred_valid.permute(0, 4, 1, 2, 3).reshape(B, C, -1)  # [B, C, T*H*W]
    target_flat = target_valid.permute(0, 4, 1, 2, 3).reshape(B, C, -1)  # [B, C, T*H*W]

    # Per (sample, channel) RMSE over spatial dim
    mse_per_bc = ((pred_flat - target_flat) ** 2).mean(dim=2)  # [B, C]
    rmse_per_bc = torch.sqrt(mse_per_bc + eps)  # [B, C]

    # Per (sample, channel) RMS (norm) over spatial dim
    rms_per_bc = torch.sqrt((target_flat ** 2).mean(dim=2) + eps)  # [B, C]

    # nRMSE per (sample, channel)
    nrmse_per_bc = rmse_per_bc / rms_per_bc  # [B, C]

    # Average over all samples and channels
    nrmse = nrmse_per_bc.mean()

    # For logging: overall RMSE (mean of per-sample-channel RMSE)
    rmse = rmse_per_bc.mean()

    return nrmse, rmse


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-End PDE Training V2 (Temporal Residual)")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_dataloaders(config: dict):
    clips_per_sample = config['dataset'].get('clips_per_sample', 80)

    train_dataset = PDEDataset(
        data_dir=config['dataset']['path'],
        temporal_length=config['dataset']['temporal_length'],
        split='train',
        train_ratio=config['dataset']['train_ratio'],
        seed=config['dataset']['seed'],
        clips_per_sample=clips_per_sample
    )

    # Validation uses ALL clips (clips_per_sample=None)
    val_dataset = PDEDataset(
        data_dir=config['dataset']['path'],
        temporal_length=config['dataset']['temporal_length'],
        split='val',
        train_ratio=config['dataset']['train_ratio'],
        seed=config['dataset']['seed'],
        clips_per_sample=None  # Use all available clips for validation
    )

    batch_size = config['dataloader']['batch_size']
    seed = config['dataset']['seed']

    train_sampler = DimensionGroupedSampler(train_dataset, batch_size, shuffle=True, seed=seed)
    val_sampler = DimensionGroupedSampler(val_dataset, batch_size, shuffle=False, seed=seed)

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


def get_lr_scheduler(optimizer, config, total_steps: int):
    """Create learning rate scheduler with warmup and cosine decay."""
    from torch.optim.lr_scheduler import LambdaLR
    import math

    warmup_steps = config['training'].get('warmup_steps', 200)
    min_lr_ratio = config['training'].get('min_lr', 1e-6) / config['training']['learning_rate']

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def validate(model, val_loader, accelerator):
    """
    Validate and return nRMSE and RMSE (nRMSE first for consistency).

    Args:
        model: The model to validate
        val_loader: Validation dataloader
        accelerator: Accelerator instance

    Returns:
        avg_nrmse: Average nRMSE across all batches
        avg_rmse: Average RMSE across all batches
    """
    model.eval()
    total_nrmse = torch.zeros(1, device=accelerator.device)
    total_rmse = torch.zeros(1, device=accelerator.device)
    num_batches = torch.zeros(1, device=accelerator.device)

    for batch in val_loader:
        data = batch['data'].to(device=accelerator.device, dtype=torch.float32)
        channel_mask = batch['channel_mask'].to(device=accelerator.device)

        input_data = data[:, :-1]
        target_data = data[:, 1:]

        output = model(input_data)

        # Compute nRMSE loss (per-sample, per-channel)
        nrmse, rmse = compute_nrmse_loss(output.float(), target_data.float(), channel_mask)

        total_nrmse += nrmse.detach()
        total_rmse += rmse.detach()
        num_batches += 1

    accelerator.wait_for_everyone()
    total_nrmse = accelerator.reduce(total_nrmse, reduction='sum')
    total_rmse = accelerator.reduce(total_rmse, reduction='sum')
    num_batches = accelerator.reduce(num_batches, reduction='sum')

    model.train()

    avg_nrmse = (total_nrmse / num_batches).item() if num_batches.item() > 0 else 0
    avg_rmse = (total_rmse / num_batches).item() if num_batches.item() > 0 else 0
    return avg_nrmse, avg_rmse


def save_checkpoint(model, optimizer, scheduler, global_step, val_rmse, val_nrmse, best_val_nrmse, config, save_dir, accelerator, filename):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        checkpoint = {
            'global_step': global_step,
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'val_rmse': val_rmse,
            'val_nrmse': val_nrmse,
            'best_val_nrmse': best_val_nrmse,
            'config': config,
            'model_version': 'v2',  # Mark as V2 checkpoint
        }
        torch.save(checkpoint, save_dir / filename)
    accelerator.wait_for_everyone()


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config['dataset']['seed'])

    model_name = config.get('model_name', 'pde_e2e_v2')
    hidden_size = config['model']['transformer']['hidden_size']
    num_layers = config['model']['transformer']['num_hidden_layers']
    warmup_steps = config['training'].get('warmup_steps', 200)
    max_epochs = config['training']['max_epochs']
    clips_per_sample = config['dataset'].get('clips_per_sample', 80)
    eval_interval = config['training'].get('eval_interval', 100)

    # DDP kwargs
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        mixed_precision=config['training'].get('mixed_precision', 'no'),  # Default to fp32
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1),
        log_with="wandb",
        kwargs_handlers=[ddp_kwargs]
    )

    log_interval = config['logging']['log_interval']
    early_stopping_patience = config['training'].get('early_stopping_patience', 20)
    grad_clip = config['training'].get('grad_clip', 1.0)

    # Create dataloaders
    train_loader, val_loader, train_sampler, val_sampler = create_dataloaders(config)

    # Calculate total steps for scheduler
    steps_per_epoch = len(train_loader)
    total_steps = max_epochs * steps_per_epoch

    if accelerator.is_main_process:
        logger.info(f"{'='*60}")
        logger.info(f"End-to-End Training V2 (Temporal Residual)")
        logger.info(f"{'='*60}")
        logger.info(f"Config: {args.config}")
        logger.info(f"Model: {model_name} (hidden={hidden_size}, layers={num_layers})")
        logger.info(f"Key Feature: output = input + delta (residual learning)")
        logger.info(f"Max Epochs: {max_epochs}")
        logger.info(f"Clips per Sample: {clips_per_sample}")
        logger.info(f"Steps per Epoch: {steps_per_epoch}")
        logger.info(f"Total Steps: {total_steps}")
        logger.info(f"Warmup Steps: {warmup_steps}")
        logger.info(f"Eval Interval: {eval_interval} steps")
        logger.info(f"Loss: nRMSE (per-sample, per-channel)")
        logger.info(f"{'='*60}")

    # Use V2 model with temporal residual
    model = PDECausalModelV2(config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        betas=tuple(config['training'].get('betas', [0.9, 0.999]))
    )

    scheduler = get_lr_scheduler(optimizer, config, total_steps)
    model, optimizer = accelerator.prepare(model, optimizer)

    # Init WandB
    if accelerator.is_main_process:
        run_name = f"e2e-v2-{model_name}-h{hidden_size}-L{num_layers}"
        accelerator.init_trackers(
            project_name=config['logging']['project'],
            config=config,
            init_kwargs={"wandb": {
                "entity": config['logging'].get('entity'),
                "name": run_name,
                "tags": ["e2e-v2", "temporal-residual", f"h{hidden_size}", f"L{num_layers}"],
            }}
        )

    save_dir = Path(config['logging']['save_dir'])
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_val_nrmse = float('inf')
    patience_counter = 0
    console = Console()
    early_stop = False

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

            # Set epoch for sampler (regenerates clips with new random starts)
            train_sampler.set_epoch(epoch)

            for batch in train_loader:
                data = batch['data'].to(device=accelerator.device, dtype=torch.float32)
                channel_mask = batch['channel_mask'].to(device=accelerator.device)

                input_data = data[:, :-1]
                target_data = data[:, 1:]

                # Check if in warmup phase
                in_warmup = global_step < warmup_steps

                with accelerator.accumulate(model):
                    output = model(input_data)

                    # Compute nRMSE loss
                    nrmse_loss, rmse_loss = compute_nrmse_loss(
                        output.float(), target_data.float(), channel_mask
                    )

                    train_loss = nrmse_loss
                    accelerator.backward(train_loss)

                    if grad_clip:
                        accelerator.clip_grad_norm_(model.parameters(), grad_clip)

                    optimizer.step()
                    optimizer.zero_grad()

                scheduler.step()
                global_step += 1

                # Reduce loss across GPUs for logging
                nrmse_reduced = accelerator.reduce(nrmse_loss.detach(), reduction='mean')
                rmse_reduced = accelerator.reduce(rmse_loss.detach(), reduction='mean')

                # Update progress bar
                phase_str = "[warmup]" if in_warmup else f"[E{epoch+1}]"
                progress.update(train_task, advance=1,
                              description=f"{phase_str} nRMSE={nrmse_reduced.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}")

                # Log to wandb
                if global_step % log_interval == 0:
                    accelerator.log({
                        'train/nrmse': nrmse_reduced.item(),
                        'train/rmse': rmse_reduced.item(),
                        'train/lr': scheduler.get_last_lr()[0],
                        'train/epoch': epoch + 1,
                    }, step=global_step)

                # Validate every eval_interval steps
                if global_step % eval_interval == 0:
                    accelerator.wait_for_everyone()
                    val_nrmse, val_rmse = validate(model, val_loader, accelerator)

                    # Log validation metrics
                    accelerator.log({
                        'val/nrmse': val_nrmse,
                        'val/rmse': val_rmse,
                    }, step=global_step)

                    if accelerator.is_main_process:
                        console.print(f"\n[green]Step {global_step}/{total_steps} {phase_str}:[/green] val_nrmse={val_nrmse:.6f}, val_rmse={val_rmse:.6f}")

                    # Post-warmup: enable best model saving and early stopping
                    if not in_warmup:
                        if val_nrmse < best_val_nrmse:
                            best_val_nrmse = val_nrmse
                            patience_counter = 0
                            save_checkpoint(model, optimizer, scheduler, global_step, val_rmse, val_nrmse, best_val_nrmse, config, save_dir, accelerator, 'best.pt')
                            if accelerator.is_main_process:
                                console.print(f"[yellow]Saved best model[/yellow] (val_nrmse: {val_nrmse:.6f})")
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

    accelerator.end_training()

    if accelerator.is_main_process:
        table = Table(title="Training Complete (V2 - Temporal Residual)", show_header=False, border_style="green")
        table.add_row("Total Epochs", str(epoch + 1))
        table.add_row("Total Steps", str(global_step))
        table.add_row("Best Val nRMSE", f"{best_val_nrmse:.6f}")
        table.add_row("Checkpoint", str(save_dir / "best.pt"))
        console.print(table)


if __name__ == "__main__":
    main()
