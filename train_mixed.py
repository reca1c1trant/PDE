"""
Training script for Mixed Dataset with Adaptive Temporal Length.

Features:
- Supports multiple data sources with ratio control
- Adaptive temporal_length (8 or 16 based on sample's T)
- No clip discarding, incomplete batches are padded
- Same-sample batching preserved

Usage:
    torchrun --nproc_per_node=8 train_mixed.py --config configs/mixed_adaptive.yaml
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

from dataset_mixed import MixedPDEDataset, AdaptiveSampler, mixed_collate_fn
from pipeline import PDECausalModel

logger = logging.getLogger(__name__)


def compute_nrmse_loss(pred: torch.Tensor, target: torch.Tensor, channel_mask: torch.Tensor):
    """
    Compute nRMSE loss (per-sample, per-channel).
    Works with variable temporal_length.
    """
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

    return nrmse, rmse


def parse_args():
    parser = argparse.ArgumentParser(description="Mixed Dataset Training")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_dataloaders(config: dict, num_replicas: int, rank: int):
    """Create train and val dataloaders."""
    sources = config['dataset']['sources']
    temporal_threshold = config['dataset'].get('temporal_threshold', 24)
    train_ratio = config['dataset'].get('train_ratio', 0.9)
    seed = config['dataset'].get('seed', 42)
    batch_size = config['dataloader']['batch_size']

    # Training dataset
    train_dataset = MixedPDEDataset(
        data_sources=sources,
        temporal_threshold=temporal_threshold,
        split='train',
        train_ratio=train_ratio,
        seed=seed,
    )

    # Validation: use all samples (ratio=1.0 for all sources)
    val_sources = [{'path': s['path'], 'ratio': 1.0} for s in sources]
    val_dataset = MixedPDEDataset(
        data_sources=val_sources,
        temporal_threshold=temporal_threshold,
        split='val',
        train_ratio=train_ratio,
        seed=seed,
    )

    train_sampler = AdaptiveSampler(
        train_dataset, batch_size, shuffle=True, seed=seed,
        num_replicas=num_replicas, rank=rank
    )
    val_sampler = AdaptiveSampler(
        val_dataset, batch_size, shuffle=False, seed=seed,
        num_replicas=num_replicas, rank=rank
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=mixed_collate_fn,
        num_workers=config['dataloader']['num_workers'],
        pin_memory=config['dataloader']['pin_memory']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=mixed_collate_fn,
        num_workers=config['dataloader']['num_workers'],
        pin_memory=config['dataloader']['pin_memory']
    )

    return train_loader, val_loader, train_sampler, val_sampler


def get_lr_scheduler(optimizer, config, total_steps: int):
    """Create LR scheduler with warmup and cosine decay."""
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
    """Validate and return nRMSE and RMSE."""
    model.eval()
    total_nrmse = torch.zeros(1, device=accelerator.device)
    total_rmse = torch.zeros(1, device=accelerator.device)
    num_batches = torch.zeros(1, device=accelerator.device)

    for batch in val_loader:
        data = batch['data'].to(device=accelerator.device, dtype=torch.bfloat16)
        channel_mask = batch['channel_mask'].to(device=accelerator.device)
        # temporal_length is same for all items in batch

        # Causal AR: input = data[:, :-1], target = data[:, 1:]
        input_data = data[:, :-1]
        target_data = data[:, 1:]

        output = model(input_data)

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
        }
        torch.save(checkpoint, save_dir / filename)
    accelerator.wait_for_everyone()


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config['dataset']['seed'])

    model_name = config.get('model_name', 'pde_mixed')
    hidden_size = config['model']['transformer']['hidden_size']
    num_layers = config['model']['transformer']['num_hidden_layers']
    warmup_steps = config['training'].get('warmup_steps', 200)
    max_epochs = config['training']['max_epochs']
    eval_interval = config['training'].get('eval_interval', 100)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        mixed_precision=config['training'].get('mixed_precision', 'bf16'),
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1),
        log_with="wandb",
        kwargs_handlers=[ddp_kwargs]
    )

    log_interval = config['logging']['log_interval']
    early_stopping_patience = config['training'].get('early_stopping_patience', 20)
    grad_clip = config['training'].get('grad_clip', 1.0)

    # Create dataloaders
    train_loader, val_loader, train_sampler, val_sampler = create_dataloaders(
        config, accelerator.num_processes, accelerator.process_index
    )

    steps_per_epoch = len(train_loader)
    total_steps = max_epochs * steps_per_epoch

    if accelerator.is_main_process:
        logger.info(f"{'='*60}")
        logger.info(f"Mixed Dataset Training (Adaptive Temporal)")
        logger.info(f"{'='*60}")
        logger.info(f"Config: {args.config}")
        logger.info(f"Model: {model_name} (hidden={hidden_size}, layers={num_layers})")
        logger.info(f"Max Epochs: {max_epochs}")
        logger.info(f"Steps per Epoch: {steps_per_epoch}")
        logger.info(f"Total Steps: {total_steps}")
        logger.info(f"Warmup Steps: {warmup_steps}")
        logger.info(f"Eval Interval: {eval_interval} steps")
        logger.info(f"{'='*60}")

    model = PDECausalModel(config)

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
        run_name = f"mixed-{model_name}-h{hidden_size}-L{num_layers}"
        accelerator.init_trackers(
            project_name=config['logging']['project'],
            config=config,
            init_kwargs={"wandb": {
                "entity": config['logging'].get('entity'),
                "name": run_name,
                "tags": ["mixed", "adaptive", f"h{hidden_size}", f"L{num_layers}"],
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

            for batch in train_loader:
                data = batch['data'].to(device=accelerator.device, dtype=torch.bfloat16)
                channel_mask = batch['channel_mask'].to(device=accelerator.device)
                temporal_length = batch['temporal_length']  # 8 or 16

                # Causal AR split
                input_data = data[:, :-1]   # [B, T, H, W, C]
                target_data = data[:, 1:]   # [B, T, H, W, C]

                in_warmup = global_step < warmup_steps

                with accelerator.accumulate(model):
                    output = model(input_data)

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

                nrmse_reduced = accelerator.reduce(nrmse_loss.detach(), reduction='mean')
                rmse_reduced = accelerator.reduce(rmse_loss.detach(), reduction='mean')

                phase_str = "[warmup]" if in_warmup else f"[E{epoch+1}]"
                progress.update(train_task, advance=1,
                              description=f"{phase_str} T={temporal_length} nRMSE={nrmse_reduced.item():.4f}")

                if global_step % log_interval == 0:
                    accelerator.log({
                        'train/nrmse': nrmse_reduced.item(),
                        'train/rmse': rmse_reduced.item(),
                        'train/lr': scheduler.get_last_lr()[0],
                        'train/epoch': epoch + 1,
                        'train/temporal_length': temporal_length,
                    }, step=global_step)

                if global_step % eval_interval == 0:
                    accelerator.wait_for_everyone()
                    val_nrmse, val_rmse = validate(model, val_loader, accelerator)

                    accelerator.log({
                        'val/nrmse': val_nrmse,
                        'val/rmse': val_rmse,
                    }, step=global_step)

                    if accelerator.is_main_process:
                        console.print(f"\n[green]Step {global_step}/{total_steps} {phase_str}:[/green] val_nrmse={val_nrmse:.6f}")

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
        table = Table(title="Training Complete", show_header=False, border_style="green")
        table.add_row("Total Epochs", str(epoch + 1))
        table.add_row("Total Steps", str(global_step))
        table.add_row("Best Val nRMSE", f"{best_val_nrmse:.6f}")
        table.add_row("Checkpoint", str(save_dir / "best.pt"))
        console.print(table)


if __name__ == "__main__":
    main()
