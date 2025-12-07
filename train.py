"""
Training script for PDE Causal Autoregressive Model.
Uses Accelerate for distributed training and WandB for logging.

Usage:
    # First time setup (choose options interactively)
    accelerate config

    # Launch training
    accelerate launch train.py

    # Or single GPU
    python train.py
"""

import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import set_seed
from torch.distributed.fsdp import ShardingStrategy
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.panel import Panel
from rich.table import Table
import logging

from dataset import PDEDataset, DimensionGroupedSampler, collate_fn
from pipeline import PDECausalModel, compute_masked_loss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_dataloaders(config: dict):
    """Create train and validation dataloaders."""
    train_dataset = PDEDataset(
        data_dir=config['dataset']['path'],
        temporal_length=config['dataset']['temporal_length'],
        split='train',
        train_ratio=config['dataset']['train_ratio'],
        seed=config['dataset']['seed']
    )

    val_dataset = PDEDataset(
        data_dir=config['dataset']['path'],
        temporal_length=config['dataset']['temporal_length'],
        split='val',
        train_ratio=config['dataset']['train_ratio'],
        seed=config['dataset']['seed']
    )

    batch_size = config['dataloader']['batch_size']

    train_sampler = DimensionGroupedSampler(train_dataset, batch_size, shuffle=True)
    val_sampler = DimensionGroupedSampler(val_dataset, batch_size, shuffle=False)

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

    return train_loader, val_loader


def infinite_dataloader(dataloader):
    """Yields batches infinitely, reshuffling when exhausted."""
    while True:
        for batch in dataloader:
            yield batch


def get_lr_scheduler(optimizer, config):
    """Create learning rate scheduler with warmup."""
    from torch.optim.lr_scheduler import LambdaLR

    warmup_steps = config['training'].get('warmup_steps', 1000)
    max_steps = config['training']['max_steps']
    min_lr_ratio = config['training'].get('min_lr', 1e-6) / config['training']['learning_rate']

    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return step / warmup_steps
        else:
            # Cosine decay
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    return LambdaLR(optimizer, lr_lambda)


def cleanup_checkpoints(save_dir: Path, keep_n: int = 3):
    """Keep only the last N checkpoints (excluding best.pt and latest.pt)."""
    checkpoints = sorted(save_dir.glob("checkpoint-step-*.pt"), key=lambda x: int(x.stem.split("-")[-1]))
    for ckpt in checkpoints[:-keep_n]:
        ckpt.unlink()


@torch.no_grad()
def validate(model, val_loader, accelerator):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in val_loader:
        # Convert data to bf16
        data = batch['data'].to(torch.bfloat16)
        channel_mask = batch['channel_mask']

        input_data = data[:, :-1]
        target_data = data[:, 1:]

        output = model(input_data)
        loss = compute_masked_loss(output, target_data, channel_mask)

        # Gather loss across all processes
        gathered_loss = accelerator.gather(loss.unsqueeze(0)).mean()
        total_loss += gathered_loss.item()
        num_batches += 1

    # Sync before returning
    accelerator.wait_for_everyone()
    model.train()
    return total_loss / num_batches if num_batches > 0 else 0.0


def save_checkpoint(model, optimizer, scheduler, global_step, val_loss, config, save_dir, accelerator, filename):
    """Save checkpoint."""
    # Sync all processes before saving
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        checkpoint = {
            'global_step': global_step,
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'val_loss': val_loss,
            'config': config
        }
        torch.save(checkpoint, save_dir / filename)
    accelerator.wait_for_everyone()


def main():
    # Load config
    config = load_config()
    set_seed(config['dataset']['seed'])

    # Initialize Accelerator
    mixed_precision = config['training'].get('mixed_precision', 'bf16')
    grad_accum = config['training'].get('gradient_accumulation_steps', 1)
    use_fsdp = config['training'].get('use_fsdp', False)

    # FSDP Plugin for large model training
    fsdp_plugin = None
    if use_fsdp:
        fsdp_plugin = FullyShardedDataParallelPlugin(
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            reshard_after_forward=True,
        )

    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=grad_accum,
        fsdp_plugin=fsdp_plugin,
        log_with="wandb"
    )

    # Training config
    max_steps = config['training']['max_steps']
    save_every_steps = config['training']['save_every_steps']
    eval_every_steps = config['training']['eval_every_steps']
    keep_last_n = config['training']['keep_last_n_checkpoints']
    log_interval = config['logging']['log_interval']

    if accelerator.is_main_process:
        logger.info(f"{'='*60}")
        logger.info(f"Training Configuration")
        logger.info(f"{'='*60}")
        logger.info(f"Max Steps: {max_steps}")
        logger.info(f"Save Every: {save_every_steps} steps")
        logger.info(f"Eval Every: {eval_every_steps} steps")
        logger.info(f"Mixed Precision: {mixed_precision}")
        logger.info(f"Gradient Accumulation: {grad_accum}")
        logger.info(f"FSDP: {'Enabled' if use_fsdp else 'Disabled'}")
        if use_fsdp:
            logger.info(f"FSDP CPU Offload: {config['training'].get('fsdp_cpu_offload', False)}")
        logger.info(f"Num Processes: {accelerator.num_processes}")
        logger.info(f"{'='*60}")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)

    if accelerator.is_main_process:
        logger.info(f"Train batches per epoch: {len(train_loader)}")
        logger.info(f"Val batches: {len(val_loader)}")

    # Create model
    model = PDECausalModel(config)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        betas=tuple(config['training'].get('betas', [0.9, 0.999]))
    )

    # Scheduler with warmup
    scheduler = get_lr_scheduler(optimizer, config)

    # Prepare with Accelerator
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # Init WandB tracker
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=config['logging']['project'],
            config=config,
            init_kwargs={"wandb": {
                "entity": config['logging'].get('entity'),
                "name": f"pde-{config['dataset']['seed']}"
            }}
        )

    # Checkpoint directory
    save_dir = Path(config['logging']['save_dir'])
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Training state
    global_step = 0
    best_val_loss = float('inf')
    train_iter = infinite_dataloader(train_loader)
    console = Console()

    # Training loop with Rich progress
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
        train_task = progress.add_task("Training", total=max_steps)

        while global_step < max_steps:
            batch = next(train_iter)
            # Convert data to bf16
            data = batch['data'].to(torch.bfloat16)
            channel_mask = batch['channel_mask']

            # Causal AR split
            input_data = data[:, :-1]
            target_data = data[:, 1:]

            # Forward & Backward
            with accelerator.accumulate(model):
                output = model(input_data)
                loss = compute_masked_loss(output, target_data, channel_mask)

                accelerator.backward(loss)

                if config['training'].get('grad_clip'):
                    accelerator.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            progress.update(train_task, advance=1, description=f"Training [loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}]")

            # Log
            if global_step % log_interval == 0:
                accelerator.log({
                    'train/loss': loss.item(),
                    'train/global_step': global_step,
                    'train/lr': scheduler.get_last_lr()[0],
                    'gpu/memory_gb': torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
                }, step=global_step)

            # Evaluate
            if global_step % eval_every_steps == 0:
                val_loss = validate(model, val_loader, accelerator)

                accelerator.log({
                    'val/loss': val_loss,
                    'val/global_step': global_step,
                }, step=global_step)

                if accelerator.is_main_process:
                    console.print(f"[green]Step {global_step}:[/green] val_loss = {val_loss:.6f}")

                    # Save best
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(model, optimizer, scheduler, global_step, val_loss, config, save_dir, accelerator, 'best.pt')
                        console.print(f"[yellow]Saved best model[/yellow] (val_loss: {val_loss:.6f})")

            # Save checkpoint
            if global_step % save_every_steps == 0:
                save_checkpoint(model, optimizer, scheduler, global_step, best_val_loss, config, save_dir, accelerator, f'checkpoint-step-{global_step}.pt')
                save_checkpoint(model, optimizer, scheduler, global_step, best_val_loss, config, save_dir, accelerator, 'latest.pt')

                if accelerator.is_main_process:
                    cleanup_checkpoints(save_dir, keep_n=keep_last_n)
                    console.print(f"[cyan]Saved checkpoint[/cyan] at step {global_step}")

    accelerator.end_training()

    if accelerator.is_main_process:
        # Final summary with Rich
        table = Table(title="Training Complete", show_header=False, border_style="green")
        table.add_row("Total Steps", str(global_step))
        table.add_row("Best Val Loss", f"{best_val_loss:.6f}")
        console.print(table)


if __name__ == "__main__":
    main()
