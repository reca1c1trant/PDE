"""
Training script for PDE Causal Autoregressive Model.
Uses Accelerate for distributed training and WandB for logging.

Usage:
    # Launch with specific config
    accelerate launch train.py --config configs/llama_1b.yaml
    accelerate launch train.py --config configs/llama_3b.yaml
    accelerate launch train.py --config configs/llama_8b.yaml

    # Or use torchrun for DDP (smaller models)
    torchrun --nproc_per_node=8 train.py --config configs/llama_1b.yaml
"""

import os
import sys
import warnings

# ============================================================
# 主进程通信控制（必须在所有其他 import 之前）
# ============================================================
def _is_main_process():
    """Check if current process is main (rank 0)."""
    return os.environ.get('LOCAL_RANK', '0') == '0'

IS_MAIN_PROCESS = _is_main_process()

if not IS_MAIN_PROCESS:
    # 非主进程：禁用所有 warnings
    warnings.filterwarnings('ignore')
else:
    # 主进程：过滤特定 warnings
    warnings.filterwarnings('ignore', message='.*FSDP upcast.*')

# 设置 Triton cache 到用户目录
triton_cache = '/tmp/triton_cache'
os.makedirs(triton_cache, exist_ok=True)
os.environ.setdefault('TRITON_CACHE_DIR', triton_cache)

# ============================================================
# 正常 imports
# ============================================================
import argparse
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

# TF32 精度优化
torch.set_float32_matmul_precision('high')

# Logging 配置（只在主进程生效）
if IS_MAIN_PROCESS:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
else:
    logging.disable(logging.CRITICAL)

from dataset import PDEDataset, DimensionGroupedSampler, collate_fn
from pipeline import PDECausalModel, compute_masked_loss

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="PDE Causal AR Training")
    parser.add_argument(
        '--config', type=str, default='config.yaml',
        help='Path to config file (e.g., configs/llama_1b.yaml)'
    )
    return parser.parse_args()


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_dataloaders(config: dict):
    """Create train and validation dataloaders with distributed-aware samplers."""
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
    seed = config['dataset']['seed']

    # Samplers are now distributed-aware (auto-detect WORLD_SIZE and RANK)
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
    total_loss = torch.zeros(1, device=accelerator.device)
    num_batches = torch.zeros(1, device=accelerator.device)

    for batch in val_loader:
        data = batch['data'].to(device=accelerator.device, dtype=torch.bfloat16)
        channel_mask = batch['channel_mask'].to(device=accelerator.device)

        input_data = data[:, :-1]
        target_data = data[:, 1:]

        output = model(input_data)
        loss = compute_masked_loss(output, target_data, channel_mask)

        total_loss += loss.detach()
        num_batches += 1

    # 同步所有进程
    accelerator.wait_for_everyone()

    # 一次性 reduce
    total_loss = accelerator.reduce(total_loss, reduction='sum')
    num_batches = accelerator.reduce(num_batches, reduction='sum')

    model.train()
    avg_loss = total_loss / num_batches if num_batches.item() > 0 else total_loss
    return avg_loss.item()


def save_checkpoint(model, optimizer, scheduler, global_step, val_loss, best_val_loss, config, save_dir, accelerator, filename):
    """Save checkpoint with all training state."""
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
            'best_val_loss': best_val_loss,
            'config': config
        }
        torch.save(checkpoint, save_dir / filename)
    accelerator.wait_for_everyone()


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, accelerator):
    """
    Load checkpoint and restore training state.

    Returns:
        global_step: Step to resume from
        best_val_loss: Best validation loss so far
    """
    accelerator.wait_for_everyone()

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Load model state
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.load_state_dict(checkpoint['model_state_dict'])

    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scheduler state
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    global_step = checkpoint.get('global_step', 0)
    best_val_loss = checkpoint.get('best_val_loss', checkpoint.get('val_loss', float('inf')))

    accelerator.wait_for_everyone()

    return global_step, best_val_loss


def main():
    # Parse arguments and load config
    args = parse_args()
    config = load_config(args.config)
    set_seed(config['dataset']['seed'])

    # Model info for logging
    model_name = config.get('model_name', 'llama')
    hidden_size = config['model']['transformer']['hidden_size']
    num_layers = config['model']['transformer']['num_hidden_layers']
    seed = config['dataset']['seed']

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
        logger.info(f"Config: {args.config}")
        logger.info(f"Model: {model_name} (hidden={hidden_size}, layers={num_layers})")
        logger.info(f"Max Steps: {max_steps}")
        logger.info(f"Save Every: {save_every_steps} steps")
        logger.info(f"Eval Every: {eval_every_steps} steps")
        logger.info(f"Mixed Precision: {mixed_precision}")
        logger.info(f"Gradient Accumulation: {grad_accum}")
        logger.info(f"Distributed: {'FSDP' if use_fsdp else 'DDP'}")
        if use_fsdp:
            logger.info(f"FSDP CPU Offload: {config['training'].get('fsdp_cpu_offload', False)}")
        logger.info(f"Num Processes: {accelerator.num_processes}")
        logger.info(f"{'='*60}")

    # Create dataloaders (samplers are already distributed-aware)
    train_loader, val_loader, train_sampler, val_sampler = create_dataloaders(config)
    
    if accelerator.is_main_process:
        logger.info(f"Train batches per rank: {len(train_loader)}")
        logger.info(f"Val batches per rank: {len(val_loader)}")

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

    # Prepare with Accelerator (don't prepare dataloaders - sampler is already distributed)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    # Init WandB tracker
    if accelerator.is_main_process:
        # WandB run name and notes
        run_name = f"{model_name}-h{hidden_size}-L{num_layers}-s{seed}"
        run_notes = (
            f"Model: {model_name}\n"
            f"Transformer: hidden_size={hidden_size}, layers={num_layers}\n"
            f"Distributed: {'FSDP' if use_fsdp else 'DDP'}\n"
            f"Config: {args.config}"
        )

        accelerator.init_trackers(
            project_name=config['logging']['project'],
            config=config,
            init_kwargs={"wandb": {
                "entity": config['logging'].get('entity'),
                "name": run_name,
                "notes": run_notes,
                "tags": [model_name, f"h{hidden_size}", f"L{num_layers}", "FSDP" if use_fsdp else "DDP"],
            }}
        )

    # Checkpoint directory
    save_dir = Path(config['logging']['save_dir'])
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Training state
    global_step = 0
    best_val_loss = float('inf')
    last_val_loss = float('inf')  # Track last validation loss for checkpointing

    # Resume from checkpoint if specified
    resume_path = config['training'].get('resume_from')
    if resume_path:
        resume_path = Path(resume_path)
        if resume_path.exists():
            if accelerator.is_main_process:
                logger.info(f"Resuming from checkpoint: {resume_path}")
            global_step, best_val_loss = load_checkpoint(
                resume_path, model, optimizer, scheduler, accelerator
            )
            if accelerator.is_main_process:
                logger.info(f"Resumed at step {global_step}, best_val_loss={best_val_loss:.6f}")
                logger.info(f"Current LR: {scheduler.get_last_lr()[0]:.2e}")
        else:
            if accelerator.is_main_process:
                logger.warning(f"Checkpoint not found: {resume_path}, starting from scratch")

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
        train_task = progress.add_task("Training", total=max_steps, completed=global_step)

        while global_step < max_steps:
            batch = next(train_iter)
            # Move data to device and convert to bf16
            data = batch['data'].to(device=accelerator.device, dtype=torch.bfloat16)
            channel_mask = batch['channel_mask'].to(device=accelerator.device)

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
                accelerator.wait_for_everyone()  # 确保所有卡同步进入 validation
                val_loss = validate(model, val_loader, accelerator)
                last_val_loss = val_loss  # Update for checkpoint saving

                accelerator.log({
                    'val/loss': val_loss,
                    'val/global_step': global_step,
                }, step=global_step)

                if accelerator.is_main_process:
                    console.print(f"[green]Step {global_step}:[/green] val_loss = {val_loss:.6f}")

                # Save best (all ranks must call save_checkpoint for sync)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, optimizer, scheduler, global_step, val_loss, best_val_loss, config, save_dir, accelerator, 'best.pt')
                    if accelerator.is_main_process:
                        console.print(f"[yellow]Saved best model[/yellow] (val_loss: {val_loss:.6f})")

            # Save checkpoint periodically
            if global_step % save_every_steps == 0:
                save_checkpoint(model, optimizer, scheduler, global_step, last_val_loss, best_val_loss, config, save_dir, accelerator, f'checkpoint-step-{global_step}.pt')
                save_checkpoint(model, optimizer, scheduler, global_step, last_val_loss, best_val_loss, config, save_dir, accelerator, 'latest.pt')

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
