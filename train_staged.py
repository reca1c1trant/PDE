"""
Staged Training for PDE Causal Autoregressive Model.

Supports freezing specific modules (encoder, transformer, decoder) for staged training.

Usage:
    # Stage 1: Train Encoder/Decoder only
    OMP_NUM_THREADS=6 torchrun --nproc_per_node=8 train_staged.py --config train/stage1.yaml

    # Stage 2: Train Transformer only
    OMP_NUM_THREADS=6 torchrun --nproc_per_node=8 train_staged.py --config train/stage2.yaml

    # Stage 3: Fine-tune all
    OMP_NUM_THREADS=6 torchrun --nproc_per_node=8 train_staged.py --config train/stage3.yaml
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
    warnings.filterwarnings('ignore')
else:
    warnings.filterwarnings('ignore', message='.*FSDP upcast.*')

# 设置 Triton cache
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
from accelerate import Accelerator
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
from pipeline import PDECausalModel, compute_masked_loss

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Staged PDE Training")
    parser.add_argument('--config', type=str, required=True, help='Path to stage config')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
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
    seed = config['dataset']['seed']
    same_sample = config['dataloader'].get('same_sample_per_batch', False)

    train_sampler = DimensionGroupedSampler(train_dataset, batch_size, shuffle=True, seed=seed, same_sample_per_batch=same_sample)
    val_sampler = DimensionGroupedSampler(val_dataset, batch_size, shuffle=False, seed=seed, same_sample_per_batch=False)

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
    while True:
        for batch in dataloader:
            yield batch


def get_lr_scheduler(optimizer, config):
    from torch.optim.lr_scheduler import LambdaLR

    warmup_steps = config['training'].get('warmup_steps', 100)
    max_steps = config['training']['max_steps']
    min_lr_ratio = config['training'].get('min_lr', 1e-6) / config['training']['learning_rate']

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    return LambdaLR(optimizer, lr_lambda)


def freeze_modules(model, freeze_config, logger):
    """Freeze specified modules based on config."""
    frozen_params = 0
    trainable_params = 0

    # Freeze transformer
    if freeze_config.get('transformer', False):
        for param in model.transformer.parameters():
            param.requires_grad = False
        frozen_params += sum(p.numel() for p in model.transformer.parameters())
        if IS_MAIN_PROCESS:
            logger.info("Transformer: FROZEN")
    else:
        trainable_params += sum(p.numel() for p in model.transformer.parameters())
        if IS_MAIN_PROCESS:
            logger.info("Transformer: TRAINABLE")

    # Freeze encoder
    if freeze_config.get('encoder', False):
        for param in model.encoder_2d.parameters():
            param.requires_grad = False
        frozen_params += sum(p.numel() for p in model.encoder_2d.parameters())
        if IS_MAIN_PROCESS:
            logger.info("Encoder: FROZEN")
    else:
        trainable_params += sum(p.numel() for p in model.encoder_2d.parameters())
        if IS_MAIN_PROCESS:
            logger.info("Encoder: TRAINABLE")

    # Freeze decoder
    if freeze_config.get('decoder', False):
        for param in model.decoder_2d.parameters():
            param.requires_grad = False
        frozen_params += sum(p.numel() for p in model.decoder_2d.parameters())
        if IS_MAIN_PROCESS:
            logger.info("Decoder: FROZEN")
    else:
        trainable_params += sum(p.numel() for p in model.decoder_2d.parameters())
        if IS_MAIN_PROCESS:
            logger.info("Decoder: TRAINABLE")

    if IS_MAIN_PROCESS:
        logger.info(f"Trainable params: {trainable_params:,}")
        logger.info(f"Frozen params: {frozen_params:,}")

    return trainable_params, frozen_params


def load_pretrained_weights(model, checkpoint_path, accelerator):
    """Load model weights only (no optimizer/scheduler)."""
    accelerator.wait_for_everyone()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.load_state_dict(checkpoint['model_state_dict'])
    accelerator.wait_for_everyone()
    return checkpoint.get('global_step', 0), checkpoint.get('best_val_loss', float('inf'))


@torch.no_grad()
def validate(model, val_loader, accelerator, loss_alpha=0.0, sigma=None):
    model.eval()
    total_loss = torch.zeros(1, device=accelerator.device)
    num_batches = torch.zeros(1, device=accelerator.device)

    for batch in val_loader:
        data = batch['data'].to(device=accelerator.device, dtype=torch.bfloat16)
        channel_mask = batch['channel_mask'].to(device=accelerator.device)

        input_data = data[:, :-1]
        target_data = data[:, 1:]

        output = model(input_data)
        loss = compute_masked_loss(output, target_data, channel_mask, alpha=loss_alpha, sigma=sigma)

        total_loss += loss.detach()
        num_batches += 1

    accelerator.wait_for_everyone()
    total_loss = accelerator.reduce(total_loss, reduction='sum')
    num_batches = accelerator.reduce(num_batches, reduction='sum')

    model.train()
    avg_loss = total_loss / num_batches if num_batches.item() > 0 else total_loss
    return avg_loss.item()


def save_checkpoint(model, optimizer, scheduler, global_step, val_loss, best_val_loss, config, save_dir, accelerator, filename):
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


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config['dataset']['seed'])

    stage = config.get('stage', 0)

    # Initialize Accelerator (DDP only for staged training)
    # DDP kwargs: static_graph=True required for gradient_checkpointing + DDP
    # Note: embed_tokens is set to None in pipeline.py, so no unused parameters
    from accelerate import DistributedDataParallelKwargs
    ddp_kwargs = DistributedDataParallelKwargs(
        static_graph=True
    )

    accelerator = Accelerator(
        mixed_precision=config['training'].get('mixed_precision', 'bf16'),
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1),
        log_with="wandb",
        kwargs_handlers=[ddp_kwargs]
    )

    # Training config
    max_steps = config['training']['max_steps']
    eval_every_steps = config['training']['eval_every_steps']
    save_every_steps = config['training'].get('save_every_steps')  # Optional
    log_interval = config['logging']['log_interval']
    loss_alpha = config['training'].get('loss_alpha', 0.0)
    early_stopping_patience = config['training'].get('early_stopping_patience')

    # nRMSE sigma
    nrmse_sigma_config = config['training'].get('nrmse_sigma')
    nrmse_sigma = torch.tensor(nrmse_sigma_config, dtype=torch.float32) if nrmse_sigma_config else None

    if accelerator.is_main_process:
        logger.info(f"{'='*60}")
        logger.info(f"Staged Training - Stage {stage}")
        logger.info(f"{'='*60}")
        logger.info(f"Config: {args.config}")
        logger.info(f"Max Steps: {max_steps}")
        logger.info(f"Loss Alpha: {loss_alpha}")
        logger.info(f"nRMSE Sigma: {'Enabled' if nrmse_sigma is not None else 'Disabled'}")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)

    # Create model
    model = PDECausalModel(config)

    # Freeze modules based on config
    freeze_config = config.get('freeze', {})
    freeze_modules(model, freeze_config, logger)

    # Optimizer (only trainable params)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        betas=tuple(config['training'].get('betas', [0.9, 0.999]))
    )

    scheduler = get_lr_scheduler(optimizer, config)

    # Prepare with Accelerator
    model, optimizer = accelerator.prepare(model, optimizer)

    # Init WandB
    if accelerator.is_main_process:
        run_name = f"stage{stage}-{config.get('model_name', 'pde')}"
        accelerator.init_trackers(
            project_name=config['logging']['project'],
            config=config,
            init_kwargs={"wandb": {
                "entity": config['logging'].get('entity'),
                "name": run_name,
                "tags": [f"stage{stage}"],
            }}
        )

    # Checkpoint directory
    save_dir = Path(config['logging']['save_dir'])
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Training state
    global_step = 0
    best_val_loss = float('inf')

    # Load pretrained weights if specified
    pretrain_path = config['training'].get('pretrain_from')
    if pretrain_path:
        pretrain_path = Path(pretrain_path)
        if pretrain_path.exists():
            if accelerator.is_main_process:
                logger.info(f"Loading pretrained weights from: {pretrain_path}")
            prev_step, prev_best = load_pretrained_weights(model, pretrain_path, accelerator)
            if accelerator.is_main_process:
                logger.info(f"Loaded weights (prev_step={prev_step}, prev_best_val={prev_best:.6f})")
        else:
            if accelerator.is_main_process:
                logger.warning(f"Pretrain checkpoint not found: {pretrain_path}")

    train_iter = infinite_dataloader(train_loader)
    console = Console()

    # Early stopping state
    patience_counter = 0

    # Training loop
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
        train_task = progress.add_task(f"Stage {stage}", total=max_steps)

        while global_step < max_steps:
            batch = next(train_iter)
            data = batch['data'].to(device=accelerator.device, dtype=torch.bfloat16)
            channel_mask = batch['channel_mask'].to(device=accelerator.device)

            input_data = data[:, :-1]
            target_data = data[:, 1:]

            sigma_device = nrmse_sigma.to(accelerator.device) if nrmse_sigma is not None else None

            with accelerator.accumulate(model):
                output = model(input_data)
                loss = compute_masked_loss(output, target_data, channel_mask, alpha=loss_alpha, sigma=sigma_device)

                accelerator.backward(loss)

                if config['training'].get('grad_clip'):
                    accelerator.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])

                optimizer.step()
                optimizer.zero_grad()

            scheduler.step()
            global_step += 1
            progress.update(train_task, advance=1, description=f"Stage {stage} [loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}]")

            # Log
            if global_step % log_interval == 0:
                accelerator.log({
                    'train/loss': loss.item(),
                    'train/global_step': global_step,
                    'train/lr': scheduler.get_last_lr()[0],
                }, step=global_step)

            # Evaluate
            if global_step % eval_every_steps == 0:
                accelerator.wait_for_everyone()
                val_loss = validate(model, val_loader, accelerator, loss_alpha=loss_alpha, sigma=sigma_device)

                accelerator.log({'val/loss': val_loss}, step=global_step)

                if accelerator.is_main_process:
                    console.print(f"[green]Step {global_step}:[/green] val_loss = {val_loss:.6f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    save_checkpoint(model, optimizer, scheduler, global_step, val_loss, best_val_loss, config, save_dir, accelerator, 'best.pt')
                    if accelerator.is_main_process:
                        console.print(f"[yellow]Saved best model[/yellow] (val_loss: {val_loss:.6f})")
                else:
                    patience_counter += 1
                    if accelerator.is_main_process and early_stopping_patience:
                        console.print(f"[dim]Patience: {patience_counter}/{early_stopping_patience}[/dim]")

                # Early stopping check
                if early_stopping_patience and patience_counter >= early_stopping_patience:
                    if accelerator.is_main_process:
                        console.print(f"[red]Early stopping triggered![/red] No improvement for {early_stopping_patience} evaluations.")
                    break

            # Save checkpoint periodically (optional)
            if save_every_steps and global_step % save_every_steps == 0:
                save_checkpoint(model, optimizer, scheduler, global_step, val_loss if 'val_loss' in dir() else float('inf'), best_val_loss, config, save_dir, accelerator, 'latest.pt')

    accelerator.end_training()

    if accelerator.is_main_process:
        table = Table(title=f"Stage {stage} Complete", show_header=False, border_style="green")
        table.add_row("Total Steps", str(global_step))
        table.add_row("Best Val Loss", f"{best_val_loss:.6f}")
        table.add_row("Checkpoint", str(save_dir / "best.pt"))
        console.print(table)


if __name__ == "__main__":
    main()
