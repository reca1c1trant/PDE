"""
Pretraining for PDE Foundation Model.

Features:
- Multi-dataset training (diffusion-reaction, 2D_CFD, SWE, NS_incom)
- 18 channels (3 vector + 15 scalar) with channel_mask
- RMSE loss in normalized space with channel masking
- Support for checkpoint resume (converted or scratch)
- PretrainSampler ensures same-batch same-equivalent-sample

Usage:
    # Single GPU (debug)
    python train_pretrain.py --config configs/pretrain.yaml

    # 8 GPUs distributed
    CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node=7 train_pretrain.py --config configs/pretrain.yaml

    # Resume from checkpoint
    python train_pretrain.py --config configs/pretrain.yaml --resume checkpoints_pretrain/latest.pt
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

from dataset_pretrain import (
    PretrainDataset, PretrainSampler, DatasetConfig,
    pretrain_collate_fn, create_pretrain_dataloaders,
    NUM_VECTOR_CHANNELS, NUM_SCALAR_CHANNELS, TOTAL_CHANNELS
)
from pipeline import PDECausalModel

logger = logging.getLogger(__name__)


def compute_normalized_rmse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    channel_mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute RMSE loss in normalized space with channel masking.

    Args:
        pred: [B, T, H, W, C] predictions (already in normalized space)
        target: [B, T, H, W, C] ground truth (already in normalized space)
        channel_mask: [B, C] valid channel mask (1=valid, 0=pad)

    Returns:
        rmse: scalar RMSE loss for backprop
    """
    # Expand channel_mask for broadcasting: [B, 1, 1, 1, C]
    mask = channel_mask[:, None, None, None, :].float()

    # Masked MSE: only compute on valid channels
    squared_error = (pred - target) ** 2  # [B, T, H, W, C]
    masked_error = squared_error * mask

    # Sum over all dimensions, divide by number of valid elements
    total_error = masked_error.sum()
    num_valid = mask.sum() * pred.shape[1] * pred.shape[2] * pred.shape[3]  # mask.sum() * T * H * W

    mse = total_error / (num_valid + 1e-8)
    rmse = torch.sqrt(mse + 1e-8)
    return rmse


def parse_args():
    parser = argparse.ArgumentParser(description="PDE Foundation Model Pretraining")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_lr_scheduler(optimizer, config: dict, total_steps: int):
    """Create learning rate scheduler with warmup and cosine decay."""
    from torch.optim.lr_scheduler import LambdaLR
    import math

    warmup_steps = config['training'].get('warmup_steps', 500)
    min_lr_ratio = config['training'].get('min_lr', 1e-6) / config['training']['learning_rate']

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def load_checkpoint(model, checkpoint_path: str, config: dict, accelerator):
    """
    Load checkpoint (model weights only, optimizer/scheduler start fresh).

    Returns:
        global_step: Always 0 (fresh start)
        best_val_rmse: Always inf (fresh start)
    """
    if not checkpoint_path or not Path(checkpoint_path).exists():
        if accelerator.is_main_process:
            logger.info("No checkpoint to load, starting from scratch")
        return 0, float('inf')

    if accelerator.is_main_process:
        logger.info(f"Loading checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location='cpu')

    # Determine checkpoint type
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    # Get unwrapped model
    unwrapped_model = accelerator.unwrap_model(model)

    # Load model weights
    load_transformer_only = config.get('checkpoint', {}).get('load_transformer_only', False)

    if load_transformer_only:
        # Only load transformer weights
        transformer_state = {k: v for k, v in state_dict.items() if 'transformer' in k}
        missing, unexpected = unwrapped_model.load_state_dict(transformer_state, strict=False)
        if accelerator.is_main_process:
            logger.info(f"Loaded transformer only. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    else:
        # Load all weights (with potential shape mismatches for encoder/decoder)
        missing, unexpected = unwrapped_model.load_state_dict(state_dict, strict=False)
        if accelerator.is_main_process:
            if missing:
                logger.warning(f"Missing keys: {missing[:5]}..." if len(missing) > 5 else f"Missing keys: {missing}")
            if unexpected:
                logger.warning(f"Unexpected keys: {unexpected[:5]}..." if len(unexpected) > 5 else f"Unexpected keys: {unexpected}")

    if accelerator.is_main_process:
        logger.info("Loaded model weights. Optimizer and scheduler will start fresh.")

    # Always start fresh (step=0, best_val_rmse=inf)
    return 0, float('inf')


@torch.no_grad()
def validate(model, val_loader, accelerator) -> tuple[float, dict]:
    """
    Validate and return RMSE in normalized space, and per-dataset metrics.

    Returns:
        avg_rmse: Average RMSE across all batches (in normalized space)
        per_dataset: Dict of {dataset_name: rmse}
    """
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    # Debug: print expected batch count from sampler
    if hasattr(val_loader, 'batch_sampler'):
        sampler = val_loader.batch_sampler
        print(f"[Rank {rank}] val_loader.batch_sampler: len={len(sampler)}, "
              f"num_batches_per_rank={getattr(sampler, 'num_batches_per_rank', 'N/A')}, "
              f"total_batches={len(getattr(sampler, '_all_batches', []))}")

    # Sync all ranks before validation
    print(f"[Rank {rank}] Before wait_for_everyone (pre-validation)")
    accelerator.wait_for_everyone()
    print(f"[Rank {rank}] After wait_for_everyone (pre-validation)")

    model.eval()

    total_rmse = torch.zeros(1, device=accelerator.device)
    num_batches = torch.zeros(1, device=accelerator.device)

    # Pre-define all dataset names to ensure consistent reduce across ranks
    all_dataset_names = ['diffusion_reaction', '2d_cfd', 'swe', 'ns_incom']
    dataset_rmse = {name: torch.zeros(1, device=accelerator.device) for name in all_dataset_names}
    dataset_count = {name: torch.zeros(1, device=accelerator.device) for name in all_dataset_names}

    batch_count = 0
    for i, batch in enumerate(val_loader):
        print(f"[Rank {rank}] Batch {i}/{len(val_loader)} loaded, dataset={batch['dataset_names'][0]}")

        data = batch['data'].to(device=accelerator.device, dtype=torch.float32)
        channel_mask = batch['channel_mask'].to(device=accelerator.device)
        dataset_names = batch['dataset_names']

        print(f"[Rank {rank}] Batch {i} moved to GPU")

        input_data = data[:, :-1]  # [B, 16, H, W, 18]
        target_data = data[:, 1:]   # [B, 16, H, W, 18]

        # Get normalized output and normalization params
        output_norm, mean, std = model(input_data, return_normalized=True)
        print(f"[Rank {rank}] Batch {i} forward done")

        # Normalize target with same mean/std
        target_norm = (target_data - mean) / std

        # Compute RMSE in normalized space
        rmse = compute_normalized_rmse_loss(output_norm.float(), target_norm.float(), channel_mask)

        total_rmse += rmse.detach()
        num_batches += 1

        # Track per-dataset
        ds_name = dataset_names[0]
        if ds_name in dataset_rmse:
            dataset_rmse[ds_name] += rmse.detach()
            dataset_count[ds_name] += 1

        batch_count += 1

    # Debug: print actual batch count
    print(f"[Rank {rank}] Finished validation loop, batch_count={batch_count}")

    # Sync before reduce to ensure all ranks finished their loops
    print(f"[Rank {rank}] Before wait_for_everyone (pre-reduce)")
    accelerator.wait_for_everyone()
    print(f"[Rank {rank}] After wait_for_everyone (pre-reduce)")

    # Reduce across GPUs
    print(f"[Rank {rank}] Before reduce (total_rmse)")
    total_rmse = accelerator.reduce(total_rmse, reduction='sum')
    print(f"[Rank {rank}] After reduce (total_rmse)")

    print(f"[Rank {rank}] Before reduce (num_batches)")
    num_batches = accelerator.reduce(num_batches, reduction='sum')
    print(f"[Rank {rank}] After reduce (num_batches)")

    # Reduce per-dataset metrics (all ranks have same keys now)
    for ds_name in all_dataset_names:
        print(f"[Rank {rank}] Before reduce ({ds_name}_rmse)")
        dataset_rmse[ds_name] = accelerator.reduce(dataset_rmse[ds_name], reduction='sum')
        print(f"[Rank {rank}] Before reduce ({ds_name}_count)")
        dataset_count[ds_name] = accelerator.reduce(dataset_count[ds_name], reduction='sum')

    print(f"[Rank {rank}] All reduces done")

    # Sync after reduce
    print(f"[Rank {rank}] Before wait_for_everyone (post-reduce)")
    accelerator.wait_for_everyone()
    print(f"[Rank {rank}] After wait_for_everyone (post-reduce)")

    model.train()

    avg_rmse = (total_rmse / num_batches).item() if num_batches.item() > 0 else 0

    per_dataset = {}
    for ds_name in all_dataset_names:
        count = dataset_count[ds_name].item()
        if count > 0:
            per_dataset[ds_name] = dataset_rmse[ds_name].item() / count

    return avg_rmse, per_dataset


def save_checkpoint(
    model, optimizer, scheduler, global_step: int,
    val_rmse: float, best_val_rmse: float,
    config: dict, save_dir: Path, accelerator, filename: str
):
    """Save checkpoint with all training state."""
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        checkpoint = {
            'global_step': global_step,
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'val_rmse': val_rmse,
            'best_val_rmse': best_val_rmse,
            'config': config,
        }
        torch.save(checkpoint, save_dir / filename)
    accelerator.wait_for_everyone()


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config['dataset']['seed'])

    model_name = config.get('model_name', 'pde_pretrain')
    hidden_size = config['model']['transformer']['hidden_size']
    num_layers = config['model']['transformer']['num_hidden_layers']
    warmup_steps = config['training'].get('warmup_steps', 500)
    max_epochs = config['training']['max_epochs']
    eval_interval = config['training'].get('eval_interval', 200)

    # Override resume from command line if provided
    resume_path = args.resume or config.get('checkpoint', {}).get('resume_from')

    # DDP kwargs
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)

    accelerator = Accelerator(
        mixed_precision=config['training'].get('mixed_precision', 'bf16'),
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 2),
        log_with="wandb",
        kwargs_handlers=[ddp_kwargs]
    )

    log_interval = config['logging']['log_interval']
    early_stopping_patience = config['training'].get('early_stopping_patience', 30)
    grad_clip = config['training'].get('grad_clip', 5.0)

    # Create dataloaders
    data_dir = config['dataset']['path']
    batch_size = config['dataloader']['batch_size']
    num_workers = config['dataloader']['num_workers']
    seed = config['dataset']['seed']

    train_loader, val_loader, train_sampler, val_sampler = create_pretrain_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=config['dataloader']['pin_memory'],
        seed=seed,
    )

    # Calculate total steps
    steps_per_epoch = len(train_sampler)
    total_steps = max_epochs * steps_per_epoch

    if accelerator.is_main_process:
        logger.info(f"{'='*60}")
        logger.info(f"PDE Foundation Model Pretraining")
        logger.info(f"{'='*60}")
        logger.info(f"Config: {args.config}")
        logger.info(f"Model: {model_name} (hidden={hidden_size}, layers={num_layers})")
        logger.info(f"Channels: {config['model']['vector_channels']} vector + {config['model']['scalar_channels']} scalar = {config['model']['in_channels']}")
        logger.info(f"Max Epochs: {max_epochs}")
        logger.info(f"Steps per Epoch: {steps_per_epoch}")
        logger.info(f"Total Steps: {total_steps}")
        logger.info(f"Warmup Steps: {warmup_steps}")
        logger.info(f"Eval Interval: {eval_interval} steps")
        logger.info(f"Resume from: {resume_path or 'scratch'}")
        logger.info(f"{'='*60}")

    # Create model
    model = PDECausalModel(config)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        betas=tuple(config['training'].get('betas', [0.9, 0.999]))
    )

    # Create scheduler
    scheduler = get_lr_scheduler(optimizer, config, total_steps)

    # Prepare with accelerator
    model, optimizer = accelerator.prepare(model, optimizer)

    # Load checkpoint (model only, optimizer/scheduler start fresh)
    global_step, best_val_rmse = load_checkpoint(
        model, resume_path, config, accelerator
    )

    # Init WandB
    if accelerator.is_main_process:
        run_name = f"pretrain-{model_name}-h{hidden_size}-L{num_layers}"
        accelerator.init_trackers(
            project_name=config['logging']['project'],
            config=config,
            init_kwargs={"wandb": {
                "entity": config['logging'].get('entity'),
                "name": run_name,
                "tags": ["pretrain", "18ch", f"h{hidden_size}", f"L{num_layers}"],
            }}
        )

    save_dir = Path(config['logging']['save_dir'])
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)

    patience_counter = 0
    console = Console()
    early_stop = False

    # Calculate starting epoch
    start_epoch = global_step // steps_per_epoch if steps_per_epoch > 0 else 0

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
        train_task = progress.add_task("Pretraining", total=total_steps, completed=global_step)

        for epoch in range(start_epoch, max_epochs):
            if early_stop:
                break

            # Set epoch for samplers
            train_sampler.set_epoch(epoch)

            for batch in train_loader:
                data = batch['data'].to(device=accelerator.device, dtype=torch.float32)
                channel_mask = batch['channel_mask'].to(device=accelerator.device)

                input_data = data[:, :-1]  # [B, 16, H, W, 18]
                target_data = data[:, 1:]   # [B, 16, H, W, 18]

                in_warmup = global_step < warmup_steps

                with accelerator.accumulate(model):
                    # Get normalized output and normalization params
                    output_norm, mean, std = model(input_data, return_normalized=True)

                    # Normalize target with same mean/std
                    target_norm = (target_data - mean) / std

                    # Compute RMSE in normalized space
                    rmse_loss = compute_normalized_rmse_loss(
                        output_norm.float(), target_norm.float(), channel_mask
                    )

                    accelerator.backward(rmse_loss)

                    if grad_clip:
                        accelerator.clip_grad_norm_(model.parameters(), grad_clip)

                    optimizer.step()
                    optimizer.zero_grad()

                scheduler.step()
                global_step += 1

                # Reduce loss across GPUs
                rmse_reduced = accelerator.reduce(rmse_loss.detach(), reduction='mean')

                # Update progress bar
                phase_str = "[warmup]" if in_warmup else f"[E{epoch+1}]"
                progress.update(
                    train_task, advance=1,
                    description=f"{phase_str} RMSE={rmse_reduced.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}"
                )

                # Log to wandb
                if global_step % log_interval == 0:
                    accelerator.log({
                        'train/rmse': rmse_reduced.item(),
                        'train/lr': scheduler.get_last_lr()[0],
                        'train/epoch': epoch + 1,
                    }, step=global_step)

                # Validate
                if global_step % eval_interval == 0:
                    print(f"[Rank {accelerator.process_index}] Entering validation at step {global_step}")
                    accelerator.wait_for_everyone()
                    print(f"[Rank {accelerator.process_index}] Calling validate()")
                    val_rmse, per_dataset = validate(model, val_loader, accelerator)
                    print(f"[Rank {accelerator.process_index}] validate() returned")

                    # Log validation metrics
                    log_dict = {'val/rmse': val_rmse}
                    for ds_name, ds_rmse in per_dataset.items():
                        log_dict[f'val/rmse_{ds_name}'] = ds_rmse
                    accelerator.log(log_dict, step=global_step)

                    if accelerator.is_main_process:
                        ds_str = ", ".join([f"{k}={v:.4f}" for k, v in per_dataset.items()])
                        console.print(f"\n[green]Step {global_step}/{total_steps} {phase_str}:[/green] val_rmse={val_rmse:.6f}")
                        if ds_str:
                            console.print(f"[dim]  Per-dataset: {ds_str}[/dim]")

                    # Post-warmup: save best model and early stopping
                    if not in_warmup:
                        if val_rmse < best_val_rmse:
                            best_val_rmse = val_rmse
                            patience_counter = 0
                            save_checkpoint(
                                model, optimizer, scheduler, global_step,
                                val_rmse, best_val_rmse,
                                config, save_dir, accelerator, 'best.pt'
                            )
                            if accelerator.is_main_process:
                                console.print(f"[yellow]Saved best model[/yellow] (val_rmse: {val_rmse:.6f})")
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
        table = Table(title="Pretraining Complete", show_header=False, border_style="green")
        table.add_row("Total Epochs", str(epoch + 1))
        table.add_row("Total Steps", str(global_step))
        table.add_row("Best Val RMSE", f"{best_val_rmse:.6f}")
        table.add_row("Checkpoint", str(save_dir / "best.pt"))
        console.print(table)


if __name__ == "__main__":
    main()
