"""
End-to-End Training for PDE Causal Autoregressive Model.

Features:
- nRMSE loss (per-sample, per-channel, matching metrics.py)
- Warmup phase: no model saving during warmup steps
- Early stopping based on validation nRMSE
- Logs nRMSE and RMSE to wandb

Usage:
    OMP_NUM_THREADS=6  torchrun --nproc_per_node=8 train_e2e.py --config configs/e2e_medium.yaml
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
from pipeline import PDECausalModel

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
    parser = argparse.ArgumentParser(description="End-to-End PDE Training")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_dataloaders(config: dict):
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

    warmup_steps = config['training'].get('warmup_steps', 200)
    max_steps = config['training']['max_steps']
    min_lr_ratio = config['training'].get('min_lr', 1e-6) / config['training']['learning_rate']

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

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
        data = batch['data'].to(device=accelerator.device, dtype=torch.bfloat16)
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
        }
        torch.save(checkpoint, save_dir / filename)
    accelerator.wait_for_everyone()


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config['dataset']['seed'])

    model_name = config.get('model_name', 'pde_e2e')
    hidden_size = config['model']['transformer']['hidden_size']
    num_layers = config['model']['transformer']['num_hidden_layers']
    warmup_steps = config['training'].get('warmup_steps', 200)

    # DDP kwargs
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        mixed_precision=config['training'].get('mixed_precision', 'bf16'),
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1),
        log_with="wandb",
        kwargs_handlers=[ddp_kwargs]
    )

    max_steps = config['training']['max_steps']
    eval_every_steps = config['training']['eval_every_steps']
    save_every_steps = config['training'].get('save_every_steps', 0)  # 0 = only save best model
    log_interval = config['logging']['log_interval']
    early_stopping_patience = config['training'].get('early_stopping_patience', 20)

    if accelerator.is_main_process:
        logger.info(f"{'='*60}")
        logger.info(f"End-to-End Training")
        logger.info(f"{'='*60}")
        logger.info(f"Config: {args.config}")
        logger.info(f"Model: {model_name} (hidden={hidden_size}, layers={num_layers})")
        logger.info(f"Max Steps: {max_steps}")
        logger.info(f"Warmup Steps: {warmup_steps} (no model saving during warmup)")
        logger.info(f"Loss: nRMSE (per-sample, per-channel, matching metrics.py)")
        logger.info(f"{'='*60}")

    train_loader, val_loader = create_dataloaders(config)
    model = PDECausalModel(config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        betas=tuple(config['training'].get('betas', [0.9, 0.999]))
    )

    scheduler = get_lr_scheduler(optimizer, config)
    model, optimizer = accelerator.prepare(model, optimizer)

    # Init WandB
    if accelerator.is_main_process:
        run_name = f"e2e-{model_name}-h{hidden_size}-L{num_layers}"
        accelerator.init_trackers(
            project_name=config['logging']['project'],
            config=config,
            init_kwargs={"wandb": {
                "entity": config['logging'].get('entity'),
                "name": run_name,
                "tags": ["e2e", f"h{hidden_size}", f"L{num_layers}"],
            }}
        )

    save_dir = Path(config['logging']['save_dir'])
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_val_nrmse = float('inf')
    patience_counter = 0

    train_iter = infinite_dataloader(train_loader)
    console = Console()

    model.train()
    val_nrmse, val_rmse = 0.0, 0.0  # Initialize for periodic checkpoint

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
            data = batch['data'].to(device=accelerator.device, dtype=torch.bfloat16)
            channel_mask = batch['channel_mask'].to(device=accelerator.device)

            input_data = data[:, :-1]
            target_data = data[:, 1:]

            # Check if in warmup phase (no model saving during warmup)
            in_warmup = global_step < warmup_steps

            with accelerator.accumulate(model):
                output = model(input_data)

                # Compute nRMSE loss (per-sample, per-channel)
                nrmse_loss, rmse_loss = compute_nrmse_loss(
                    output.float(), target_data.float(), channel_mask
                )

                # Use nRMSE as training loss (全程使用 nRMSE)
                train_loss = nrmse_loss

                accelerator.backward(train_loss)

                if config['training'].get('grad_clip'):
                    accelerator.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])

                optimizer.step()
                optimizer.zero_grad()

            scheduler.step()
            global_step += 1

            # Reduce loss across GPUs for logging
            nrmse_reduced = accelerator.reduce(nrmse_loss.detach(), reduction='mean')
            rmse_reduced = accelerator.reduce(rmse_loss.detach(), reduction='mean')

            # Update progress bar
            phase_str = "[warmup]" if in_warmup else "[train]"
            progress.update(train_task, advance=1,
                          description=f"{phase_str} nRMSE={nrmse_reduced.item():.4f} RMSE={rmse_reduced.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}")

            # Log to wandb (nRMSE first, then RMSE)
            if global_step % log_interval == 0:
                accelerator.log({
                    'train/nrmse': nrmse_reduced.item(),
                    'train/rmse': rmse_reduced.item(),
                    'train/lr': scheduler.get_last_lr()[0],
                }, step=global_step)

            # Evaluate
            if global_step % eval_every_steps == 0:
                accelerator.wait_for_everyone()
                val_nrmse, val_rmse = validate(model, val_loader, accelerator)

                # Log to wandb (nRMSE first, then RMSE)
                accelerator.log({
                    'val/nrmse': val_nrmse,
                    'val/rmse': val_rmse,
                }, step=global_step)

                if accelerator.is_main_process:
                    console.print(f"[green]Step {global_step} {phase_str}:[/green] val_nrmse={val_nrmse:.6f}, val_rmse={val_rmse:.6f}")

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
                        break
                else:
                    if accelerator.is_main_process:
                        console.print(f"[dim](warmup phase - no model saving)[/dim]")

            # Save periodic checkpoint (also skip during warmup)
            if save_every_steps > 0 and global_step % save_every_steps == 0 and not in_warmup:
                save_checkpoint(model, optimizer, scheduler, global_step,
                              val_rmse, val_nrmse,
                              best_val_nrmse, config, save_dir, accelerator, 'latest.pt')
                if accelerator.is_main_process:
                    console.print(f"[cyan]Saved checkpoint[/cyan] at step {global_step}")

    accelerator.end_training()

    if accelerator.is_main_process:
        table = Table(title="Training Complete", show_header=False, border_style="green")
        table.add_row("Total Steps", str(global_step))
        table.add_row("Best Val nRMSE", f"{best_val_nrmse:.6f}")
        table.add_row("Checkpoint", str(save_dir / "best.pt"))
        console.print(table)


if __name__ == "__main__":
    main()
