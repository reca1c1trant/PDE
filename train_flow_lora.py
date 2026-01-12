"""
LoRA Finetuning for 2D Flow Mixing Equation.

Features:
- Pure PDE residual loss (default, no ground truth supervision)
- Optional RMSE loss via lambda_rmse config
- LoRA applied to Transformer layers only
- Encoder and Decoder frozen
- Boundary conditions used for PDE loss computation

Usage:
    torchrun --nproc_per_node=8 train_flow_lora.py --config configs/finetune_flow.yaml
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

from dataset_flow import FlowMixingDataset, FlowMixingSampler, flow_mixing_collate_fn
from model_lora import PDELoRAModel, save_lora_checkpoint
from pde_loss_flow import flow_mixing_pde_loss

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA Finetuning for Flow Mixing PDE")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_dataloaders(config: dict):
    """Create train and validation dataloaders."""
    clips_per_sample = config['dataset'].get('clips_per_sample', 100)

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
        clips_per_sample=None,  # Use all clips for validation
    )

    batch_size = config['dataloader']['batch_size']
    seed = config['dataset']['seed']

    train_sampler = FlowMixingSampler(train_dataset, batch_size, shuffle=True, seed=seed)
    val_sampler = FlowMixingSampler(val_dataset, batch_size, shuffle=False, seed=seed)

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=flow_mixing_collate_fn,
        num_workers=config['dataloader']['num_workers'],
        pin_memory=config['dataloader']['pin_memory']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=flow_mixing_collate_fn,
        num_workers=config['dataloader']['num_workers'],
        pin_memory=config['dataloader']['pin_memory']
    )

    return train_loader, val_loader, train_sampler, val_sampler


def get_lr_scheduler(optimizer, config, total_steps: int):
    """Create learning rate scheduler with warmup and cosine decay."""
    from torch.optim.lr_scheduler import LambdaLR
    import math

    warmup_steps = config['training'].get('warmup_steps', 100)
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


def compute_pde_loss(output, input_data, batch, config, accelerator):
    """
    Compute PDE residual loss for Flow Mixing equation.

    Args:
        output: Model output [B, 16, H, W, 6]
        input_data: Model input [B, 16, H, W, 6]
        batch: Batch dict with boundaries and vtmax
        config: Config dict
        accelerator: Accelerator instance

    Returns:
        pde_loss: Scalar PDE loss
        loss_time: Time derivative loss
        loss_advection: Advection loss
    """
    # Prepend input's first frame to output
    t0_frame = input_data[:, 0:1]  # [B, 1, H, W, 6]
    pred_with_t0 = torch.cat([t0_frame, output], dim=1)  # [B, 17, H, W, 6]

    # Extract u channel and convert to float32
    pred_u = pred_with_t0[..., :1].float()  # [B, 17, H, W, 1]

    # Get boundaries
    boundary_left = batch['boundary_left'].to(accelerator.device).float()
    boundary_right = batch['boundary_right'].to(accelerator.device).float()
    boundary_bottom = batch['boundary_bottom'].to(accelerator.device).float()
    boundary_top = batch['boundary_top'].to(accelerator.device).float()
    vtmax = batch['vtmax'].to(accelerator.device).float()

    # Get dt from config
    dt = config.get('physics', {}).get('dt', 1/999)

    # Use mean vtmax for batch
    vtmax_mean = vtmax.mean().item()

    # Compute PDE loss
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


def compute_rmse_loss(output, target):
    """
    Compute RMSE loss between output and ground truth.

    Args:
        output: Model output [B, 16, H, W, 6]
        target: Ground truth [B, 16, H, W, 6]

    Returns:
        rmse_loss: RMSE loss (only on valid channel u)
    """
    # Only compute on channel 0 (u)
    pred_u = output[..., 0]
    target_u = target[..., 0]
    mse = torch.mean((pred_u - target_u) ** 2)
    rmse_loss = torch.sqrt(mse + 1e-8)
    return rmse_loss


@torch.no_grad()
def validate(model, val_loader, config, accelerator):
    """Validate model on validation set."""
    model.eval()

    total_pde_loss = torch.zeros(1, device=accelerator.device)
    total_rmse_loss = torch.zeros(1, device=accelerator.device)
    num_batches = torch.zeros(1, device=accelerator.device)

    for batch in val_loader:
        data = batch['data'].to(device=accelerator.device, dtype=torch.bfloat16)

        input_data = data[:, :-1]  # [B, 16, H, W, 6]
        target = data[:, 1:]       # [B, 16, H, W, 6]

        output = model(input_data)

        pde_loss, _, _ = compute_pde_loss(output, input_data, batch, config, accelerator)
        rmse_loss = compute_rmse_loss(output.float(), target.float())

        total_pde_loss += pde_loss.detach()
        total_rmse_loss += rmse_loss.detach()
        num_batches += 1

    accelerator.wait_for_everyone()

    total_pde_loss = accelerator.reduce(total_pde_loss, reduction='sum')
    total_rmse_loss = accelerator.reduce(total_rmse_loss, reduction='sum')
    num_batches = accelerator.reduce(num_batches, reduction='sum')

    model.train()

    avg_pde_loss = (total_pde_loss / num_batches).item() if num_batches.item() > 0 else 0
    avg_rmse_loss = (total_rmse_loss / num_batches).item() if num_batches.item() > 0 else 0

    return avg_pde_loss, avg_rmse_loss


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
    lambda_pde = config['training'].get('lambda_pde', 1.0)
    lambda_rmse = config['training'].get('lambda_rmse', 0.0)  # Default: no RMSE loss
    grad_clip = config['training'].get('grad_clip', 1.0)
    clips_per_sample = config['dataset'].get('clips_per_sample', 100)
    eval_interval = config['training'].get('eval_interval', 100)
    early_stopping_patience = config['training'].get('early_stopping_patience', 10)

    if accelerator.is_main_process:
        logger.info(f"{'='*60}")
        logger.info(f"Flow Mixing LoRA Finetuning")
        logger.info(f"{'='*60}")
        logger.info(f"Config: {args.config}")
        logger.info(f"Max Epochs: {max_epochs}")
        logger.info(f"Clips per Sample: {clips_per_sample}")
        logger.info(f"Warmup Steps: {warmup_steps}")
        logger.info(f"Eval Interval: {eval_interval} steps")
        logger.info(f"Early Stopping Patience: {early_stopping_patience}")
        logger.info(f"Loss: lambda_pde={lambda_pde}, lambda_rmse={lambda_rmse}")
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

    model, optimizer = accelerator.prepare(model, optimizer)

    # Init WandB
    if accelerator.is_main_process:
        lora_r = config['model'].get('lora', {}).get('r', 16)
        run_name = f"flow-lora-r{lora_r}"
        accelerator.init_trackers(
            project_name=config['logging']['project'],
            config=config,
            init_kwargs={"wandb": {
                "entity": config['logging'].get('entity'),
                "name": run_name,
                "tags": ["flow-mixing", "lora", "pde-loss"],
            }}
        )

    # Save directory
    save_dir = Path(config['logging']['save_dir'])
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Training state
    global_step = 0
    best_pde_loss = float('inf')
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

                data = batch['data'].to(device=accelerator.device, dtype=torch.bfloat16)

                input_data = data[:, :-1]  # [B, 16, H, W, 6]
                target = data[:, 1:]       # [B, 16, H, W, 6]

                in_warmup = global_step < warmup_steps

                with accelerator.accumulate(model):
                    output = model(input_data)

                    # PDE loss
                    pde_loss, loss_time, loss_advection = compute_pde_loss(
                        output, input_data, batch, config, accelerator
                    )

                    # MSE loss (optional)
                    rmse_loss = torch.tensor(0.0, device=accelerator.device)
                    if lambda_rmse > 0:
                        rmse_loss = compute_rmse_loss(output.float(), target.float())

                    # Total loss
                    train_loss = lambda_pde * pde_loss + lambda_rmse * rmse_loss

                    accelerator.backward(train_loss)

                    if grad_clip > 0:
                        accelerator.clip_grad_norm_(trainable_params, grad_clip)

                    optimizer.step()
                    optimizer.zero_grad()

                scheduler.step()
                global_step += 1
                epoch_steps += 1

                # Reduce for logging
                pde_loss_reduced = accelerator.reduce(pde_loss.detach(), reduction='mean')
                loss_time_reduced = accelerator.reduce(loss_time.detach(), reduction='mean')
                loss_advection_reduced = accelerator.reduce(loss_advection.detach(), reduction='mean')

                epoch_pde_loss += pde_loss_reduced.item()
                if lambda_rmse > 0:
                    rmse_loss_reduced = accelerator.reduce(rmse_loss.detach(), reduction='mean')
                    epoch_rmse_loss += rmse_loss_reduced.item()

                # Update progress bar
                phase_str = "[warmup]" if in_warmup else f"[E{epoch+1}]"
                progress.update(
                    train_task, advance=1,
                    description=f"{phase_str} PDE={pde_loss_reduced.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}"
                )

                # Log to wandb
                if global_step % log_interval == 0:
                    log_dict = {
                        'train/pde_loss': pde_loss_reduced.item(),
                        'train/loss_time': loss_time_reduced.item(),
                        'train/loss_advection': loss_advection_reduced.item(),
                        'train/lr': scheduler.get_last_lr()[0],
                        'train/epoch': epoch + 1,
                    }
                    if lambda_rmse > 0:
                        log_dict['train/rmse_loss'] = rmse_loss_reduced.item()
                    accelerator.log(log_dict, step=global_step)

                # Step-based validation
                if global_step % eval_interval == 0:
                    accelerator.wait_for_everyone()
                    val_pde_loss, val_rmse_loss = validate(model, val_loader, config, accelerator)

                    accelerator.log({
                        'val/pde_loss': val_pde_loss,
                        'val/rmse_loss': val_rmse_loss,
                    }, step=global_step)

                    if accelerator.is_main_process:
                        console.print(
                            f"\n[green]Step {global_step}/{total_steps} {phase_str}:[/green] "
                            f"val_pde={val_pde_loss:.6f}, val_rmse={val_rmse_loss:.6f}"
                        )

                    # Post-warmup: best model saving and early stopping
                    if not in_warmup:
                        if val_pde_loss < best_pde_loss:
                            best_pde_loss = val_pde_loss
                            patience_counter = 0

                            if accelerator.is_main_process:
                                unwrapped_model = accelerator.unwrap_model(model)
                                save_lora_checkpoint(
                                    model=unwrapped_model.model,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    global_step=global_step,
                                    metrics={'pde_loss': val_pde_loss, 'rmse_loss': val_rmse_loss},
                                    save_path=str(save_dir / 'best_lora.pt'),
                                    config=config,
                                )
                                console.print(f"[yellow]Saved best model[/yellow] (pde_loss: {val_pde_loss:.6f})")
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
                avg_epoch_pde = epoch_pde_loss / epoch_steps
                log_dict = {
                    'epoch/train_pde_loss': avg_epoch_pde,
                    'epoch': epoch + 1,
                }
                if lambda_rmse > 0:
                    log_dict['epoch/train_rmse_loss'] = epoch_rmse_loss / epoch_steps
                accelerator.log(log_dict, step=global_step)

                if accelerator.is_main_process:
                    console.print(
                        f"\n[blue]Epoch {epoch+1}/{max_epochs} completed:[/blue] "
                        f"avg_train_pde={avg_epoch_pde:.6f}"
                    )

    accelerator.end_training()

    if accelerator.is_main_process:
        table = Table(title="Training Complete", show_header=False, border_style="green")
        table.add_row("Total Epochs", str(epoch + 1))
        table.add_row("Total Steps", str(global_step))
        table.add_row("Best Val PDE Loss", f"{best_pde_loss:.6f}")
        table.add_row("Early Stopped", "Yes" if early_stop else "No")
        table.add_row("Checkpoint", str(save_dir / "best_lora.pt"))
        console.print(table)


if __name__ == "__main__":
    main()
