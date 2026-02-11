"""
LoRA Finetuning for 2D Burgers Equation using PDEModelV2.

Loss: BC loss + PDE loss (NO RMSE - that would be cheating!)
- BC loss: RMSE on boundary pixels (outermost interior points)
- PDE loss: Burgers equation residual

Uses Neighborhood Attention model with LoRA on:
- NA Attention: qkv, proj
- FFN: gate_proj, up_proj, down_proj

Usage:
    torchrun --nproc_per_node=8 train_burgers_lora.py --config configs/finetune_burgers.yaml
"""

import os
import sys
import warnings

def _is_main_process():
    return os.environ.get('LOCAL_RANK', '0') == '0'

IS_MAIN_PROCESS = _is_main_process()

if not IS_MAIN_PROCESS:
    warnings.filterwarnings('ignore')

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
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    TaskProgressColumn, TimeRemainingColumn, MofNCompleteColumn
)
from rich.table import Table
import logging

torch.set_float32_matmul_precision('high')

if IS_MAIN_PROCESS:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
else:
    logging.disable(logging.CRITICAL)

from dataset_finetune import (
    FinetuneDataset, FinetuneSampler, finetune_collate_fn,
    create_finetune_dataloaders, TOTAL_CHANNELS
)
from model_lora_v2 import PDELoRAModelV2, save_lora_checkpoint, load_lora_checkpoint
from pde_loss import burgers_pde_loss

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA Finetuning for Burgers (V2 Model)")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume')
    parser.add_argument('--reset_patience', action='store_true', help='Reset early stopping')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_lr_scheduler(optimizer, config: dict, total_steps: int):
    """Cosine LR with warmup."""
    from torch.optim.lr_scheduler import LambdaLR
    import math

    warmup_steps = config['training'].get('warmup_steps', 100)
    min_lr = config['training'].get('min_lr', 1e-6)
    base_lr = config['training']['learning_rate']
    min_lr_ratio = min_lr / base_lr

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def compute_boundary_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    channel_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute boundary RMSE on 4 edges (outermost interior points).

    These are the points at coordinates like (1/256, 1/256), (3/256, 1/256), etc.
    """
    if channel_mask.dim() == 1:
        valid_ch = torch.where(channel_mask > 0)[0]
    else:
        valid_ch = torch.where(channel_mask[0] > 0)[0]

    # Left edge (x = 1/256, all y)
    left_pred = pred[:, :, :, 0, :][:, :, :, valid_ch]
    left_target = target[:, :, :, 0, :][:, :, :, valid_ch]

    # Right edge (x = 255/256, all y)
    right_pred = pred[:, :, :, -1, :][:, :, :, valid_ch]
    right_target = target[:, :, :, -1, :][:, :, :, valid_ch]

    # Bottom edge (y = 1/256, all x)
    bottom_pred = pred[:, :, 0, :, :][:, :, :, valid_ch]
    bottom_target = target[:, :, 0, :, :][:, :, :, valid_ch]

    # Top edge (y = 255/256, all x)
    top_pred = pred[:, :, -1, :, :][:, :, :, valid_ch]
    top_target = target[:, :, -1, :, :][:, :, :, valid_ch]

    bc_pred = torch.cat([
        left_pred.reshape(-1), right_pred.reshape(-1),
        bottom_pred.reshape(-1), top_pred.reshape(-1),
    ])
    bc_target = torch.cat([
        left_target.reshape(-1), right_target.reshape(-1),
        bottom_target.reshape(-1), top_target.reshape(-1),
    ])

    mse = torch.mean((bc_pred - bc_target) ** 2)
    return torch.sqrt(mse + 1e-8)


def compute_pde_loss(
    output: torch.Tensor,
    input_data: torch.Tensor,
    batch: dict,
    config: dict,
    device: torch.device,
) -> tuple:
    """
    Compute PDE residual loss for Burgers equation.

    Uses ghost cell extrapolation for boundary derivatives.
    PDE loss computed on [1:127, 1:127] (126x126 interior points).

    Args:
        output: Model output [B, T_out, H, W, C] (denormalized)
        input_data: Model input [B, T_in, H, W, C]
        batch: Batch dict with nu
        config: Config dict
        device: Device

    Returns:
        pde_loss, loss_u, loss_v
    """
    with torch.autocast(device_type='cuda', enabled=False):
        # Prepend t0 frame to output for time derivative
        t0_frame = input_data[:, 0:1, ..., :2].float()
        output_uv = output[..., :2].float()
        pred_with_t0 = torch.cat([t0_frame, output_uv], dim=1)

        nu = batch['nu'].to(device).float()
        nu_mean = nu.mean().item()

        # Physics params
        dt = config.get('physics', {}).get('dt', 1/999)
        dx = config.get('physics', {}).get('dx', 1/127)
        dy = config.get('physics', {}).get('dy', 1/127)

        # Compute PDE loss (ghost cell version)
        pde_loss, loss_u, loss_v, _, _ = burgers_pde_loss(
            pred=pred_with_t0,
            nu=nu_mean,
            dt=dt,
            dx=dx,
            dy=dy,
        )

    return pde_loss, loss_u, loss_v


@torch.no_grad()
def validate(model, val_loader, accelerator, config, t_input: int = 8):
    """Validate and return (bc_loss, pde_loss, rmse)."""
    accelerator.wait_for_everyone()
    model.eval()

    total_bc = torch.zeros(1, device=accelerator.device)
    total_pde = torch.zeros(1, device=accelerator.device)
    total_rmse = torch.zeros(1, device=accelerator.device)
    num_batches = torch.zeros(1, device=accelerator.device)

    total_val_batches = len(val_loader)
    if accelerator.is_main_process:
        logger.info(f"[Val] Starting validation: {total_val_batches} batches")

    for batch_idx, batch in enumerate(val_loader):
        if accelerator.is_main_process:
            logger.info(f"[Val][{batch_idx+1}/{total_val_batches}] Loading data...")

        data = batch['data'].to(device=accelerator.device, dtype=torch.float32)
        channel_mask = batch['channel_mask'].to(device=accelerator.device)

        if accelerator.is_main_process:
            logger.info(f"[Val][{batch_idx+1}] Data shape: {data.shape}, doing forward...")

        input_data = data[:, :t_input]
        target_data = data[:, 1:t_input + 1]

        # Forward (get denormalized output)
        output_norm, mean, std = model(input_data, return_normalized=True)
        output = output_norm * std + mean

        if accelerator.is_main_process:
            logger.info(f"[Val][{batch_idx+1}] Forward done, computing BC loss...")

        # BC loss (denormalized)
        bc_loss = compute_boundary_loss(output, target_data, channel_mask)

        if accelerator.is_main_process:
            logger.info(f"[Val][{batch_idx+1}] BC done={bc_loss.item():.4f}, computing PDE loss...")

        # PDE loss (denormalized, requires nu)
        if 'nu' in batch:
            pde_loss, _, _ = compute_pde_loss(output, input_data, batch, config, accelerator.device)
        else:
            pde_loss = torch.tensor(0.0, device=accelerator.device)

        if accelerator.is_main_process:
            logger.info(f"[Val][{batch_idx+1}] PDE done={pde_loss.item():.4f}, computing RMSE...")

        # RMSE (denormalized, for monitoring only)
        valid_ch = torch.where(channel_mask[0] > 0)[0] if channel_mask.dim() > 1 else torch.where(channel_mask > 0)[0]
        output_valid = output[..., valid_ch]
        target_valid = target_data[..., valid_ch]
        mse = torch.mean((output_valid - target_valid) ** 2)
        rmse = torch.sqrt(mse + 1e-8)

        if accelerator.is_main_process:
            logger.info(f"[Val][{batch_idx+1}] RMSE done={rmse.item():.4f}, batch complete!")

        total_bc += bc_loss.detach()
        total_pde += pde_loss.detach()
        total_rmse += rmse.detach()
        num_batches += 1

    if accelerator.is_main_process:
        logger.info(f"[Val] Finished {num_batches.item():.0f} batches, reducing...")

    accelerator.wait_for_everyone()

    total_bc = accelerator.reduce(total_bc, reduction='sum')
    total_pde = accelerator.reduce(total_pde, reduction='sum')
    total_rmse = accelerator.reduce(total_rmse, reduction='sum')
    num_batches = accelerator.reduce(num_batches, reduction='sum')

    accelerator.wait_for_everyone()
    model.train()

    avg_bc = (total_bc / num_batches).item() if num_batches.item() > 0 else 0
    avg_pde = (total_pde / num_batches).item() if num_batches.item() > 0 else 0
    avg_rmse = (total_rmse / num_batches).item() if num_batches.item() > 0 else 0

    return avg_bc, avg_pde, avg_rmse


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config['dataset']['seed'])

    max_epochs = config['training'].get('max_epochs', 20)
    warmup_steps = config['training'].get('warmup_steps', 100)
    log_interval = config['logging']['log_interval']
    lambda_bc = config['training'].get('lambda_bc', 1.0)
    lambda_pde = config['training'].get('lambda_pde', 1.0)
    grad_clip = config['training'].get('grad_clip', 1.0)
    eval_interval = config['training'].get('eval_interval', 100)
    early_stopping_patience = config['training'].get('early_stopping_patience', 15)
    t_input = config['dataset'].get('t_input', 8)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        mixed_precision=config['training'].get('mixed_precision', 'no'),
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1),
        log_with="wandb",
        kwargs_handlers=[ddp_kwargs]
    )

    if accelerator.is_main_process:
        logger.info(f"{'='*60}")
        logger.info(f"Burgers2D LoRA Finetuning (V2 Model)")
        logger.info(f"{'='*60}")
        logger.info(f"Config: {args.config}")
        logger.info(f"Loss: BC + PDE (NO RMSE!)")
        logger.info(f"Lambda BC: {lambda_bc}, Lambda PDE: {lambda_pde}")
        logger.info(f"{'='*60}")

    temporal_length = t_input + 1

    train_loader, val_loader, train_sampler, val_sampler = create_finetune_dataloaders(
        data_path=config['dataset']['path'],
        batch_size=config['dataloader']['batch_size'],
        num_workers=config['dataloader']['num_workers'],
        pin_memory=config['dataloader']['pin_memory'],
        seed=config['dataset']['seed'],
        temporal_length=temporal_length,
        train_ratio=config['dataset'].get('train_ratio', 0.9),
        clips_per_sample=config['dataset'].get('clips_per_sample', 100),
        vector_dim=config['dataset'].get('vector_dim', 2),
        val_time_interval=config['dataset'].get('val_time_interval', 8),
    )

    steps_per_epoch = len(train_sampler)
    total_steps = max_epochs * steps_per_epoch

    if accelerator.is_main_process:
        logger.info(f"Steps per epoch: {steps_per_epoch}")
        logger.info(f"Total steps: {total_steps}")

    pretrained_path = config['model'].get('pretrained_path', None)
    freeze_encoder = config['model'].get('freeze_encoder', False)
    freeze_decoder = config['model'].get('freeze_decoder', False)

    model = PDELoRAModelV2(
        config=config,
        pretrained_path=pretrained_path,
        freeze_encoder=freeze_encoder,
        freeze_decoder=freeze_decoder,
    )

    if config['training'].get('mixed_precision', 'no') == 'no':
        model = model.float()

    trainable_params = model.get_trainable_params()
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        betas=tuple(config['training'].get('betas', [0.9, 0.999]))
    )

    scheduler = get_lr_scheduler(optimizer, config, total_steps)
    model, optimizer = accelerator.prepare(model, optimizer)

    global_step = 0
    best_val_loss = float('inf')
    patience_counter = 0

    if args.resume:
        ckpt = load_lora_checkpoint(
            accelerator.unwrap_model(model), args.resume, optimizer, scheduler
        )
        global_step = ckpt.get('global_step', 0)
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        patience_counter = ckpt.get('patience_counter', 0)
        if args.reset_patience:
            patience_counter = 0

    if accelerator.is_main_process:
        lora_r = config['model'].get('lora', {}).get('r', 16)
        run_name = f"burgers-lora-v2-r{lora_r}"
        accelerator.init_trackers(
            project_name=config['logging']['project'],
            config=config,
            init_kwargs={"wandb": {
                "entity": config['logging'].get('entity'),
                "name": run_name,
                "tags": ["burgers", "lora", "v2", "NA", "BC+PDE"],
            }}
        )

    save_dir = Path(config['logging']['save_dir'])
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)

    console = Console()
    early_stop = False
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
        train_task = progress.add_task("LoRA Training", total=total_steps, completed=global_step)

        for epoch in range(start_epoch, max_epochs):
            if early_stop:
                break

            train_sampler.set_epoch(epoch)
            epoch_bc = 0.0
            epoch_pde = 0.0
            epoch_steps = 0

            for batch in train_loader:
                if early_stop:
                    break

                data = batch['data'].to(device=accelerator.device, dtype=torch.float32)
                channel_mask = batch['channel_mask'].to(device=accelerator.device)

                input_data = data[:, :t_input]
                target_data = data[:, 1:t_input + 1]
                in_warmup = global_step < warmup_steps

                with accelerator.accumulate(model):
                    # Forward pass
                    output_norm, mean, std = model(input_data, return_normalized=True)
                    output = output_norm * std + mean  # Denormalize

                    # BC loss (boundary pixels)
                    bc_loss = compute_boundary_loss(output, target_data, channel_mask)

                    # PDE loss (requires nu, no longer needs boundary)
                    if 'nu' in batch:
                        pde_loss, loss_u, loss_v = compute_pde_loss(
                            output, input_data, batch, config, accelerator.device
                        )
                    else:
                        pde_loss = torch.tensor(0.0, device=accelerator.device)

                    # Total loss = BC + PDE (NO RMSE!)
                    loss = lambda_bc * bc_loss + lambda_pde * pde_loss

                    accelerator.backward(loss)
                    if grad_clip > 0:
                        accelerator.clip_grad_norm_(trainable_params, grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()

                scheduler.step()
                global_step += 1
                epoch_steps += 1

                bc_reduced = accelerator.reduce(bc_loss.detach(), reduction='mean')
                pde_reduced = accelerator.reduce(pde_loss.detach(), reduction='mean')
                epoch_bc += bc_reduced.item()
                epoch_pde += pde_reduced.item()

                phase_str = "[warmup]" if in_warmup else f"[E{epoch+1}]"
                progress.update(
                    train_task, advance=1,
                    description=f"{phase_str} BC={bc_reduced.item():.4f} PDE={pde_reduced.item():.4f}"
                )

                if global_step % log_interval == 0:
                    accelerator.log({
                        'train/bc_loss': bc_reduced.item(),
                        'train/pde_loss': pde_reduced.item(),
                        'train/total_loss': lambda_bc * bc_reduced.item() + lambda_pde * pde_reduced.item(),
                        'train/lr': scheduler.get_last_lr()[0],
                        'train/epoch': epoch + 1,
                    }, step=global_step)

                if global_step % eval_interval == 0:
                    val_bc, val_pde, val_rmse = validate(model, val_loader, accelerator, config, t_input)
                    val_loss = lambda_bc * val_bc + lambda_pde * val_pde

                    accelerator.log({
                        'val/bc_loss': val_bc,
                        'val/pde_loss': val_pde,
                        'val/rmse': val_rmse,
                        'val/total_loss': val_loss,
                    }, step=global_step)

                    if accelerator.is_main_process:
                        console.print(
                            f"\n[green]Step {global_step}/{total_steps} {phase_str}:[/green] "
                            f"val_bc={val_bc:.6f}, val_pde={val_pde:.6f}, val_rmse={val_rmse:.6f}"
                        )

                    if not in_warmup:
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            patience_counter = 0
                            if accelerator.is_main_process:
                                save_lora_checkpoint(
                                    model=accelerator.unwrap_model(model),
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    global_step=global_step,
                                    metrics={'bc': val_bc, 'pde': val_pde, 'rmse': val_rmse},
                                    save_path=str(save_dir / 'best_lora.pt'),
                                    config=config,
                                    patience_counter=patience_counter,
                                    best_val_loss=best_val_loss,
                                )
                                console.print(f"[yellow]Saved best model[/yellow] (val_loss: {val_loss:.6f})")
                        else:
                            patience_counter += 1
                            if accelerator.is_main_process:
                                console.print(f"[dim]Patience: {patience_counter}/{early_stopping_patience}[/dim]")
                            if patience_counter >= early_stopping_patience:
                                if accelerator.is_main_process:
                                    console.print(f"[red]Early stopping![/red]")
                                early_stop = True
                                break
                    else:
                        if accelerator.is_main_process:
                            console.print(f"[dim](warmup - no saving)[/dim]")

                    model.train()

            if epoch_steps > 0 and accelerator.is_main_process:
                avg_bc = epoch_bc / epoch_steps
                avg_pde = epoch_pde / epoch_steps
                console.print(
                    f"\n[blue]Epoch {epoch+1}/{max_epochs}:[/blue] "
                    f"avg_bc={avg_bc:.6f}, avg_pde={avg_pde:.6f}"
                )

    accelerator.end_training()

    if accelerator.is_main_process:
        table = Table(title="Training Complete", show_header=False, border_style="green")
        table.add_row("Total Epochs", str(epoch + 1))
        table.add_row("Total Steps", str(global_step))
        table.add_row("Best Val Loss", f"{best_val_loss:.6f}")
        table.add_row("Early Stopped", "Yes" if early_stop else "No")
        table.add_row("Checkpoint", str(save_dir / "best_lora.pt"))
        console.print(table)


if __name__ == "__main__":
    main()
