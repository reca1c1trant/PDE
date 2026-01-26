"""
LoRA Finetuning V2 for 2D Burgers Equation - Unfrozen Encoder/Decoder.

Key difference from v1:
- Encoder and Decoder are UNFROZEN (trainable)
- LoRA still applied to Transformer layers
- This allows spatial feature adaptation to new physics

Usage:
    torchrun --nproc_per_node=8 train_burgers_lora_v2.py --config configs/finetune_burgers_v2.yaml
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

from dataset_burgers import BurgersDataset, BurgersSampler, burgers_collate_fn
from model_lora import PDELoRAModel
from pde_loss import burgers_pde_loss

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA V2 Finetuning for Burgers PDE (Unfrozen Enc/Dec)")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_dataloaders(config: dict):
    """Create train and validation dataloaders."""
    clips_per_sample = config['dataset'].get('clips_per_sample', 100)

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
        clips_per_sample=None,
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
    """Compute PDE residual loss for Burgers equation."""
    with torch.autocast(device_type='cuda', enabled=False):
        t0_frame = input_data[:, 0:1, ..., :2].float()
        output_uv = output[..., :2].float()
        pred_with_t0 = torch.cat([t0_frame, output_uv], dim=1)

        boundary_left = batch['boundary_left'].to(accelerator.device).float()
        boundary_right = batch['boundary_right'].to(accelerator.device).float()
        boundary_bottom = batch['boundary_bottom'].to(accelerator.device).float()
        boundary_top = batch['boundary_top'].to(accelerator.device).float()
        nu = batch['nu'].to(accelerator.device).float()

        dt = config.get('physics', {}).get('dt', 1/999)
        Lx = config.get('physics', {}).get('Lx', 1.0)
        Ly = config.get('physics', {}).get('Ly', 1.0)
        nu_mean = nu.mean().item()

        pde_loss, loss_u, loss_v, _, _ = burgers_pde_loss(
            pred=pred_with_t0,
            boundary_left=boundary_left,
            boundary_right=boundary_right,
            boundary_bottom=boundary_bottom,
            boundary_top=boundary_top,
            nu=nu_mean,
            dt=dt,
            Lx=Lx,
            Ly=Ly,
        )

    return pde_loss, loss_u, loss_v


def compute_rmse_loss(output, target, channel_mask=None):
    """Compute RMSE loss between output and ground truth."""
    with torch.autocast(device_type='cuda', enabled=False):
        output = output.float()
        target = target.float()

        if channel_mask is not None:
            if channel_mask.dim() == 1:
                mask = channel_mask.view(1, 1, 1, 1, -1).float()
            else:
                mask = channel_mask.view(channel_mask.shape[0], 1, 1, 1, -1).float()

            diff_sq = (output - target) ** 2
            masked_diff_sq = diff_sq * mask
            mse = masked_diff_sq.sum() / (mask.sum() * output.shape[1] * output.shape[2] * output.shape[3])
        else:
            mse = torch.mean((output[..., :2] - target[..., :2]) ** 2)

        rmse_loss = torch.sqrt(mse + 1e-8)

    return rmse_loss


def compute_boundary_loss(output, target, channel_mask=None):
    """Compute boundary RMSE loss on 4 edges."""
    with torch.autocast(device_type='cuda', enabled=False):
        output = output.float()
        target = target.float()

        if channel_mask is not None:
            if channel_mask.dim() == 1:
                real_channels = torch.where(channel_mask > 0)[0]
            else:
                real_channels = torch.where(channel_mask[0] > 0)[0]
        else:
            real_channels = torch.tensor([0, 1], device=output.device)

        left_pred = output[:, :, 1:-1, 0, :][:, :, :, real_channels]
        left_target = target[:, :, 1:-1, 0, :][:, :, :, real_channels]
        right_pred = output[:, :, 1:-1, -1, :][:, :, :, real_channels]
        right_target = target[:, :, 1:-1, -1, :][:, :, :, real_channels]
        bottom_pred = output[:, :, 0, 1:-1, :][:, :, :, real_channels]
        bottom_target = target[:, :, 0, 1:-1, :][:, :, :, real_channels]
        top_pred = output[:, :, -1, 1:-1, :][:, :, :, real_channels]
        top_target = target[:, :, -1, 1:-1, :][:, :, :, real_channels]

        bc_pred = torch.cat([
            left_pred.reshape(-1), right_pred.reshape(-1),
            bottom_pred.reshape(-1), top_pred.reshape(-1),
        ])
        bc_target = torch.cat([
            left_target.reshape(-1), right_target.reshape(-1),
            bottom_target.reshape(-1), top_target.reshape(-1),
        ])

        mse = torch.mean((bc_pred - bc_target) ** 2)
        boundary_rmse = torch.sqrt(mse + 1e-8)

    return boundary_rmse


def unfreeze_encoder_decoder(model):
    """
    Unfreeze Encoder and Decoder parameters.

    Returns parameter counts for logging.
    """
    enc_params = 0
    dec_params = 0

    # Unfreeze encoder
    for param in model.model.encoder_2d.parameters():
        param.requires_grad = True
        enc_params += param.numel()

    # Unfreeze decoder
    for param in model.model.decoder_2d.parameters():
        param.requires_grad = True
        dec_params += param.numel()

    return enc_params, dec_params


def get_all_trainable_params(model):
    """Get all trainable parameters (LoRA + Encoder + Decoder)."""
    return [p for p in model.parameters() if p.requires_grad]


def log_param_summary(model, accelerator):
    """Log parameter summary."""
    if not accelerator.is_main_process:
        return

    # Count parameters
    enc_total = sum(p.numel() for p in model.model.encoder_2d.parameters())
    enc_train = sum(p.numel() for p in model.model.encoder_2d.parameters() if p.requires_grad)

    dec_total = sum(p.numel() for p in model.model.decoder_2d.parameters())
    dec_train = sum(p.numel() for p in model.model.decoder_2d.parameters() if p.requires_grad)

    trans_total = sum(p.numel() for p in model.model.transformer.parameters())
    trans_train = sum(p.numel() for p in model.model.transformer.parameters() if p.requires_grad)

    total = enc_total + dec_total + trans_total
    trainable = enc_train + dec_train + trans_train

    logger.info(f"\n{'='*60}")
    logger.info(f"LoRA V2 Model Parameter Summary (Unfrozen Enc/Dec)")
    logger.info(f"{'='*60}")
    logger.info(f"Encoder:     {enc_total:>12,} total, {enc_train:>10,} trainable")
    logger.info(f"Decoder:     {dec_total:>12,} total, {dec_train:>10,} trainable")
    logger.info(f"Transformer: {trans_total:>12,} total, {trans_train:>10,} trainable (LoRA)")
    logger.info(f"{'-'*60}")
    logger.info(f"Total:       {total:>12,} total, {trainable:>10,} trainable")
    logger.info(f"Trainable ratio: {100 * trainable / total:.2f}%")
    logger.info(f"{'='*60}\n")


def save_full_checkpoint(model, optimizer, scheduler, global_step, metrics, save_path, config):
    """Save full checkpoint (all trainable weights)."""
    # Get all trainable state dict
    trainable_state_dict = {}
    for name, param in model.model.named_parameters():
        if param.requires_grad:
            trainable_state_dict[name] = param.data.clone()

    checkpoint = {
        'global_step': global_step,
        'trainable_state_dict': trainable_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'config': config,
    }

    torch.save(checkpoint, save_path)


@torch.no_grad()
def validate(model, val_loader, config, accelerator):
    """Validate model on validation set."""
    model.eval()

    total_pde_loss = torch.zeros(1, device=accelerator.device)
    total_rmse_loss = torch.zeros(1, device=accelerator.device)
    total_bc_loss = torch.zeros(1, device=accelerator.device)
    num_batches = torch.zeros(1, device=accelerator.device)

    for batch in val_loader:
        data = batch['data'].to(device=accelerator.device, dtype=torch.float32)
        channel_mask = batch['channel_mask'].to(device=accelerator.device)

        input_data = data[:, :-1]
        target = data[:, 1:]

        output = model(input_data)

        pde_loss, _, _ = compute_pde_loss(output, input_data, batch, config, accelerator)
        rmse_loss = compute_rmse_loss(output.float(), target.float(), channel_mask)
        bc_loss = compute_boundary_loss(output.float(), target.float(), channel_mask)

        total_pde_loss += pde_loss.detach()
        total_rmse_loss += rmse_loss.detach()
        total_bc_loss += bc_loss.detach()
        num_batches += 1

    accelerator.wait_for_everyone()

    total_pde_loss = accelerator.reduce(total_pde_loss, reduction='sum')
    total_rmse_loss = accelerator.reduce(total_rmse_loss, reduction='sum')
    total_bc_loss = accelerator.reduce(total_bc_loss, reduction='sum')
    num_batches = accelerator.reduce(num_batches, reduction='sum')

    model.train()

    avg_pde_loss = (total_pde_loss / num_batches).item() if num_batches.item() > 0 else 0
    avg_rmse_loss = (total_rmse_loss / num_batches).item() if num_batches.item() > 0 else 0
    avg_bc_loss = (total_bc_loss / num_batches).item() if num_batches.item() > 0 else 0

    return avg_pde_loss, avg_rmse_loss, avg_bc_loss


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config['dataset']['seed'])

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        mixed_precision=config['training'].get('mixed_precision', 'no'),
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1),
        log_with="wandb",
        kwargs_handlers=[ddp_kwargs]
    )

    # Training params
    max_epochs = config['training'].get('max_epochs', 10)
    warmup_steps = config['training'].get('warmup_steps', 100)
    log_interval = config['logging']['log_interval']
    lambda_pde = config['training'].get('lambda_pde', 1.0)
    lambda_bc = config['training'].get('lambda_bc', 1.0)
    grad_clip = config['training'].get('grad_clip', 1.0)
    clips_per_sample = config['dataset'].get('clips_per_sample', 100)
    eval_interval = config['training'].get('eval_interval', 100)
    early_stopping_patience = config['training'].get('early_stopping_patience', 10)

    if accelerator.is_main_process:
        logger.info(f"{'='*60}")
        logger.info(f"Burgers2D LoRA V2 - UNFROZEN Encoder/Decoder")
        logger.info(f"{'='*60}")
        logger.info(f"Config: {args.config}")
        logger.info(f"Max Epochs: {max_epochs}")
        logger.info(f"Clips per Sample: {clips_per_sample}")
        logger.info(f"Loss: lambda_pde={lambda_pde}, lambda_bc={lambda_bc}")
        logger.info(f"{'='*60}")

    # Create dataloaders
    train_loader, val_loader, train_sampler, val_sampler = create_dataloaders(config)
    steps_per_epoch = len(train_loader)
    total_steps = max_epochs * steps_per_epoch

    # Create model with LoRA
    pretrained_path = config['model'].get('pretrained_path', None)
    model = PDELoRAModel(config, pretrained_path=pretrained_path)

    # KEY CHANGE: Unfreeze Encoder and Decoder
    enc_params, dec_params = unfreeze_encoder_decoder(model)
    if accelerator.is_main_process:
        logger.info(f"Unfroze Encoder: {enc_params:,} params")
        logger.info(f"Unfroze Decoder: {dec_params:,} params")

    # Convert to fp32 if needed
    if config['training'].get('mixed_precision', 'no') == 'no':
        model = model.float()

    # Log parameter summary
    log_param_summary(model, accelerator)

    # Optimizer: ALL trainable params (LoRA + Encoder + Decoder)
    trainable_params = get_all_trainable_params(model)
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
        run_name = f"burgers-lora-v2-unfrozen-r{lora_r}"
        accelerator.init_trackers(
            project_name=config['logging']['project'],
            config=config,
            init_kwargs={"wandb": {
                "entity": config['logging'].get('entity'),
                "name": run_name,
                "tags": ["burgers", "lora-v2", "unfrozen-enc-dec"],
            }}
        )

    save_dir = Path(config['logging']['save_dir'])
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Training state
    global_step = 0
    best_val_loss = float('inf')
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
            epoch_bc_loss = 0.0
            epoch_steps = 0

            for batch in train_loader:
                if early_stop:
                    break

                data = batch['data'].to(device=accelerator.device, dtype=torch.float32)
                channel_mask = batch['channel_mask'].to(device=accelerator.device)

                input_data = data[:, :-1]
                target = data[:, 1:]

                in_warmup = global_step < warmup_steps

                with accelerator.accumulate(model):
                    output = model(input_data)

                    pde_loss, loss_u, loss_v = compute_pde_loss(
                        output, input_data, batch, config, accelerator
                    )
                    rmse_loss = compute_rmse_loss(output.float(), target.float(), channel_mask)
                    bc_loss = compute_boundary_loss(output.float(), target.float(), channel_mask)

                    train_loss = lambda_pde * pde_loss + lambda_bc * bc_loss

                    accelerator.backward(train_loss)

                    if grad_clip > 0:
                        accelerator.clip_grad_norm_(trainable_params, grad_clip)

                    optimizer.step()
                    optimizer.zero_grad()

                scheduler.step()
                global_step += 1
                epoch_steps += 1

                pde_loss_reduced = accelerator.reduce(pde_loss.detach(), reduction='mean')
                rmse_loss_reduced = accelerator.reduce(rmse_loss.detach(), reduction='mean')
                bc_loss_reduced = accelerator.reduce(bc_loss.detach(), reduction='mean')

                epoch_pde_loss += pde_loss_reduced.item()
                epoch_rmse_loss += rmse_loss_reduced.item()
                epoch_bc_loss += bc_loss_reduced.item()

                phase_str = "[warmup]" if in_warmup else f"[E{epoch+1}]"
                progress.update(
                    train_task, advance=1,
                    description=f"{phase_str} PDE={pde_loss_reduced.item():.4f} RMSE={rmse_loss_reduced.item():.4f} BC={bc_loss_reduced.item():.4f}"
                )

                if global_step % log_interval == 0:
                    total_loss = lambda_pde * pde_loss_reduced + lambda_bc * bc_loss_reduced
                    accelerator.log({
                        'train/pde_loss': pde_loss_reduced.item(),
                        'train/rmse_loss': rmse_loss_reduced.item(),
                        'train/bc_loss': bc_loss_reduced.item(),
                        'train/total_loss': total_loss.item(),
                        'train/lr': scheduler.get_last_lr()[0],
                        'train/epoch': epoch + 1,
                    }, step=global_step)

                if global_step % eval_interval == 0:
                    accelerator.wait_for_everyone()
                    val_pde, val_rmse, val_bc = validate(model, val_loader, config, accelerator)

                    val_loss = lambda_pde * val_pde + lambda_bc * val_bc
                    accelerator.log({
                        'val/pde_loss': val_pde,
                        'val/rmse_loss': val_rmse,
                        'val/bc_loss': val_bc,
                        'val/total_loss': val_loss,
                    }, step=global_step)

                    if accelerator.is_main_process:
                        console.print(
                            f"\n[green]Step {global_step}/{total_steps} {phase_str}:[/green] "
                            f"val_pde={val_pde:.6f}, val_rmse={val_rmse:.6f}, val_bc={val_bc:.6f}"
                        )

                    if not in_warmup:
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            patience_counter = 0

                            if accelerator.is_main_process:
                                unwrapped_model = accelerator.unwrap_model(model)
                                save_full_checkpoint(
                                    model=unwrapped_model,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    global_step=global_step,
                                    metrics={'pde_loss': val_pde, 'rmse_loss': val_rmse, 'bc_loss': val_bc},
                                    save_path=str(save_dir / 'best_lora_v2.pt'),
                                    config=config,
                                )
                                console.print(f"[yellow]Saved best model[/yellow] (val_loss: {val_loss:.6f})")
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

            if epoch_steps > 0:
                avg_pde = epoch_pde_loss / epoch_steps
                avg_rmse = epoch_rmse_loss / epoch_steps
                avg_bc = epoch_bc_loss / epoch_steps
                accelerator.log({
                    'epoch/train_pde_loss': avg_pde,
                    'epoch/train_rmse_loss': avg_rmse,
                    'epoch/train_bc_loss': avg_bc,
                    'epoch': epoch + 1,
                }, step=global_step)

                if accelerator.is_main_process:
                    console.print(
                        f"\n[blue]Epoch {epoch+1}/{max_epochs} completed:[/blue] "
                        f"avg_pde={avg_pde:.6f}, avg_rmse={avg_rmse:.6f}, avg_bc={avg_bc:.6f}"
                    )

    accelerator.end_training()

    if accelerator.is_main_process:
        table = Table(title="Training Complete (LoRA V2 - Unfrozen)", show_header=False, border_style="green")
        table.add_row("Total Epochs", str(epoch + 1))
        table.add_row("Total Steps", str(global_step))
        table.add_row("Best Val Loss", f"{best_val_loss:.6f}")
        table.add_row("Early Stopped", "Yes" if early_stop else "No")
        table.add_row("Checkpoint", str(save_dir / "best_lora_v2.pt"))
        console.print(table)


if __name__ == "__main__":
    main()
