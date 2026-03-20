"""
Full Fine-Tuning for NS-PwC (all parameters, full-field RMSE).

Stage 1 of two-stage transfer:
  1. Full fine-tune pretrained dec_v3 on NS-PwC (this script)
  2. LoRA fine-tune the resulting checkpoint on NS-SVS

Loss: Normalized RMSE over all valid channels (same as pretrain).
No BC loss, no PDE loss — pure supervised full-field regression.

Channel mapping (18-channel):
- Channel 0 = Vx, Channel 1 = Vy (vector_dim=2)
- Channel 14 = scalar[11] = passive_tracer

Usage:
    torchrun --nproc_per_node=4 finetune/train_ns_pwc_fullft.py \
        --config configs/finetune_ns_pwc_fullft.yaml
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

from finetune.dataset_finetune import (
    FinetuneDataset, FinetuneSampler, finetune_collate_fn,
    create_finetune_dataloaders,
)
from pretrain.model_v3 import PDEModelV3

logger = logging.getLogger(__name__)


def compute_normalized_rmse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    channel_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute RMSE loss in normalized space with correct channel masking.

    Divides by number of valid channels, not total 18 channels.
    """
    mask = channel_mask
    for _ in range(pred.ndim - 2):
        mask = mask.unsqueeze(1)
    mask = mask.float()

    squared_error = (pred - target) ** 2
    masked_error = squared_error * mask

    total_error = masked_error.sum()
    spatial_volume = 1
    for i in range(1, pred.ndim - 1):
        spatial_volume *= pred.shape[i]
    num_valid = mask.sum() * spatial_volume
    mse = total_error / (num_valid + 1e-8)
    rmse = torch.sqrt(mse + 1e-8)
    return rmse


def compute_vrmse(
    output: torch.Tensor,
    target: torch.Tensor,
    channel_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute VRMSE and RMSE for valid channels (denormalized space).

    VRMSE = mean of per-channel sqrt(MSE_ch / Var_ch).
    This avoids scale mixing when channels have different magnitudes.

    Returns:
        vrmse, rmse
    """
    valid_ch = (torch.where(channel_mask[0] > 0)[0] if channel_mask.dim() > 1
                else torch.where(channel_mask > 0)[0])
    output_valid = output[..., valid_ch]
    target_valid = target[..., valid_ch]

    # Overall RMSE (unchanged)
    mse = torch.mean((output_valid - target_valid) ** 2)
    rmse = torch.sqrt(mse + 1e-8)

    # Per-channel VRMSE, then average
    n_ch = len(valid_ch)
    vrmse_sum = torch.tensor(0.0, device=output.device)
    for c in range(n_ch):
        pred_c = output[..., valid_ch[c]]
        gt_c = target[..., valid_ch[c]]
        mse_c = torch.mean((pred_c - gt_c) ** 2)
        var_c = torch.mean((gt_c - gt_c.mean()) ** 2)
        vrmse_sum = vrmse_sum + torch.sqrt(mse_c / (var_c + 1e-8))
    vrmse = vrmse_sum / n_ch

    return vrmse, rmse


def parse_args():
    parser = argparse.ArgumentParser(description="NS-PwC Full Fine-Tuning")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_lr_scheduler(optimizer, config: dict, total_steps: int):
    """Cosine LR with warmup."""
    from torch.optim.lr_scheduler import LambdaLR
    import math

    warmup_steps = config['training'].get('warmup_steps', 50)
    min_lr = config['training'].get('min_lr', 1e-7)
    base_lr = config['training']['learning_rate']
    min_lr_ratio = min_lr / base_lr

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def load_init_weights(model: PDEModelV3, init_path: str, is_main: bool = True) -> None:
    """Load pretrained weights into model."""
    ckpt = torch.load(init_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))

    cleaned = {}
    for k, v in state_dict.items():
        k = k.removeprefix('module.').removeprefix('_orig_mod.')
        cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)

    if is_main:
        logger.info(f"  Loaded: {len(cleaned) - len(unexpected)} keys")
        if missing:
            logger.info(f"  Missing (randomly init): {len(missing)} keys")
            for k in missing[:10]:
                logger.info(f"    - {k}")
        if unexpected:
            logger.info(f"  Unexpected (ignored): {len(unexpected)} keys")


@torch.no_grad()
def validate(
    model, val_loader, accelerator,
    t_input: int = 8,
):
    """Validate and return (norm_rmse, denorm_rmse, vrmse)."""
    accelerator.wait_for_everyone()
    model.eval()

    total_norm_rmse = torch.zeros(1, device=accelerator.device)
    total_rmse = torch.zeros(1, device=accelerator.device)
    total_vrmse = torch.zeros(1, device=accelerator.device)
    num_batches = torch.zeros(1, device=accelerator.device)

    for batch in val_loader:
        data = batch['data'].to(device=accelerator.device, dtype=torch.float32)
        channel_mask = batch['channel_mask'].to(device=accelerator.device)

        input_data = data[:, :t_input]
        target_data = data[:, 1:t_input + 1]

        output_norm, mean, std = model(input_data, return_normalized=True)
        target_norm = (target_data - mean) / std

        norm_rmse = compute_normalized_rmse_loss(
            output_norm.float(), target_norm.float(), channel_mask,
        )

        output = output_norm * std + mean
        vrmse, rmse = compute_vrmse(output, target_data, channel_mask)

        total_norm_rmse += norm_rmse.detach()
        total_rmse += rmse.detach()
        total_vrmse += vrmse.detach()
        num_batches += 1

    accelerator.wait_for_everyone()

    total_norm_rmse = accelerator.reduce(total_norm_rmse, reduction='sum')
    total_rmse = accelerator.reduce(total_rmse, reduction='sum')
    total_vrmse = accelerator.reduce(total_vrmse, reduction='sum')
    num_batches = accelerator.reduce(num_batches, reduction='sum')

    accelerator.wait_for_everyone()
    model.train()

    n = num_batches.item()
    avg_norm_rmse = (total_norm_rmse / num_batches).item() if n > 0 else 0
    avg_rmse = (total_rmse / num_batches).item() if n > 0 else 0
    avg_vrmse = (total_vrmse / num_batches).item() if n > 0 else 0

    return avg_norm_rmse, avg_rmse, avg_vrmse


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config['dataset']['seed'])

    max_epochs = config['training'].get('max_epochs', 30)
    warmup_steps = config['training'].get('warmup_steps', 50)
    log_interval = config['logging']['log_interval']
    grad_clip = config['training'].get('grad_clip', 1.0)
    eval_interval = config['training'].get('eval_interval', 100)
    early_stopping_patience = config['training'].get('early_stopping_patience', 30)
    t_input = config['dataset'].get('t_input', 8)

    init_from = config.get('checkpoint', {}).get('init_from')
    resume_from = args.resume or config.get('checkpoint', {}).get('resume_from')

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        mixed_precision=config['training'].get('mixed_precision', 'no'),
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1),
        log_with="wandb",
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_main_process:
        logger.info(f"{'='*60}")
        logger.info("NS-PwC Full Fine-Tuning (all params, full-field RMSE)")
        logger.info(f"{'='*60}")
        logger.info(f"Config: {args.config}")
        logger.info(f"LR: {config['training']['learning_rate']}")
        logger.info(f"Init from: {init_from}")
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
        clips_per_sample=config['dataset'].get('clips_per_sample'),
        vector_dim=config['dataset'].get('vector_dim', 2),
        val_time_interval=config['dataset'].get('val_time_interval', 4),
    )

    steps_per_epoch = len(train_sampler)
    total_steps = max_epochs * steps_per_epoch

    if accelerator.is_main_process:
        logger.info(f"Steps per epoch: {steps_per_epoch}")
        logger.info(f"Total steps: {total_steps}")

    # Create model (full PDEModelV3, no LoRA)
    model = PDEModelV3(config)

    # Load pretrained weights
    if init_from and not resume_from:
        if accelerator.is_main_process:
            logger.info(f"Loading init weights from: {init_from}")
        load_init_weights(model, init_from, is_main=accelerator.is_main_process)

    if config['training'].get('mixed_precision', 'no') == 'no':
        model = model.float()

    if accelerator.is_main_process:
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Parameters: {n_params:,} total, {n_trainable:,} trainable")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        betas=tuple(config['training'].get('betas', [0.9, 0.999])),
    )

    scheduler = get_lr_scheduler(optimizer, config, total_steps)
    model, optimizer = accelerator.prepare(model, optimizer)

    global_step = 0
    best_val_metric = float('inf')
    patience_counter = 0

    if resume_from:
        ckpt = torch.load(resume_from, map_location='cpu', weights_only=False)
        unwrapped = accelerator.unwrap_model(model)
        state_dict = ckpt.get('model_state_dict', ckpt)
        cleaned = {}
        for k, v in state_dict.items():
            k = k.removeprefix('module.').removeprefix('_orig_mod.')
            cleaned[k] = v
        unwrapped.load_state_dict(cleaned, strict=True)
        global_step = ckpt.get('global_step', 0)
        best_val_metric = ckpt.get('best_val_metric', float('inf'))
        if accelerator.is_main_process:
            logger.info(f"Resumed from step {global_step}, best_val_metric={best_val_metric:.6f}")

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=config['logging']['project'],
            config=config,
            init_kwargs={"wandb": {
                "entity": config['logging'].get('entity'),
                "name": config.get('model_name', 'ns_pwc_fullft'),
                "tags": ["ns_pwc", "fullft", "rmse"],
            }},
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
        train_task = progress.add_task("Full FT", total=total_steps, completed=global_step)

        for epoch in range(start_epoch, max_epochs):
            if early_stop:
                break

            train_sampler.set_epoch(epoch)
            epoch_loss = 0.0
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
                    output_norm, mean, std = model(input_data, return_normalized=True)
                    target_norm = (target_data - mean) / std

                    loss = compute_normalized_rmse_loss(
                        output_norm.float(), target_norm.float(), channel_mask,
                    )

                    accelerator.backward(loss)
                    if grad_clip > 0:
                        accelerator.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()

                scheduler.step()
                global_step += 1
                epoch_steps += 1

                loss_reduced = accelerator.reduce(loss.detach(), reduction='mean')
                epoch_loss += loss_reduced.item()

                phase_str = "[warmup]" if in_warmup else f"[E{epoch+1}]"
                progress.update(
                    train_task, advance=1,
                    description=f"{phase_str} loss={loss_reduced.item():.4f}",
                )

                if global_step % log_interval == 0:
                    accelerator.log({
                        'train/loss': loss_reduced.item(),
                        'train/lr': scheduler.get_last_lr()[0],
                        'train/epoch': epoch + 1,
                    }, step=global_step)

                if global_step % eval_interval == 0:
                    val_norm_rmse, val_rmse, val_vrmse = validate(
                        model, val_loader, accelerator, t_input,
                    )

                    accelerator.log({
                        'val/norm_rmse': val_norm_rmse,
                        'val/rmse': val_rmse,
                        'val/vrmse': val_vrmse,
                    }, step=global_step)

                    if accelerator.is_main_process:
                        console.print(
                            f"\n[green]Step {global_step}/{total_steps} {phase_str}:[/green] "
                            f"norm_rmse={val_norm_rmse:.6f}, rmse={val_rmse:.6f}, vrmse={val_vrmse:.6f}"
                        )

                    if not in_warmup:
                        # Save by normalized RMSE (same metric as pretrain)
                        if val_norm_rmse < best_val_metric:
                            best_val_metric = val_norm_rmse
                            patience_counter = 0

                            accelerator.wait_for_everyone()
                            state_dict = accelerator.get_state_dict(model, unwrap=True)
                            if accelerator.is_main_process:
                                checkpoint = {
                                    'global_step': global_step,
                                    'model_state_dict': state_dict,
                                    'best_val_metric': best_val_metric,
                                    'val_norm_rmse': val_norm_rmse,
                                    'val_rmse': val_rmse,
                                    'val_vrmse': val_vrmse,
                                    'config': config,
                                }
                                torch.save(checkpoint, save_dir / 'best_tf.pt')
                                console.print(
                                    f"[yellow]Saved best model[/yellow] "
                                    f"(norm_rmse={val_norm_rmse:.6f})"
                                )
                            accelerator.wait_for_everyone()
                        else:
                            patience_counter += 1
                            if accelerator.is_main_process:
                                console.print(
                                    f"[dim]Patience: {patience_counter}/{early_stopping_patience}[/dim]"
                                )
                            if patience_counter >= early_stopping_patience:
                                if accelerator.is_main_process:
                                    console.print("[red]Early stopping![/red]")
                                early_stop = True
                                break
                    else:
                        if accelerator.is_main_process:
                            console.print("[dim](warmup - no saving)[/dim]")

                    model.train()

            if epoch_steps > 0 and accelerator.is_main_process:
                avg_loss = epoch_loss / epoch_steps
                console.print(
                    f"\n[blue]Epoch {epoch+1}/{max_epochs}:[/blue] avg_loss={avg_loss:.6f}"
                )

    accelerator.end_training()

    if accelerator.is_main_process:
        table = Table(title="Training Complete", show_header=False, border_style="green")
        table.add_row("Total Epochs", str(epoch + 1))
        table.add_row("Total Steps", str(global_step))
        table.add_row("Best Val Metric", f"{best_val_metric:.6f}")
        table.add_row("Early Stopped", "Yes" if early_stop else "No")
        table.add_row("Checkpoint", str(save_dir / "best_tf.pt"))
        console.print(table)


if __name__ == "__main__":
    main()
