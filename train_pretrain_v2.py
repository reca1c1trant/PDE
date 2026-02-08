"""
V2 Pretraining for PDE Foundation Model with Neighborhood Attention.

Features:
- Multi-dataset training (diffusion-reaction, 2D_CFD, SWE)
- 18 channels (3 vector + 15 scalar) with channel_mask
- RMSE loss in normalized space
- Two-phase training:
    - Phase 1 (teacher forcing): eval every epoch
    - Phase 2 (multi-step loss): eval every N steps
- Support for checkpoint resume

Usage:
    # Single GPU (debug)
    python train_pretrain_v2.py --config configs/pretrain_v2.yaml

    # 8 GPUs distributed
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train_pretrain_v2.py --config configs/pretrain_v2.yaml
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

from dataset_pretrain import (
    PretrainDataset, PretrainSampler, DatasetConfig,
    pretrain_collate_fn, create_pretrain_dataloaders,
    NUM_VECTOR_CHANNELS, NUM_SCALAR_CHANNELS, TOTAL_CHANNELS
)
from model_v2 import PDEModelV2

logger = logging.getLogger(__name__)


def compute_normalized_rmse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    channel_mask: torch.Tensor
) -> torch.Tensor:
    """Compute RMSE loss in normalized space with channel masking."""
    mask = channel_mask[:, None, None, None, :].float()
    squared_error = (pred - target) ** 2
    masked_error = squared_error * mask
    total_error = masked_error.sum()
    num_valid = mask.sum() * pred.shape[1] * pred.shape[2] * pred.shape[3]
    mse = total_error / (num_valid + 1e-8)
    rmse = torch.sqrt(mse + 1e-8)
    return rmse


def compute_multi_step_loss(
    model: torch.nn.Module,
    data: torch.Tensor,
    channel_mask: torch.Tensor,
    num_steps: int = 3,
    lambda_decay: float = 0.5,
    t_input: int = 8,
) -> torch.Tensor:
    """
    Compute multi-step loss for AR training.

    Args:
        model: The model
        data: [B, T_total, H, W, C] where T_total >= t_input + num_steps
        channel_mask: [B, C]
        num_steps: Number of rollout steps (default 3)
        lambda_decay: Decay factor for loss weights
        t_input: Number of input timesteps (fixed at 8 for V2)

    Returns:
        total_loss: Weighted sum of losses

    Example with t_input=8, num_steps=3:
        Step 0: input=data[0:8], target=data[1:9], weight=1.0
        Step 1: input=data[1:8]+pred[-1], target=data[2:10], weight=0.5
        Step 2: input=data[2:9]+pred[-1], target=data[3:11], weight=0.25
        Total data needed: 11 timesteps
    """
    B, T_total, H, W, C = data.shape

    # Validate data has enough timesteps
    required_T = t_input + num_steps  # 8 + 3 = 11
    assert T_total >= required_T, \
        f"Data has {T_total} timesteps, but need {required_T} for AR (t_input={t_input}, num_steps={num_steps})"

    total_loss = torch.tensor(0.0, device=data.device)
    weight = 1.0

    # Step 0: Normal forward (teacher forcing)
    input_data = data[:, :t_input]  # [B, 8, H, W, C]
    target_data = data[:, 1:t_input + 1]  # [B, 8, H, W, C]

    output_norm, mean, std = model(input_data, return_normalized=True)
    target_norm = (target_data - mean) / std

    loss_0 = compute_normalized_rmse_loss(output_norm.float(), target_norm.float(), channel_mask)
    total_loss = total_loss + weight * loss_0

    # Denormalize output for next step input
    pred_denorm = output_norm * std + mean

    # Subsequent steps: use predictions as input
    for step in range(1, num_steps):
        weight = weight * lambda_decay

        # Build new input: GT[step : step+t_input-1] + pred[-1]
        # e.g., step=1: GT[1:8] (7 frames) + pred[-1] (1 frame) = 8 frames
        gt_part = data[:, step:step + t_input - 1]  # [B, t_input-1, H, W, C]
        pred_part = pred_denorm[:, -1:]  # [B, 1, H, W, C]
        new_input = torch.cat([gt_part, pred_part], dim=1)  # [B, t_input, H, W, C]

        # Target: data[step+1 : step+1+t_input]
        # e.g., step=1: target = data[2:10]
        target_data = data[:, step + 1:step + 1 + t_input]  # [B, t_input, H, W, C]

        output_norm, mean, std = model(new_input, return_normalized=True)
        target_norm = (target_data - mean) / std

        loss_step = compute_normalized_rmse_loss(output_norm.float(), target_norm.float(), channel_mask)
        total_loss = total_loss + weight * loss_step

        # Update pred for next step
        pred_denorm = output_norm * std + mean

    return total_loss


def parse_args():
    parser = argparse.ArgumentParser(description="PDE Foundation Model V2 Pretraining")
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

    warmup_steps = config['training'].get('warmup_steps', 1000)
    min_lr_ratio = config['training'].get('min_lr', 1e-6) / config['training']['learning_rate']

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def load_checkpoint(model, checkpoint_path: str, config: dict, accelerator):
    """Load checkpoint (model weights only)."""
    if not checkpoint_path or not Path(checkpoint_path).exists():
        if accelerator.is_main_process:
            logger.info("No checkpoint to load, starting from scratch")
        return 0, float('inf')

    if accelerator.is_main_process:
        logger.info(f"Loading checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location='cpu')

    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    unwrapped_model = accelerator.unwrap_model(model)
    missing, unexpected = unwrapped_model.load_state_dict(state_dict, strict=False)

    if accelerator.is_main_process:
        if missing:
            logger.warning(f"Missing keys: {missing[:5]}..." if len(missing) > 5 else f"Missing keys: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected[:5]}..." if len(unexpected) > 5 else f"Unexpected keys: {unexpected}")
        logger.info("Loaded model weights. Optimizer and scheduler will start fresh.")

    return 0, float('inf')


@torch.no_grad()
def validate(model, val_loader, accelerator) -> tuple[float, dict]:
    """Validate and return RMSE."""
    accelerator.wait_for_everyone()
    model.eval()

    total_rmse = torch.zeros(1, device=accelerator.device)
    num_batches = torch.zeros(1, device=accelerator.device)

    all_dataset_names = ['diffusion_reaction', '2d_cfd', 'swe']
    dataset_rmse = {name: torch.zeros(1, device=accelerator.device) for name in all_dataset_names}
    dataset_count = {name: torch.zeros(1, device=accelerator.device) for name in all_dataset_names}

    for batch in val_loader:
        data = batch['data'].to(device=accelerator.device, dtype=torch.float32)
        channel_mask = batch['channel_mask'].to(device=accelerator.device)
        dataset_names = batch['dataset_names']

        # Use only first 8 timesteps for input (V2 uses T=8)
        T_input = min(8, data.shape[1] - 1)
        input_data = data[:, :T_input]
        target_data = data[:, 1:T_input + 1]

        output_norm, mean, std = model(input_data, return_normalized=True)
        target_norm = (target_data - mean) / std

        rmse = compute_normalized_rmse_loss(output_norm.float(), target_norm.float(), channel_mask)

        total_rmse += rmse.detach()
        num_batches += 1

        ds_name = dataset_names[0]
        if ds_name in dataset_rmse:
            dataset_rmse[ds_name] += rmse.detach()
            dataset_count[ds_name] += 1

    accelerator.wait_for_everyone()

    total_rmse = accelerator.reduce(total_rmse, reduction='sum')
    num_batches = accelerator.reduce(num_batches, reduction='sum')

    for ds_name in all_dataset_names:
        dataset_rmse[ds_name] = accelerator.reduce(dataset_rmse[ds_name], reduction='sum')
        dataset_count[ds_name] = accelerator.reduce(dataset_count[ds_name], reduction='sum')

    accelerator.wait_for_everyone()
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
    """Save checkpoint."""
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

    model_name = config.get('model_name', 'pde_v2')
    hidden_dim = config['model']['hidden_dim']
    num_layers = config['model']['num_layers']
    warmup_steps = config['training'].get('warmup_steps', 1000)
    max_epochs = config['training']['max_epochs']
    teacher_forcing_ratio = config['training'].get('teacher_forcing_ratio', 0.5)
    eval_interval_tf = config['training'].get('eval_interval_tf', 0)  # 0 = every epoch
    eval_interval_ar = config['training'].get('eval_interval_ar', 200)
    t_input = config['dataset'].get('t_input', 8)  # Input timesteps (default 8)

    # Multi-step loss config (needed early for temporal_length calculation)
    use_multi_step = config['training'].get('multi_step_loss', {}).get('enabled', True)
    multi_step_n = config['training'].get('multi_step_loss', {}).get('num_steps', 3)
    multi_step_lambda = config['training'].get('multi_step_loss', {}).get('lambda', 0.5)

    resume_path = args.resume or config.get('checkpoint', {}).get('resume_from')

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)

    accelerator = Accelerator(
        mixed_precision=config['training'].get('mixed_precision', 'no'),
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 2),
        log_with="wandb",
        kwargs_handlers=[ddp_kwargs]
    )

    log_interval = config['logging']['log_interval']
    early_stopping_patience = config['training'].get('early_stopping_patience', 30)
    grad_clip = config['training'].get('grad_clip', 1.0)

    # Create dataloaders
    data_dir = config['dataset']['path']
    batch_size = config['dataloader']['batch_size']
    num_workers = config['dataloader']['num_workers']
    seed = config['dataset']['seed']
    dataset_overrides = config.get('dataset', {}).get('overrides', {})

    # Compute temporal_length = t_input + num_steps for AR
    temporal_length = t_input + multi_step_n  # e.g., 8 + 3 = 11

    train_loader, val_loader, train_sampler, val_sampler = create_pretrain_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=config['dataloader']['pin_memory'],
        seed=seed,
        dataset_overrides=dataset_overrides,
        temporal_length=temporal_length,
    )

    steps_per_epoch = len(train_sampler)
    total_steps = max_epochs * steps_per_epoch
    tf_steps = int(total_steps * teacher_forcing_ratio)  # Teacher forcing phase

    if accelerator.is_main_process:
        logger.info(f"{'='*60}")
        logger.info(f"PDE Foundation Model V2 Pretraining")
        logger.info(f"{'='*60}")
        logger.info(f"Config: {args.config}")
        logger.info(f"Model: {model_name} (hidden={hidden_dim}, layers={num_layers})")
        logger.info(f"Max Epochs: {max_epochs}")
        logger.info(f"Steps per Epoch: {steps_per_epoch}")
        logger.info(f"Total Steps: {total_steps}")
        logger.info(f"Teacher Forcing Steps: {tf_steps} ({teacher_forcing_ratio*100:.0f}%)")
        logger.info(f"TF Phase Eval Interval: {eval_interval_tf if eval_interval_tf > 0 else 'every epoch'}")
        logger.info(f"AR Phase Eval Interval: {eval_interval_ar}")
        logger.info(f"T_input: {t_input}, Temporal Length: {temporal_length}")
        logger.info(f"Resume from: {resume_path or 'scratch'}")
        logger.info(f"{'='*60}")

    # Create model
    model = PDEModelV2(config)

    if accelerator.is_main_process:
        logger.info(f"Model parameters: {model.get_num_params():,}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        betas=tuple(config['training'].get('betas', [0.9, 0.95]))
    )

    scheduler = get_lr_scheduler(optimizer, config, total_steps)

    model, optimizer = accelerator.prepare(model, optimizer)

    global_step, best_val_rmse = load_checkpoint(
        model, resume_path, config, accelerator
    )

    if accelerator.is_main_process:
        run_name = f"v2-{model_name}-h{hidden_dim}-L{num_layers}"
        accelerator.init_trackers(
            project_name=config['logging']['project'],
            config=config,
            init_kwargs={"wandb": {
                "entity": config['logging'].get('entity'),
                "name": run_name,
                "tags": ["v2", "NA", f"h{hidden_dim}", f"L{num_layers}"],
            }}
        )

    save_dir = Path(config['logging']['save_dir'])
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)

    patience_counter = 0
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
        train_task = progress.add_task("V2 Pretraining", total=total_steps, completed=global_step)

        for epoch in range(start_epoch, max_epochs):
            if early_stop:
                break

            train_sampler.set_epoch(epoch)

            for batch in train_loader:
                data = batch['data'].to(device=accelerator.device, dtype=torch.float32)
                channel_mask = batch['channel_mask'].to(device=accelerator.device)

                in_warmup = global_step < warmup_steps
                in_tf_phase = global_step < tf_steps

                with accelerator.accumulate(model):
                    if in_tf_phase or not use_multi_step:
                        # Teacher forcing: single step loss
                        input_data = data[:, :t_input]  # [B, 8, H, W, C]
                        target_data = data[:, 1:t_input + 1]  # [B, 8, H, W, C]

                        output_norm, mean, std = model(input_data, return_normalized=True)
                        target_norm = (target_data - mean) / std

                        loss = compute_normalized_rmse_loss(
                            output_norm.float(), target_norm.float(), channel_mask
                        )
                    else:
                        # AR phase: multi-step loss
                        loss = compute_multi_step_loss(
                            model, data, channel_mask,
                            num_steps=multi_step_n,
                            lambda_decay=multi_step_lambda,
                            t_input=t_input,
                        )

                    accelerator.backward(loss)

                    if grad_clip:
                        accelerator.clip_grad_norm_(model.parameters(), grad_clip)

                    optimizer.step()
                    optimizer.zero_grad()

                scheduler.step()
                global_step += 1

                loss_reduced = accelerator.reduce(loss.detach(), reduction='mean')

                phase_str = "[warmup]" if in_warmup else ("[TF]" if in_tf_phase else "[AR]")
                progress.update(
                    train_task, advance=1,
                    description=f"{phase_str} E{epoch+1} loss={loss_reduced.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}"
                )

                if global_step % log_interval == 0:
                    accelerator.log({
                        'train/loss': loss_reduced.item(),
                        'train/lr': scheduler.get_last_lr()[0],
                        'train/epoch': epoch + 1,
                        'train/phase': 'tf' if in_tf_phase else 'ar',
                    }, step=global_step)

                # Validation logic
                should_validate = False

                if in_tf_phase:
                    # In TF phase: validate based on eval_interval_tf
                    if eval_interval_tf > 0:
                        # Validate every eval_interval_tf steps
                        if global_step % eval_interval_tf == 0:
                            should_validate = True
                    else:
                        # Validate every epoch (at end of epoch)
                        if global_step % steps_per_epoch == 0:
                            should_validate = True
                else:
                    # In AR phase: validate every eval_interval_ar steps
                    if global_step % eval_interval_ar == 0:
                        should_validate = True

                if should_validate:
                    val_rmse, per_dataset = validate(model, val_loader, accelerator)

                    log_dict = {'val/rmse': val_rmse}
                    for ds_name, ds_rmse in per_dataset.items():
                        log_dict[f'val/rmse_{ds_name}'] = ds_rmse
                    accelerator.log(log_dict, step=global_step)

                    if accelerator.is_main_process:
                        ds_str = ", ".join([f"{k}={v:.4f}" for k, v in per_dataset.items()])
                        console.print(f"\n[green]Step {global_step}/{total_steps} {phase_str}:[/green] val_rmse={val_rmse:.6f}")
                        if ds_str:
                            console.print(f"[dim]  Per-dataset: {ds_str}[/dim]")

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
        table = Table(title="V2 Pretraining Complete", show_header=False, border_style="green")
        table.add_row("Total Epochs", str(epoch + 1))
        table.add_row("Total Steps", str(global_step))
        table.add_row("Best Val RMSE", f"{best_val_rmse:.6f}")
        table.add_row("Checkpoint", str(save_dir / "best.pt"))
        console.print(table)


if __name__ == "__main__":
    main()
