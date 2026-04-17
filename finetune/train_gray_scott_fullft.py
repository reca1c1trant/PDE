"""
Full-Param Finetuning for Gray-Scott Reaction-Diffusion using PDEModelV3.

Loss: BC loss + PDE loss (NO RMSE)
- BC loss: RMSE on boundary pixels (supervised anchor for periodic domain)
- PDE loss: Gray-Scott equation residual (dA/dt, dB/dt), per-equation normalized

Channel mapping (18-channel):
- Channel 5 = scalar[2] = concentration_u (A, activator)
- Channel 6 = scalar[3] = concentration_v (B, inhibitor)
- No velocity fields (vector_dim=0)

Metrics:
- RMSE: full-field monitoring (not in training loss)
- save_metric = save_rmse_weight * val_rmse + save_pde_weight * val_pde

Usage:
    torchrun --nproc_per_node=8 finetune/train_gray_scott_fullft.py --config configs/finetune_gray_scott_v3.yaml
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
    create_finetune_dataloaders, TOTAL_CHANNELS
)
from pretrain.model_v3 import PDEModelV3
from finetune.pde_loss_verified import GrayScottPDELoss

logger = logging.getLogger(__name__)


# Gray-Scott channel indices in 18-channel layout
CH_A = 5  # scalar[2] = concentration_u (activator)
CH_B = 6  # scalar[3] = concentration_v (inhibitor)


def parse_args():
    parser = argparse.ArgumentParser(description="Full-Param Finetuning for Gray-Scott (V3 Model)")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
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

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def compute_boundary_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    channel_mask: torch.Tensor,
    bc_width: int = 1,
) -> torch.Tensor:
    """
    Compute boundary RMSE on border strips of width `bc_width`.

    bc_width=1: single-pixel edges (original behavior).
    bc_width=16: full boundary patches (patch_size-wide strips).

    For periodic domains this serves as a supervised anchor
    (not a physical BC, but provides ground truth signal).
    """
    if channel_mask.dim() == 1:
        valid_ch = torch.where(channel_mask > 0)[0]
    else:
        valid_ch = torch.where(channel_mask[0] > 0)[0]

    w = bc_width
    # Top/bottom strips: full width, avoid double-counting corners with left/right
    top_pred = pred[:, :, :w, :, :][:, :, :, :, valid_ch]
    top_target = target[:, :, :w, :, :][:, :, :, :, valid_ch]
    bottom_pred = pred[:, :, -w:, :, :][:, :, :, :, valid_ch]
    bottom_target = target[:, :, -w:, :, :][:, :, :, :, valid_ch]
    # Left/right strips: exclude corners already in top/bottom
    left_pred = pred[:, :, w:-w, :w, :][:, :, :, :, valid_ch]
    left_target = target[:, :, w:-w, :w, :][:, :, :, :, valid_ch]
    right_pred = pred[:, :, w:-w, -w:, :][:, :, :, :, valid_ch]
    right_target = target[:, :, w:-w, -w:, :][:, :, :, :, valid_ch]

    bc_pred = torch.cat([
        top_pred.reshape(-1), bottom_pred.reshape(-1),
        left_pred.reshape(-1), right_pred.reshape(-1),
    ])
    bc_target = torch.cat([
        top_target.reshape(-1), bottom_target.reshape(-1),
        left_target.reshape(-1), right_target.reshape(-1),
    ])

    mse = torch.mean((bc_pred - bc_target) ** 2)
    return torch.sqrt(mse + 1e-8)


def compute_pde_loss(
    output: torch.Tensor,
    input_data: torch.Tensor,
    pde_loss_fn: GrayScottPDELoss,
) -> tuple:
    """
    Compute Gray-Scott PDE residual loss.

    Prepends t0 frame from input for time derivatives.

    Args:
        output: Model output [B, T_out, H, W, 18] (denormalized)
        input_data: Model input [B, T_in, H, W, 18]
        pde_loss_fn: GrayScottPDELoss instance

    Returns:
        total_loss, losses_dict
    """
    with torch.autocast(device_type='cuda', enabled=False):
        # Prepend t0 frame for time derivative
        t0_A = input_data[:, 0:1, :, :, CH_A].float()
        t0_B = input_data[:, 0:1, :, :, CH_B].float()

        out_A = output[:, :, :, :, CH_A].float()
        out_B = output[:, :, :, :, CH_B].float()

        # [B, T_out+1, H, W] for 2nd-order central time difference
        A = torch.cat([t0_A, out_A], dim=1)
        B = torch.cat([t0_B, out_B], dim=1)

        total_loss, losses = pde_loss_fn(A, B)

    return total_loss, losses


def compute_vrmse(
    output: torch.Tensor,
    target: torch.Tensor,
    channel_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute VRMSE and RMSE for valid channels.

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


@torch.no_grad()
def validate(
    model, val_loader, accelerator,
    pde_loss_fn: GrayScottPDELoss,
    t_input: int = 8,
    bc_width: int = 1,
):
    """Validate and return (bc_loss, pde_loss, rmse, vrmse)."""
    accelerator.wait_for_everyone()
    model.eval()

    total_bc = torch.zeros(1, device=accelerator.device)
    total_pde = torch.zeros(1, device=accelerator.device)
    total_rmse = torch.zeros(1, device=accelerator.device)
    total_vrmse = torch.zeros(1, device=accelerator.device)
    num_batches = torch.zeros(1, device=accelerator.device)

    for batch in val_loader:
        data = batch['data'].to(device=accelerator.device, dtype=torch.float32)
        channel_mask = batch['channel_mask'].to(device=accelerator.device)

        input_data = data[:, :t_input]
        target_data = data[:, 1:t_input + 1]

        output_norm, mean, std = model(input_data, return_normalized=True)
        output = output_norm * std + mean

        bc_loss = compute_boundary_loss(output, target_data, channel_mask, bc_width=bc_width)
        pde_loss, _ = compute_pde_loss(output, input_data, pde_loss_fn)
        vrmse, rmse = compute_vrmse(output, target_data, channel_mask)

        total_bc += bc_loss.detach()
        total_pde += pde_loss.detach()
        total_rmse += rmse.detach()
        total_vrmse += vrmse.detach()
        num_batches += 1

    accelerator.wait_for_everyone()

    total_bc = accelerator.reduce(total_bc, reduction='sum')
    total_pde = accelerator.reduce(total_pde, reduction='sum')
    total_rmse = accelerator.reduce(total_rmse, reduction='sum')
    total_vrmse = accelerator.reduce(total_vrmse, reduction='sum')
    num_batches = accelerator.reduce(num_batches, reduction='sum')

    accelerator.wait_for_everyone()
    model.train()

    n = num_batches.item()
    avg_bc = (total_bc / num_batches).item() if n > 0 else 0
    avg_pde = (total_pde / num_batches).item() if n > 0 else 0
    avg_rmse = (total_rmse / num_batches).item() if n > 0 else 0
    avg_vrmse = (total_vrmse / num_batches).item() if n > 0 else 0

    return avg_bc, avg_pde, avg_rmse, avg_vrmse


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config['dataset']['seed'])

    max_epochs = config['training'].get('max_epochs', 20)
    warmup_steps = config['training'].get('warmup_steps', 100)
    log_interval = config['logging']['log_interval']
    lambda_bc = config['training'].get('lambda_bc', 10000.0)
    lambda_pde = config['training'].get('lambda_pde', 1.0)
    grad_clip = config['training'].get('grad_clip', 1.0)
    eval_interval = config['training'].get('eval_interval', 100)
    early_stopping_patience = config['training'].get('early_stopping_patience', 15)
    save_vrmse_weight = config['training'].get('save_vrmse_weight', 100.0)
    save_pde_weight = config['training'].get('save_pde_weight', 1.0)
    bc_width = config['training'].get('bc_width', 1)
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
        logger.info("Gray-Scott Full-Param Finetuning (V3 Model)")
        logger.info(f"{'='*60}")
        logger.info(f"Config: {args.config}")
        logger.info(f"Loss: BC (boundary w={bc_width}) + PDE (A_eq + B_eq)")
        logger.info(f"Lambda BC: {lambda_bc}, PDE: {lambda_pde}")
        logger.info(f"Save: {save_vrmse_weight}*vrmse + {save_pde_weight}*pde")
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
        vector_dim=config['dataset'].get('vector_dim', 0),
        val_time_interval=config['dataset'].get('val_time_interval', 8),
    )

    steps_per_epoch = len(train_sampler)
    total_steps = max_epochs * steps_per_epoch

    if accelerator.is_main_process:
        logger.info(f"Steps per epoch: {steps_per_epoch}")
        logger.info(f"Total steps: {total_steps}")

    model = PDEModelV3(config)
    pretrained_path = config['model'].get('pretrained_path')
    if pretrained_path:
        ckpt = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state_dict, strict=False)
        if accelerator.is_main_process:
            logger.info(f'Loaded pretrained weights from {pretrained_path}')

    if config['training'].get('mixed_precision', 'no') == 'no':
        model = model.float()

    trainable_params = list(model.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        betas=tuple(config['training'].get('betas', [0.9, 0.999]))
    )

    scheduler = get_lr_scheduler(optimizer, config, total_steps)
    model, optimizer = accelerator.prepare(model, optimizer)

    # Create PDE loss function with per-equation normalization
    physics = config.get('physics', {})
    eq_scales = physics.get('eq_scales', None)
    eq_weights = physics.get('eq_weights', None)

    # Load per-timestep scales if specified
    eq_scales_per_t_path = physics.get('eq_scales_per_t_path', None)
    eq_scales_per_t = None
    if eq_scales_per_t_path and Path(eq_scales_per_t_path).exists():
        eq_scales_per_t = torch.load(eq_scales_per_t_path, map_location='cpu', weights_only=False)
        if accelerator.is_main_process:
            logger.info(f"Loaded per-timestep eq_scales from {eq_scales_per_t_path}")
            for k, v in eq_scales_per_t.items():
                logger.info(f"  {k}: shape={v.shape}, min={v.min():.4e}, max={v.max():.4e}")

    pde_loss_fn = GrayScottPDELoss(
        nx=physics.get('nx', 128),
        ny=physics.get('ny', 128),
        dx=physics.get('dx', 2.0 / 128),
        dy=physics.get('dy', 2.0 / 128),
        dt=physics.get('dt', 10.0),
        F=physics.get('F', 0.098),
        k=physics.get('k', 0.057),
        D_A=physics.get('D_A', 1.81e-5),
        D_B=physics.get('D_B', 1.39e-5),
        eq_scales=eq_scales,
        eq_weights=eq_weights,
        eq_scales_per_t=eq_scales_per_t,
    ).to(accelerator.device)

    if accelerator.is_main_process:
        logger.info(f"PDE eq_scales: {pde_loss_fn.eq_scales}")
        logger.info(f"PDE eq_weights: {pde_loss_fn.eq_weights}")
        logger.info(f"PDE per-timestep scales: {pde_loss_fn.use_per_t_scales}")

    global_step = 0
    best_val_loss = float('inf')
    patience_counter = 0

    if accelerator.is_main_process:
        run_name = "gray_scott-fullft-v3"
        accelerator.init_trackers(
            project_name=config['logging']['project'],
            config=config,
            init_kwargs={"wandb": {
                "entity": config['logging'].get('entity'),
                "name": run_name,
                "tags": ["gray_scott", "fullft", "v3", "periodic", "BC+PDE", "normalized"],
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
        train_task = progress.add_task("Full-FT Training", total=total_steps, completed=global_step)

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
                    output = output_norm * std + mean

                    # BC loss (boundary patches as supervised anchor)
                    bc_loss = compute_boundary_loss(output, target_data, channel_mask, bc_width=bc_width)

                    # PDE loss (A_eq + B_eq, per-equation normalized)
                    pde_loss, losses = compute_pde_loss(output, input_data, pde_loss_fn)

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
                    description=(
                        f"{phase_str} BC={bc_reduced.item():.4f} "
                        f"PDE={pde_reduced.item():.4f}"
                    )
                )

                if global_step % log_interval == 0:
                    accelerator.log({
                        'train/bc_loss': bc_reduced.item(),
                        'train/pde_loss': pde_reduced.item(),
                        'train/total_loss': (lambda_bc * bc_reduced.item()
                                             + lambda_pde * pde_reduced.item()),
                        'train/lr': scheduler.get_last_lr()[0],
                        'train/epoch': epoch + 1,
                    }, step=global_step)

                if global_step % eval_interval == 0:
                    val_bc, val_pde, val_rmse, val_vrmse = validate(
                        model, val_loader, accelerator, pde_loss_fn, t_input,
                        bc_width=bc_width,
                    )

                    accelerator.log({
                        'val/bc_loss': val_bc,
                        'val/pde_loss': val_pde,
                        'val/rmse': val_rmse,
                        'val/vrmse': val_vrmse,
                    }, step=global_step)

                    if accelerator.is_main_process:
                        console.print(
                            f"\n[green]Step {global_step}/{total_steps} {phase_str}:[/green] "
                            f"vrmse={val_vrmse:.6f}, rmse={val_rmse:.6f}, "
                            f"pde={val_pde:.6f}, bc={val_bc:.6f}"
                        )

                    if not in_warmup:
                        save_metric = save_vrmse_weight * val_vrmse + save_pde_weight * val_pde
                        if save_metric < best_val_loss:
                            best_val_loss = save_metric
                            patience_counter = 0
                            if accelerator.is_main_process:
                                torch.save({
                                    'global_step': global_step,
                                    'model_state_dict': accelerator.get_state_dict(model, unwrap=True),
                                    'best_val_loss': best_val_loss,
                                    'patience_counter': patience_counter,
                                    'config': config,
                                }, str(save_dir / 'best_scratch.pt'))
                                console.print(
                                    f"[yellow]Saved best model[/yellow] "
                                    f"(save={save_metric:.4f}: "
                                    f"{save_vrmse_weight}*{val_vrmse:.4f} + "
                                    f"{save_pde_weight}*{val_pde:.4f})"
                                )
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
        table.add_row("Checkpoint", str(save_dir / "best_scratch.pt"))
        console.print(table)


if __name__ == "__main__":
    main()
