"""
LoRA Finetuning for 3D Advection using PDEModelV3.

Loss: BC loss + PDE loss (NO RMSE)
- BC loss: RMSE on boundary pixels (supervised anchor for periodic domain)
- PDE loss: u_t + a*u_x + b*u_y + c*u_z = 0, per-sample (a,b,c) from batch

Channel mapping (18-channel):
- Channel 14 = scalar[11] = passive_tracer u
- vector_dim=0

BCs: Periodic in x, y, z
Per-sample (a, b, c): from dataset params_a, params_b, params_c

Metrics:
- VRMSE: per-channel variance-normalized RMSE
- save_metric = save_vrmse_weight * val_vrmse + save_pde_weight * val_pde

Usage:
    torchrun --nproc_per_node=4 finetune/train_advection_3d_lora_v3.py \
        --config configs/finetune_advection_3d_v3_rescaled_norm.yaml
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
from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
import datetime
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
from finetune.model_lora_v3 import PDELoRAModelV3, save_lora_checkpoint, load_lora_checkpoint
from finetune.pde_loss_verified import Advection3DPDELoss

logger = logging.getLogger(__name__)

# 3D Advection channel index: u -> scalar[11] = channel 14 (3+11)
CH_U = 14


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA Finetuning for 3D Advection (V3 Model)")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume')
    parser.add_argument('--init_weights', type=str, default=None,
                        help='Load LoRA weights only (no optimizer/scheduler/step)')
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
    Compute boundary RMSE on border strips of width `bc_width` for 3D data.

    3D data layout: [B, T, X, Y, Z, 18]
    For periodic domains: compare 6 face boundary strips.
    """
    if channel_mask.dim() == 1:
        valid_ch = torch.where(channel_mask > 0)[0]
    else:
        valid_ch = torch.where(channel_mask[0] > 0)[0]

    w = bc_width
    # 3D faces: x-left/right, y-front/back, z-bottom/top
    # X faces: pred[:, :, :w, :, :, valid_ch] and pred[:, :, -w:, :, :, valid_ch]
    x_left_pred = pred[:, :, :w, :, :, :][:, :, :, :, :, valid_ch]
    x_left_target = target[:, :, :w, :, :, :][:, :, :, :, :, valid_ch]
    x_right_pred = pred[:, :, -w:, :, :, :][:, :, :, :, :, valid_ch]
    x_right_target = target[:, :, -w:, :, :, :][:, :, :, :, :, valid_ch]

    # Y faces (excluding X overlaps): pred[:, :, w:-w, :w, :, valid_ch]
    y_front_pred = pred[:, :, w:-w, :w, :, :][:, :, :, :, :, valid_ch]
    y_front_target = target[:, :, w:-w, :w, :, :][:, :, :, :, :, valid_ch]
    y_back_pred = pred[:, :, w:-w, -w:, :, :][:, :, :, :, :, valid_ch]
    y_back_target = target[:, :, w:-w, -w:, :, :][:, :, :, :, :, valid_ch]

    # Z faces (excluding X,Y overlaps): pred[:, :, w:-w, w:-w, :w, valid_ch]
    z_bot_pred = pred[:, :, w:-w, w:-w, :w, :][:, :, :, :, :, valid_ch]
    z_bot_target = target[:, :, w:-w, w:-w, :w, :][:, :, :, :, :, valid_ch]
    z_top_pred = pred[:, :, w:-w, w:-w, -w:, :][:, :, :, :, :, valid_ch]
    z_top_target = target[:, :, w:-w, w:-w, -w:, :][:, :, :, :, :, valid_ch]

    bc_pred = torch.cat([
        x_left_pred.reshape(-1), x_right_pred.reshape(-1),
        y_front_pred.reshape(-1), y_back_pred.reshape(-1),
        z_bot_pred.reshape(-1), z_top_pred.reshape(-1),
    ])
    bc_target = torch.cat([
        x_left_target.reshape(-1), x_right_target.reshape(-1),
        y_front_target.reshape(-1), y_back_target.reshape(-1),
        z_bot_target.reshape(-1), z_top_target.reshape(-1),
    ])

    mse = torch.mean((bc_pred - bc_target) ** 2)
    return torch.sqrt(mse + 1e-8)


def compute_pde_loss(
    output: torch.Tensor,
    input_data: torch.Tensor,
    pde_loss_fn: Advection3DPDELoss,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
) -> tuple:
    """
    Compute 3D Advection PDE residual loss.

    Prepends t0 frame from input for time derivatives.

    Args:
        output: Model output [B, T_out, X, Y, Z, 18] (denormalized)
        input_data: Model input [B, T_in, X, Y, Z, 18]
        pde_loss_fn: Advection3DPDELoss instance
        a: advection velocity x-component [B] (per-sample)
        b: advection velocity y-component [B] (per-sample)
        c: advection velocity z-component [B] (per-sample)

    Returns:
        total_loss, losses_dict
    """
    with torch.autocast(device_type='cuda', enabled=False):
        t0_u = input_data[:, 0:1, :, :, :, CH_U].float()
        out_u = output[:, :, :, :, :, CH_U].float()

        # [B, T_out+1, X, Y, Z]
        u = torch.cat([t0_u, out_u], dim=1)

        total_loss, losses = pde_loss_fn(u, a=a.float(), b=b.float(), c=c.float())

    return total_loss, losses


def compute_vrmse(
    output: torch.Tensor,
    target: torch.Tensor,
    channel_mask: torch.Tensor,
) -> tuple:
    """
    Compute VRMSE and RMSE for valid channels.

    VRMSE = mean of per-channel sqrt(MSE_ch / Var_ch).
    """
    valid_ch = (torch.where(channel_mask[0] > 0)[0] if channel_mask.dim() > 1
                else torch.where(channel_mask > 0)[0])
    output_valid = output[..., valid_ch]
    target_valid = target[..., valid_ch]

    mse = torch.mean((output_valid - target_valid) ** 2)
    rmse = torch.sqrt(mse + 1e-8)

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
    pde_loss_fn: Advection3DPDELoss,
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
        a = batch['params_a'].to(device=accelerator.device)
        b = batch['params_b'].to(device=accelerator.device)
        c = batch['params_c'].to(device=accelerator.device)

        input_data = data[:, :t_input]
        target_data = data[:, 1:t_input + 1]

        output_norm, mean, std = model(input_data, return_normalized=True)
        output = output_norm * std + mean

        bc_loss = compute_boundary_loss(output, target_data, channel_mask, bc_width=bc_width)
        pde_loss, _ = compute_pde_loss(output, input_data, pde_loss_fn, a, b, c)
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
    timeout_kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(minutes=30))

    accelerator = Accelerator(
        mixed_precision=config['training'].get('mixed_precision', 'no'),
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1),
        log_with="wandb",
        kwargs_handlers=[ddp_kwargs, timeout_kwargs]
    )

    if accelerator.is_main_process:
        logger.info(f"{'='*60}")
        logger.info("3D Advection LoRA Finetuning (V3 Model)")
        logger.info(f"{'='*60}")
        logger.info(f"Config: {args.config}")
        logger.info(f"Loss: BC (w={bc_width}) + PDE (u_t + a*u_x + b*u_y + c*u_z)")
        logger.info(f"Lambda BC: {lambda_bc}, Lambda PDE: {lambda_pde}")
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

    pretrained_path = config['model'].get('pretrained_path', None)
    freeze_encoder = config['model'].get('freeze_encoder', False)
    freeze_decoder = config['model'].get('freeze_decoder', False)

    model = PDELoRAModelV3(
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

    # Create PDE loss function
    physics = config.get('physics', {})
    eq_scales = physics.get('eq_scales', None)
    eq_weights = physics.get('eq_weights', None)

    eq_scales_per_t = None
    eq_scales_per_t_path = physics.get('eq_scales_per_t_path', None)
    if eq_scales_per_t_path:
        p = Path(eq_scales_per_t_path)
        if p.exists():
            eq_scales_per_t = torch.load(str(p), map_location='cpu', weights_only=False)
            if accelerator.is_main_process:
                logger.info(f"Loaded per-timestep PDE scales from {p}")

    pde_loss_fn = Advection3DPDELoss(
        dx=physics.get('dx', 2 * 3.141592653589793 / 64),
        dy=physics.get('dy', 2 * 3.141592653589793 / 64),
        dz=physics.get('dz', 2 * 3.141592653589793 / 64),
        dt=physics.get('dt', 0.05),
        eq_scales=eq_scales,
        eq_weights=eq_weights,
        eq_scales_per_t=eq_scales_per_t,
    ).to(accelerator.device)

    if accelerator.is_main_process:
        logger.info(f"PDE eq_scales: {pde_loss_fn.eq_scales}")
        logger.info(f"PDE eq_weights: {pde_loss_fn.eq_weights}")

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
    elif args.init_weights:
        ckpt = load_lora_checkpoint(
            accelerator.unwrap_model(model), args.init_weights,
            optimizer=None, scheduler=None,
        )
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        if accelerator.is_main_process:
            logger.info(f"Loaded weights from {args.init_weights}")
            logger.info(f"  best_val_loss = {best_val_loss:.6f}")

    if accelerator.is_main_process:
        lora_r = config['model'].get('lora', {}).get('r', 16)
        run_name = f"advection3d-lora-v3-r{lora_r}"
        accelerator.init_trackers(
            project_name=config['logging']['project'],
            config=config,
            init_kwargs={"wandb": {
                "entity": config['logging'].get('entity'),
                "name": run_name,
                "tags": ["advection_3d", "lora", "v3", "BC+PDE", "normalized"],
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
                a = batch['params_a'].to(device=accelerator.device)
                b = batch['params_b'].to(device=accelerator.device)
                c = batch['params_c'].to(device=accelerator.device)

                input_data = data[:, :t_input]
                target_data = data[:, 1:t_input + 1]
                in_warmup = global_step < warmup_steps

                with accelerator.accumulate(model):
                    output_norm, mean, std = model(input_data, return_normalized=True)
                    output = output_norm * std + mean

                    bc_loss = compute_boundary_loss(output, target_data, channel_mask, bc_width=bc_width)
                    pde_loss, losses = compute_pde_loss(output, input_data, pde_loss_fn, a, b, c)

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
                                save_lora_checkpoint(
                                    model=accelerator.unwrap_model(model),
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    global_step=global_step,
                                    metrics={
                                        'bc': val_bc, 'pde': val_pde,
                                        'rmse': val_rmse, 'vrmse': val_vrmse,
                                    },
                                    save_path=str(save_dir / 'best_lora.pt'),
                                    config=config,
                                    patience_counter=patience_counter,
                                    best_val_loss=best_val_loss,
                                )
                                console.print(
                                    f"[yellow]Saved best model[/yellow] "
                                    f"(save={save_metric:.4f}: {save_vrmse_weight}*{val_vrmse:.4f} + {save_pde_weight}*{val_pde:.4f})"
                                )
                        else:
                            patience_counter += 1
                            if accelerator.is_main_process:
                                console.print(f"[dim]Patience: {patience_counter}/{early_stopping_patience}[/dim]")
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
        table.add_row("Checkpoint", str(save_dir / "best_lora.pt"))
        console.print(table)


if __name__ == "__main__":
    main()
