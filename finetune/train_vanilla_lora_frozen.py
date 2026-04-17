"""
Vanilla LoRA Finetuning with Frozen Enc/Dec — for ablation comparison.

Same training loop as train_muon_lora.py but uses standard PEFT LoRA (PDELoRAModelV3).
Supports init_from to load trained enc/dec from a checkpoint with shape checking.

Usage:
    torchrun --nproc_per_node=4 finetune/train_vanilla_lora_frozen.py \
        --config configs/finetune_taylor_green_2d_v3_vanilla_r32.yaml
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
import inspect
import yaml
import torch
import datetime
from pathlib import Path
from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
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
from finetune.model_lora_v3 import PDELoRAModelV3 as PDELoRAModelMuon, save_lora_checkpoint, load_lora_checkpoint

logger = logging.getLogger(__name__)


# ============================================================
# PDE Loss registry
# ============================================================
PDE_LOSS_REGISTRY = {}

def _lazy_import_pde_losses():
    global PDE_LOSS_REGISTRY
    if PDE_LOSS_REGISTRY:
        return
    from finetune.pde_loss_verified import (
        TaylorGreen2DPDELoss,
        KP2DPDELoss,
        Wave2DPDELoss,
        AdvDiff2DPDELoss,
        Burgers2DCHPDELoss,
    )
    PDE_LOSS_REGISTRY = {
        'taylor_green_2d': TaylorGreen2DPDELoss,
        'kp2_2d': KP2DPDELoss,
        'wave_2d': Wave2DPDELoss,
        'advdiff_2d': AdvDiff2DPDELoss,
        'burgers_2d_ch': Burgers2DCHPDELoss,
    }


def create_pde_loss(config: dict, device: torch.device):
    _lazy_import_pde_losses()
    pde_type = config.get('pde_type', '')
    physics = config.get('physics', {})

    if pde_type not in PDE_LOSS_REGISTRY:
        raise ValueError(f"Unknown pde_type: {pde_type}. Available: {list(PDE_LOSS_REGISTRY.keys())}")

    cls = PDE_LOSS_REGISTRY[pde_type]
    meta_keys = {'eq_scales', 'eq_weights', 'eq_scales_per_t_path', 'pde_type'}
    valid_params = set(inspect.signature(cls.__init__).parameters.keys()) - {'self'}
    kwargs = {k: v for k, v in physics.items() if k in valid_params and k not in meta_keys}
    kwargs['eq_scales'] = physics.get('eq_scales', None)
    kwargs['eq_weights'] = physics.get('eq_weights', None)

    # Per-timestep scales
    eq_scales_per_t_path = physics.get('eq_scales_per_t_path', None)
    if eq_scales_per_t_path and 'eq_scales_per_t' in valid_params:
        p = Path(eq_scales_per_t_path)
        if p.exists():
            kwargs['eq_scales_per_t'] = torch.load(str(p), map_location='cpu', weights_only=False)

    return cls(**kwargs).to(device)


# ============================================================
# Loss computation
# ============================================================

def compute_boundary_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    channel_mask: torch.Tensor,
    bc_width: int = 1,
) -> torch.Tensor:
    if channel_mask.dim() == 1:
        valid_ch = torch.where(channel_mask > 0)[0]
    else:
        valid_ch = torch.where(channel_mask[0] > 0)[0]

    w = bc_width
    top_pred = pred[:, :, :w, :, :][:, :, :, :, valid_ch]
    top_target = target[:, :, :w, :, :][:, :, :, :, valid_ch]
    bottom_pred = pred[:, :, -w:, :, :][:, :, :, :, valid_ch]
    bottom_target = target[:, :, -w:, :, :][:, :, :, :, valid_ch]
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


def compute_vrmse(
    output: torch.Tensor,
    target: torch.Tensor,
    channel_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
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


def compute_pde_loss_generic(
    output: torch.Tensor,
    input_data: torch.Tensor,
    pde_loss_fn,
    config: dict,
    batch: dict,
) -> tuple:
    """Generic PDE loss — extracts channels based on pde_type."""
    pde_type = config.get('pde_type', '')

    with torch.autocast(device_type='cuda', enabled=False):
        if pde_type == 'taylor_green_2d':
            CH_VX, CH_VY, CH_PRESS = 0, 1, 15
            t0_u = input_data[:, 0:1, :, :, CH_VX].float()
            t0_v = input_data[:, 0:1, :, :, CH_VY].float()
            t0_p = input_data[:, 0:1, :, :, CH_PRESS].float()
            u = torch.cat([t0_u, output[:, :, :, :, CH_VX].float()], dim=1)
            v = torch.cat([t0_v, output[:, :, :, :, CH_VY].float()], dim=1)
            p = torch.cat([t0_p, output[:, :, :, :, CH_PRESS].float()], dim=1)
            nu = batch.get('nu', None)
            if nu is not None:
                nu = nu.to(output.device)
            t_offset = batch.get('start_t', None)
            if t_offset is not None:
                t_offset = t_offset.to(output.device)
            total, losses = pde_loss_fn(u, v, p, nu=nu, t_offset=t_offset)

        elif pde_type in ('kp2_2d', 'advdiff_2d'):
            CH_U = 3
            t0_u = input_data[:, 0:1, :, :, CH_U].float()
            u = torch.cat([t0_u, output[:, :, :, :, CH_U].float()], dim=1)
            total, losses = pde_loss_fn(u)

        elif pde_type == 'wave_2d':
            CH_U, CH_W = 3, 4
            t0_u = input_data[:, 0:1, :, :, CH_U].float()
            t0_w = input_data[:, 0:1, :, :, CH_W].float()
            u = torch.cat([t0_u, output[:, :, :, :, CH_U].float()], dim=1)
            w = torch.cat([t0_w, output[:, :, :, :, CH_W].float()], dim=1)
            c = batch['nu'].to(output.device)
            total, losses = pde_loss_fn(u, w, c)

        elif pde_type == 'burgers_2d_ch':
            CH_VX, CH_VY = 0, 1
            t0_u = input_data[:, 0:1, :, :, CH_VX].float()
            t0_v = input_data[:, 0:1, :, :, CH_VY].float()
            u = torch.cat([t0_u, output[:, :, :, :, CH_VX].float()], dim=1)
            v = torch.cat([t0_v, output[:, :, :, :, CH_VY].float()], dim=1)
            nu = batch['nu'].to(output.device)
            total, losses = pde_loss_fn(u, v, nu=nu)

        else:
            raise ValueError(f"Unknown pde_type: {pde_type}")

    return total, losses


# ============================================================
# Args / Config / Scheduler
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Muon-LoRA Finetuning (BC + PDE)")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--reset_patience', action='store_true')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_lr_scheduler(optimizer, config: dict, total_steps: int):
    from torch.optim.lr_scheduler import LambdaLR
    import math
    warmup_steps = config['training'].get('warmup_steps', 200)
    min_lr = config['training'].get('min_lr', 1e-6)
    base_lr = config['training']['learning_rate']
    min_lr_ratio = min_lr / base_lr

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


# ============================================================
# Validate
# ============================================================

@torch.no_grad()
def validate(model, val_loader, accelerator, pde_loss_fn, config, t_input, bc_width):
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
        pde_loss, _ = compute_pde_loss_generic(output, input_data, pde_loss_fn, config, batch)
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
    return (
        (total_bc / num_batches).item() if n > 0 else 0,
        (total_pde / num_batches).item() if n > 0 else 0,
        (total_rmse / num_batches).item() if n > 0 else 0,
        (total_vrmse / num_batches).item() if n > 0 else 0,
    )


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config['dataset']['seed'])

    max_epochs = config['training'].get('max_epochs', 30)
    warmup_steps = config['training'].get('warmup_steps', 200)
    log_interval = config['logging']['log_interval']
    lambda_bc = config['training'].get('lambda_bc', 10000.0)
    lambda_pde = config['training'].get('lambda_pde', 1.0)
    grad_clip = config['training'].get('grad_clip', 1.0)
    eval_interval = config['training'].get('eval_interval', 200)
    early_stopping_patience = config['training'].get('early_stopping_patience', 30)
    save_vrmse_weight = config['training'].get('save_vrmse_weight', 100.0)
    save_pde_weight = config['training'].get('save_pde_weight', 0.0)
    bc_width = config['training'].get('bc_width', 1)
    t_input = config['dataset'].get('t_input', 8)
    pde_type = config.get('pde_type', 'unknown')

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    timeout_kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(minutes=30))

    accelerator = Accelerator(
        mixed_precision=config['training'].get('mixed_precision', 'no'),
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1),
        log_with="wandb",
        kwargs_handlers=[ddp_kwargs, timeout_kwargs],
    )

    if accelerator.is_main_process:
        ns_steps = config.get('model', {}).get('lora', {}).get('ns_steps', 5)
        logger.info(f"{'='*60}")
        logger.info("Muon-LoRA Finetuning (NS Orthogonalization)")
        logger.info(f"{'='*60}")
        logger.info(f"Config: {args.config}")
        logger.info(f"PDE type: {pde_type}")
        logger.info(f"NS steps: {ns_steps}")
        logger.info(f"Lambda BC: {lambda_bc}, Lambda PDE: {lambda_pde}")
        logger.info(f"BC width: {bc_width}")
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
        val_time_interval=config['dataset'].get('val_time_interval', 20),
    )

    steps_per_epoch = len(train_sampler)
    total_steps = max_epochs * steps_per_epoch

    if accelerator.is_main_process:
        logger.info(f"Steps per epoch: {steps_per_epoch}, Total: {total_steps}")

    model = PDELoRAModelMuon(
        config=config,
        pretrained_path=config['model'].get('pretrained_path'),
        freeze_encoder=config['model'].get('freeze_encoder', False),
        freeze_decoder=config['model'].get('freeze_decoder', False),
    )

    # Load encoder/decoder (and optionally LoRA) from a trained checkpoint
    init_from = config['model'].get('init_from', None)
    if init_from:
        ckpt = torch.load(init_from, map_location='cpu', weights_only=False)
        if 'trainable_state_dict' in ckpt:
            src_state = ckpt['trainable_state_dict']
            model_state = model.model.state_dict()
            loaded = 0
            for k, v in src_state.items():
                if k in model_state and model_state[k].shape == v.shape:
                    model_state[k] = v
                    loaded += 1
            model.model.load_state_dict(model_state)
            if accelerator.is_main_process:
                logger.info(f"Loaded {loaded}/{len(src_state)} keys from {init_from}")
        del ckpt

    if config['training'].get('mixed_precision', 'no') == 'no':
        model = model.float()

    trainable_params = model.get_trainable_params()
    if accelerator.is_main_process:
        n_total = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for p in trainable_params)
        logger.info(f"Parameters: {n_total:,} total, {n_train:,} trainable ({100*n_train/n_total:.1f}%)")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        betas=tuple(config['training'].get('betas', [0.9, 0.999])),
    )

    scheduler = get_lr_scheduler(optimizer, config, total_steps)
    model, optimizer = accelerator.prepare(model, optimizer)

    pde_loss_fn = create_pde_loss(config, accelerator.device)

    if accelerator.is_main_process:
        logger.info(f"PDE loss: {type(pde_loss_fn).__name__}")

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
        ns_steps = config['model'].get('lora', {}).get('ns_steps', 5)
        model_name = config.get('model_name', f'{pde_type}_muon_lora')
        accelerator.init_trackers(
            project_name=config['logging']['project'],
            config=config,
            init_kwargs={"wandb": {
                "entity": config['logging'].get('entity'),
                "name": model_name,
                "tags": [pde_type, "muon_lora", f"r{lora_r}", f"ns{ns_steps}", "BC+PDE"],
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
        SpinnerColumn(), TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40), TaskProgressColumn(),
        MofNCompleteColumn(), TimeRemainingColumn(),
        console=console, disable=not accelerator.is_main_process,
    ) as progress:
        train_task = progress.add_task("Muon-LoRA", total=total_steps, completed=global_step)

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
                    output_norm, mean, std = model(input_data, return_normalized=True)
                    output = output_norm * std + mean

                    bc_loss = compute_boundary_loss(output, target_data, channel_mask, bc_width=bc_width)
                    pde_loss, losses = compute_pde_loss_generic(
                        output, input_data, pde_loss_fn, config, batch)

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
                    log_dict = {
                        'train/bc_loss': bc_reduced.item(),
                        'train/pde_loss': pde_reduced.item(),
                        'train/total_loss': lambda_bc * bc_reduced.item() + lambda_pde * pde_reduced.item(),
                        'train/lr': scheduler.get_last_lr()[0],
                        'train/epoch': epoch + 1,
                    }

                    # Skip scale logging for vanilla LoRA (PEFT ModuleDict not compatible)

                    accelerator.log(log_dict, step=global_step)

                if global_step % eval_interval == 0:
                    val_bc, val_pde, val_rmse, val_vrmse = validate(
                        model, val_loader, accelerator, pde_loss_fn, config, t_input,
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

                            accelerator.wait_for_everyone()
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
                                    f"[yellow]Saved best[/yellow] "
                                    f"(save={save_metric:.4f}: "
                                    f"{save_vrmse_weight}*{val_vrmse:.4f} + {save_pde_weight}*{val_pde:.4f})"
                                )
                            accelerator.wait_for_everyone()
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
                console.print(
                    f"\n[blue]Epoch {epoch+1}/{max_epochs}:[/blue] "
                    f"avg_bc={epoch_bc/epoch_steps:.6f}, avg_pde={epoch_pde/epoch_steps:.6f}"
                )

    accelerator.end_training()

    if accelerator.is_main_process:
        table = Table(title="Training Complete (Muon-LoRA)", show_header=False, border_style="green")
        table.add_row("PDE Type", pde_type)
        table.add_row("Total Steps", str(global_step))
        table.add_row("Best Val Metric", f"{best_val_loss:.6f}")
        table.add_row("Early Stopped", "Yes" if early_stop else "No")
        table.add_row("Checkpoint", str(save_dir / "best_lora.pt"))
        console.print(table)


if __name__ == "__main__":
    main()
