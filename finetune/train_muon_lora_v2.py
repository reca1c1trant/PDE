"""
Muon-LoRA V2 Finetuning — Orthogonal SVD-Parameterized LoRA.

Supports separate optimizer param groups for scale vs A/B vs encoder/decoder.
Config-driven: scale_lr_mult, lora_weight_decay control optimization.

Usage:
    torchrun --nproc_per_node=4 finetune/train_muon_lora_v2.py \
        --config configs/finetune_taylor_green_2d_v3_muon_v2.yaml
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
import numpy as np

torch.set_float32_matmul_precision('high')

if IS_MAIN_PROCESS:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
else:
    logging.disable(logging.CRITICAL)

from finetune.dataset_finetune import create_finetune_dataloaders
from finetune.model_lora_muon_v2 import PDELoRAModelMuonV2, save_lora_checkpoint, load_lora_checkpoint
from finetune.train_muon_lora import (
    create_pde_loss, compute_boundary_loss, compute_vrmse,
    compute_pde_loss_generic, validate, get_lr_scheduler,
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Muon-LoRA V2 Finetuning")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--reset_patience', action='store_true')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_optimizer(model: PDELoRAModelMuonV2, config: dict) -> torch.optim.AdamW:
    """Build AdamW with separate param groups for scale / lora_AB / others."""
    base_lr = config['training']['learning_rate']
    base_wd = config['training']['weight_decay']
    betas = tuple(config['training'].get('betas', [0.9, 0.999]))

    scale_lr_mult = config['training'].get('scale_lr_mult', 1.0)
    lora_wd = config['training'].get('lora_weight_decay', base_wd)

    scale_params: list[torch.nn.Parameter] = []
    lora_ab_params: list[torch.nn.Parameter] = []
    other_params: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'scale' in name:
            scale_params.append(param)
        elif 'lora_A' in name or 'lora_B' in name:
            lora_ab_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {'params': scale_params, 'lr': base_lr * scale_lr_mult, 'weight_decay': 0.0,
         'group_name': 'scale'},
        {'params': lora_ab_params, 'lr': base_lr, 'weight_decay': lora_wd,
         'group_name': 'lora_ab'},
        {'params': other_params, 'lr': base_lr, 'weight_decay': base_wd,
         'group_name': 'enc_dec'},
    ]

    if _is_main_process():
        for pg in param_groups:
            n = sum(p.numel() for p in pg['params'])
            logger.info(f"  Param group [{pg['group_name']}]: {n:,} params, "
                        f"lr={pg['lr']:.1e}, wd={pg['weight_decay']}")

    return torch.optim.AdamW(param_groups, betas=betas)


def log_scale_stats(model: torch.nn.Module) -> dict:
    """Collect scale parameter statistics for logging."""
    scale_vals = []
    unwrapped = model
    if hasattr(model, 'module'):
        unwrapped = model.module

    for name, module in unwrapped.model.transformer.named_modules():
        if hasattr(module, 'scale') and isinstance(module.scale, torch.nn.Parameter):
            s = module.scale.detach()
            if s.abs().max() > 0:  # skip inactive layers
                scale_vals.append(s)

    if not scale_vals:
        return {}

    all_s = torch.cat(scale_vals)
    return {
        'scale/mean_abs': all_s.abs().mean().item(),
        'scale/max_abs': all_s.abs().max().item(),
        'scale/l2_norm': all_s.norm().item(),
        'scale/nonzero_frac': (all_s.abs() > 0.01).float().mean().item(),
    }


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
        lora_cfg = config.get('model', {}).get('lora', {})
        logger.info(f"{'='*60}")
        logger.info("Muon-LoRA V2 Finetuning (Orthogonal SVD)")
        logger.info(f"{'='*60}")
        logger.info(f"Config: {args.config}")
        logger.info(f"PDE type: {pde_type}")
        logger.info(f"NS steps: {lora_cfg.get('ns_steps', 5)}")
        logger.info(f"Scale init: {lora_cfg.get('scale_init', 0.0)}")
        logger.info(f"Scale lr mult: {config['training'].get('scale_lr_mult', 1.0)}")
        logger.info(f"LoRA weight decay: {config['training'].get('lora_weight_decay', 'default')}")
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
        clips_per_sample=config['dataset'].get('clips_per_sample'),
        vector_dim=config['dataset'].get('vector_dim', 2),
        val_time_interval=config['dataset'].get('val_time_interval', 20),
    )

    steps_per_epoch = len(train_sampler)
    total_steps = max_epochs * steps_per_epoch

    if accelerator.is_main_process:
        logger.info(f"Steps per epoch: {steps_per_epoch}, Total: {total_steps}")

    model = PDELoRAModelMuonV2(
        config=config,
        pretrained_path=config['model'].get('pretrained_path'),
        freeze_encoder=config['model'].get('freeze_encoder', False),
        freeze_decoder=config['model'].get('freeze_decoder', False),
    )

    if config['training'].get('mixed_precision', 'no') == 'no':
        model = model.float()

    if accelerator.is_main_process:
        n_total = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Parameters: {n_total:,} total, {n_train:,} trainable ({100*n_train/n_total:.1f}%)")

    optimizer = build_optimizer(model, config)
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
        model_name = config.get('model_name', f'{pde_type}_muon_lora_v2')
        accelerator.init_trackers(
            project_name=config['logging']['project'],
            config=config,
            init_kwargs={"wandb": {
                "entity": config['logging'].get('entity'),
                "name": model_name,
                "tags": [pde_type, "muon_lora_v2", f"r{lora_r}", f"ns{ns_steps}", "BC+PDE"],
            }},
        )

    save_dir = Path(config['logging']['save_dir'])
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)

    console = Console()
    early_stop = False
    start_epoch = global_step // steps_per_epoch if steps_per_epoch > 0 else 0
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    model.train()

    with Progress(
        SpinnerColumn(), TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40), TaskProgressColumn(),
        MofNCompleteColumn(), TimeRemainingColumn(),
        console=console, disable=not accelerator.is_main_process,
    ) as progress:
        train_task = progress.add_task("Muon-LoRA V2", total=total_steps, completed=global_step)

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

                    # V2-specific: log scale statistics
                    scale_stats = log_scale_stats(accelerator.unwrap_model(model))
                    log_dict.update(scale_stats)

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
                        # Also log scale stats at eval time
                        s_stats = log_scale_stats(accelerator.unwrap_model(model))
                        s_info = f"||s||={s_stats.get('scale/l2_norm', 0):.4f}" if s_stats else ""
                        console.print(
                            f"\n[green]Step {global_step}/{total_steps} {phase_str}:[/green] "
                            f"vrmse={val_vrmse:.6f}, rmse={val_rmse:.6f}, "
                            f"pde={val_pde:.6f}, bc={val_bc:.6f} {s_info}"
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
        table = Table(title="Training Complete (Muon-LoRA V2)", show_header=False, border_style="green")
        table.add_row("PDE Type", pde_type)
        table.add_row("Total Steps", str(global_step))
        table.add_row("Best Val Metric", f"{best_val_loss:.6f}")
        table.add_row("Early Stopped", "Yes" if early_stop else "No")
        table.add_row("Checkpoint", str(save_dir / "best_lora.pt"))
        console.print(table)


if __name__ == "__main__":
    main()
