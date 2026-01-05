"""
Evaluate and compare pretrained and LoRA finetuned models on Burgers validation set.

Metrics (from metrics.py):
- RMSE
- nRMSE (normalized RMSE)
- Max Error
- Boundary RMSE
- Fourier Space RMSE

Additional:
- PDE loss (burgers_pde_loss_upwind)

Usage:
    # Single GPU
    python eval_model.py

    # Multi-GPU
    torchrun --nproc_per_node=8 eval_model.py
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from accelerate import Accelerator
from pipeline import PDECausalModel
from model_lora import PDELoRAModel
from dataset_burgers import BurgersDataset, BurgersSampler, burgers_collate_fn
from pde_loss import burgers_pde_loss_upwind
from metrics import metric_func
from torch.utils.data import DataLoader


def load_pretrained_model(checkpoint_path: str, config: dict):
    """Load pretrained PDECausalModel."""
    model = PDECausalModel(config)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model


def load_lora_model(pretrained_path: str, lora_path: str, config: dict):
    """Load LoRA finetuned model."""
    model = PDELoRAModel(config, pretrained_path=pretrained_path)

    checkpoint = torch.load(lora_path, map_location='cpu')
    if 'lora_state_dict' in checkpoint:
        lora_state_dict = checkpoint['lora_state_dict']
        current_state = model.model.transformer.state_dict()
        for name, param in lora_state_dict.items():
            if name in current_state:
                current_state[name].copy_(param)
        model.model.transformer.load_state_dict(current_state)

    model.eval()

    return model


def compute_pde_loss(output, input_data, batch, device, dt=1/999):
    """Compute PDE loss only."""
    t0_frame = input_data[:, 0:1]
    pred_with_t0 = torch.cat([t0_frame, output], dim=1)
    pred_uv_pde = pred_with_t0[..., :2].float()

    boundary_left = batch['boundary_left'].to(device).float()
    boundary_right = batch['boundary_right'].to(device).float()
    boundary_bottom = batch['boundary_bottom'].to(device).float()
    boundary_top = batch['boundary_top'].to(device).float()
    nu = batch['nu'].to(device).float()
    nu_mean = nu.mean().item()

    pde_loss, _, _, _, _ = burgers_pde_loss_upwind(
        pred=pred_uv_pde,
        boundary_left=boundary_left,
        boundary_right=boundary_right,
        boundary_bottom=boundary_bottom,
        boundary_top=boundary_top,
        nu=nu_mean,
        dt=dt
    )

    return pde_loss


@torch.no_grad()
def evaluate_model(model, val_loader, accelerator, model_name="Model"):
    """
    Evaluate a model using metrics.py style evaluation.

    Collects all predictions and targets, then computes metrics globally.
    """
    if accelerator.is_main_process:
        print(f"\nEvaluating {model_name}...")

    all_preds = []
    all_targets = []
    total_pde_loss = torch.zeros(1, device=accelerator.device)
    num_batches = torch.zeros(1, device=accelerator.device)

    iterator = tqdm(val_loader, desc=f"Evaluating {model_name}", disable=not accelerator.is_main_process)

    for batch in iterator:
        data = batch['data'].to(device=accelerator.device, dtype=torch.bfloat16)

        input_data = data[:, :-1]   # [B, 16, H, W, 6]
        target_data = data[:, 1:]   # [B, 16, H, W, 6]

        output = model(input_data)  # [B, 16, H, W, 6]

        # Compute PDE loss
        pde_loss = compute_pde_loss(output, input_data, batch, accelerator.device)
        total_pde_loss += pde_loss
        num_batches += 1

        # Collect predictions and targets (only u, v channels)
        # Take last timestep prediction: [B, H, W, 2]
        pred_last = output[:, -1, :, :, :2].float()
        target_last = target_data[:, -1, :, :, :2].float()

        all_preds.append(pred_last.cpu())
        all_targets.append(target_last.cpu())

    # Reduce PDE loss across GPUs
    accelerator.wait_for_everyone()
    total_pde_loss = accelerator.reduce(total_pde_loss, reduction='sum')
    num_batches = accelerator.reduce(num_batches, reduction='sum')
    avg_pde_loss = (total_pde_loss / num_batches).item()

    # Gather all predictions and targets from all GPUs
    all_preds = torch.cat(all_preds, dim=0)      # [N_local, H, W, 2]
    all_targets = torch.cat(all_targets, dim=0)  # [N_local, H, W, 2]

    # Gather across all processes
    all_preds_gathered = accelerator.gather(all_preds.to(accelerator.device))
    all_targets_gathered = accelerator.gather(all_targets.to(accelerator.device))

    # Compute metrics only on main process
    results = {'pde_loss': avg_pde_loss}

    if accelerator.is_main_process:
        # Add time dimension for metric_func: [N, H, W, T=1, C=2]
        all_preds_gathered = all_preds_gathered.unsqueeze(-2)
        all_targets_gathered = all_targets_gathered.unsqueeze(-2)

        # Compute metrics using metric_func
        err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F = metric_func(
            all_preds_gathered.float(),
            all_targets_gathered.float(),
            if_mean=True,
            Lx=1.0, Ly=1.0, Lz=1.0,
            iLow=4, iHigh=12,
            initial_step=0
        )

        results.update({
            'rmse': err_RMSE.item(),
            'nrmse': err_nRMSE.item(),
            'max_error': err_Max.item(),
            'boundary_rmse': err_BD.item(),
            'csv_rmse': err_CSV.item(),
            'fourier_low': err_F[0].item(),
            'fourier_mid': err_F[1].item(),
            'fourier_high': err_F[2].item(),
        })

    # Broadcast results to all processes
    accelerator.wait_for_everyone()

    return results


def main():
    accelerator = Accelerator()

    # Paths
    pretrained_path = "./checkpoints_e2e_medium/best.pt"
    lora_path_v1 = "./checkpoints_burgers_lora/best_lora.pt"
    lora_path_v2 = "./checkpoints_burgers_lora_v2/best_lora.pt"
    data_path = "./burgers2d_nu0.1_0.15_res128_t1000_n100.h5"

    # Check paths
    if accelerator.is_main_process:
        for path, name in [
            (pretrained_path, "Pretrained"),
            (lora_path_v1, "LoRA v1"),
            (lora_path_v2, "LoRA v2"),
            (data_path, "Data")
        ]:
            if not Path(path).exists():
                print(f"ERROR: {name} not found: {path}")
                return

    accelerator.wait_for_everyone()

    # Config
    config = {
        'model': {
            'in_channels': 6,
            'noise_level': 0.0,
            'use_flash_attention': False,
            'gradient_checkpointing': False,
            'encoder': {
                'version': 'v2',
                'channels': [64, 128, 256],
                'use_resblock': True,
            },
            'transformer': {
                'hidden_size': 768,
                'num_hidden_layers': 10,
                'num_attention_heads': 12,
                'num_key_value_heads': 4,
                'intermediate_size': 3072,
                'hidden_act': 'silu',
                'max_position_embeddings': 4096,
                'rms_norm_eps': 1e-5,
                'rope_theta': 500000.0,
                'attention_dropout': 0.0,
            },
            'lora': {
                'r': 16,
                'alpha': 32,
                'dropout': 0.0,
                'target_modules': [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ],
            },
        },
    }

    if accelerator.is_main_process:
        print(f"Device: {accelerator.device}")
        print(f"Num processes: {accelerator.num_processes}")

    # Create validation dataloader
    val_dataset = BurgersDataset(
        data_path=data_path,
        temporal_length=16,
        split='val',
        train_ratio=0.9,
        seed=42,
        clips_per_sample=None,
    )

    batch_size = 8
    val_sampler = BurgersSampler(
        val_dataset, batch_size, shuffle=False, seed=42,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=burgers_collate_fn,
        num_workers=4,
        pin_memory=True
    )

    if accelerator.is_main_process:
        print(f"\nValidation set: {len(val_dataset)} clips, {len(val_sampler)} batches/rank")

    # Evaluate all models
    all_results = {}

    # Pretrained
    if accelerator.is_main_process:
        print("\n" + "="*60)
        print("Loading Pretrained Model...")
        print("="*60)
    model = load_pretrained_model(pretrained_path, config)
    model = model.to(accelerator.device)
    all_results['Pretrained'] = evaluate_model(model, val_loader, accelerator, "Pretrained")
    del model
    torch.cuda.empty_cache()

    # LoRA v1
    if accelerator.is_main_process:
        print("\n" + "="*60)
        print("Loading LoRA v1 Model...")
        print("="*60)
    model = load_lora_model(pretrained_path, lora_path_v1, config)
    model = model.to(accelerator.device)
    all_results['LoRA v1'] = evaluate_model(model, val_loader, accelerator, "LoRA v1")
    del model
    torch.cuda.empty_cache()

    # LoRA v2
    if accelerator.is_main_process:
        print("\n" + "="*60)
        print("Loading LoRA v2 Model...")
        print("="*60)
    model = load_lora_model(pretrained_path, lora_path_v2, config)
    model = model.to(accelerator.device)
    all_results['LoRA v2'] = evaluate_model(model, val_loader, accelerator, "LoRA v2")
    del model
    torch.cuda.empty_cache()

    # Print results
    if accelerator.is_main_process:
        print("\n" + "="*90)
        print("RESULTS (metrics.py style)")
        print("="*90)

        # Main metrics table
        metrics_to_show = ['rmse', 'nrmse', 'pde_loss', 'max_error', 'boundary_rmse']
        print(f"{'Metric':<15} {'Pretrained':>14} {'LoRA v1':>14} {'LoRA v2':>14} {'v1 vs v2':>12} {'v2 vs Pre':>12}")
        print("-"*90)

        for metric in metrics_to_show:
            pre_val = all_results['Pretrained'].get(metric, 0)
            v1_val = all_results['LoRA v1'].get(metric, 0)
            v2_val = all_results['LoRA v2'].get(metric, 0)

            imp_v1_vs_v2 = (v2_val - v1_val) / v2_val * 100 if v2_val != 0 else 0
            imp_v2_vs_pre = (pre_val - v2_val) / pre_val * 100 if pre_val != 0 else 0

            print(f"{metric.upper():<15} {pre_val:>14.6f} {v1_val:>14.6f} {v2_val:>14.6f} {imp_v1_vs_v2:>11.2f}% {imp_v2_vs_pre:>11.2f}%")

        print("-"*90)

        # Fourier metrics
        print("\nFourier Space RMSE:")
        print("-"*60)
        for freq, key in [('Low', 'fourier_low'), ('Mid', 'fourier_mid'), ('High', 'fourier_high')]:
            pre_val = all_results['Pretrained'].get(key, 0)
            v1_val = all_results['LoRA v1'].get(key, 0)
            v2_val = all_results['LoRA v2'].get(key, 0)
            print(f"  {freq:<10} Pretrained={pre_val:.6f}  LoRA_v1={v1_val:.6f}  LoRA_v2={v2_val:.6f}")

        print("="*90)


if __name__ == "__main__":
    main()
