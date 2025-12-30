"""
Evaluate pretrained model and LoRA finetuned model on Burgers validation set.

Metrics:
- PDE loss (burgers_pde_loss_upwind)
- RMSE
- MSE

Usage:
    python eval_models.py
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from pipeline import PDECausalModel
from model_lora import PDELoRAModel
from dataset_burgers import BurgersDataset, BurgersSampler, burgers_collate_fn
from pde_loss import burgers_pde_loss_upwind
from torch.utils.data import DataLoader


def load_pretrained_model(checkpoint_path: str, config: dict, device: torch.device):
    """Load pretrained PDECausalModel."""
    model = PDECausalModel(config)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    return model


def load_lora_model(pretrained_path: str, lora_path: str, config: dict, device: torch.device):
    """Load LoRA finetuned model."""
    # Create model with LoRA
    model = PDELoRAModel(config, pretrained_path=pretrained_path)

    # Load LoRA weights
    checkpoint = torch.load(lora_path, map_location='cpu')
    if 'lora_state_dict' in checkpoint:
        lora_state_dict = checkpoint['lora_state_dict']
        # Load LoRA weights into transformer
        current_state = model.model.transformer.state_dict()
        for name, param in lora_state_dict.items():
            if name in current_state:
                current_state[name].copy_(param)
        model.model.transformer.load_state_dict(current_state)

    model = model.to(device)
    model.eval()

    return model


def compute_metrics(output, input_data, target_data, batch, dt=1/999):
    """
    Compute all metrics.

    Args:
        output: [B, 16, H, W, 6] model prediction
        input_data: [B, 16, H, W, 6] model input
        target_data: [B, 16, H, W, 6] ground truth
        batch: batch dict with boundaries
        dt: time step

    Returns:
        dict with mse, rmse, pde_loss
    """
    device = output.device

    # MSE and RMSE (only u, v channels)
    pred_uv = output[..., :2].float()
    target_uv = target_data[..., :2].float()

    mse = torch.mean((pred_uv - target_uv) ** 2)
    rmse = torch.sqrt(mse)

    # PDE loss
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

    return {
        'mse': mse.item(),
        'rmse': rmse.item(),
        'pde_loss': pde_loss.item(),
    }


@torch.no_grad()
def evaluate_model(model, val_loader, device, model_name="Model"):
    """Evaluate a model on validation set."""
    print(f"\nEvaluating {model_name}...")

    total_mse = 0.0
    total_rmse = 0.0
    total_pde_loss = 0.0
    num_batches = 0

    for batch in tqdm(val_loader, desc=f"Evaluating {model_name}"):
        data = batch['data'].to(device=device, dtype=torch.bfloat16)

        input_data = data[:, :-1]   # [B, 16, H, W, 6]
        target_data = data[:, 1:]   # [B, 16, H, W, 6]

        output = model(input_data)

        metrics = compute_metrics(output, input_data, target_data, batch)

        total_mse += metrics['mse']
        total_rmse += metrics['rmse']
        total_pde_loss += metrics['pde_loss']
        num_batches += 1

    avg_mse = total_mse / num_batches
    avg_rmse = total_rmse / num_batches
    avg_pde_loss = total_pde_loss / num_batches

    return {
        'mse': avg_mse,
        'rmse': avg_rmse,
        'pde_loss': avg_pde_loss,
    }


def main():
    # Paths
    pretrained_path = "./checkpoints_e2e_medium/best.pt"
    lora_path = "./checkpoints_burgers_lora/best_lora.pt"
    data_path = "./burgers2d_nu0.1_0.15_res128_t1000_n100.h5"

    # Check paths
    for path, name in [(pretrained_path, "Pretrained"), (lora_path, "LoRA"), (data_path, "Data")]:
        if not Path(path).exists():
            print(f"ERROR: {name} not found: {path}")
            return

    # Config (must match pretrained model)
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create validation dataloader
    val_dataset = BurgersDataset(
        data_path=data_path,
        temporal_length=16,
        split='val',
        train_ratio=0.9,
        seed=42,
        clips_per_sample=120,  # Same as training
    )

    batch_size = 8
    val_sampler = BurgersSampler(val_dataset, batch_size, shuffle=False, seed=42)
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=burgers_collate_fn,
        num_workers=4,
        pin_memory=True
    )

    print(f"\nValidation set: {len(val_dataset)} clips, {len(val_sampler)} batches")

    # Evaluate pretrained model
    print("\n" + "="*60)
    print("Loading Pretrained Model...")
    print("="*60)
    pretrained_model = load_pretrained_model(pretrained_path, config, device)
    pretrained_results = evaluate_model(pretrained_model, val_loader, device, "Pretrained")
    del pretrained_model
    torch.cuda.empty_cache()

    # Evaluate LoRA model
    print("\n" + "="*60)
    print("Loading LoRA Finetuned Model...")
    print("="*60)
    lora_model = load_lora_model(pretrained_path, lora_path, config, device)
    lora_results = evaluate_model(lora_model, val_loader, device, "LoRA Finetuned")
    del lora_model
    torch.cuda.empty_cache()

    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"{'Metric':<15} {'Pretrained':>15} {'LoRA Finetuned':>15} {'Improvement':>15}")
    print("-"*60)

    for metric in ['mse', 'rmse', 'pde_loss']:
        pre_val = pretrained_results[metric]
        lora_val = lora_results[metric]
        improvement = (pre_val - lora_val) / pre_val * 100 if pre_val != 0 else 0

        print(f"{metric.upper():<15} {pre_val:>15.6f} {lora_val:>15.6f} {improvement:>14.2f}%")

    print("="*60)


if __name__ == "__main__":
    main()
