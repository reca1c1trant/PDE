"""
Reproduce LoRA training validation process exactly.

Usage:
    python validate_pde_lora.py --checkpoint checkpoints_flow_lora_v2/best_lora.pt --config configs/finetune_flow_v2.yaml
    python validate_pde_lora.py --checkpoint checkpoints_flow_lora_v2/best_lora.pt --config configs/finetune_flow_v2.yaml --fp32
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from dataset_flow import FlowMixingDataset, FlowMixingSampler, flow_mixing_collate_fn
from pde_loss_flow import flow_mixing_pde_loss
from pde_loss_flow_v2 import flow_mixing_pde_loss_v2


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_lora_model(config: dict, checkpoint_path: str, device: str = 'cuda'):
    """Load LoRA model from checkpoint."""
    from model_lora import PDELoRAModel, load_lora_checkpoint

    # Get pretrained path from config
    pretrained_path = config.get('model', {}).get('pretrained_path')

    # Create model with pretrained weights
    model = PDELoRAModel(config, pretrained_path=pretrained_path)

    # Load LoRA checkpoint
    print(f"Loading LoRA checkpoint from: {checkpoint_path}")
    load_lora_checkpoint(model.model, checkpoint_path)

    model = model.to(device)
    model.eval()
    return model


def compute_pde_loss(output, input_data, batch, config, device, pde_version="v1"):
    """
    Exact copy of compute_pde_loss from train_flow_lora.py
    """
    # CRITICAL: Convert to float32 BEFORE any computation
    t0_frame = input_data[:, 0:1].float()  # [B, 1, H, W, 6]
    output_f32 = output.float()  # [B, 16, H, W, 6]
    pred_with_t0 = torch.cat([t0_frame, output_f32], dim=1)  # [B, 17, H, W, 6]

    # Extract u channel (already float32)
    pred_u = pred_with_t0[..., :1]  # [B, 17, H, W, 1]

    # Get boundaries
    boundary_left = batch['boundary_left'].to(device).float()
    boundary_right = batch['boundary_right'].to(device).float()
    boundary_bottom = batch['boundary_bottom'].to(device).float()
    boundary_top = batch['boundary_top'].to(device).float()
    vtmax = batch['vtmax'].to(device).float()

    dt = config.get('physics', {}).get('dt', 1/999)
    vtmax_mean = vtmax.mean().item()

    if pde_version == "v2":
        pde_loss, loss_time, loss_advection, _ = flow_mixing_pde_loss_v2(
            pred=pred_u,
            boundary_left=boundary_left,
            boundary_right=boundary_right,
            boundary_bottom=boundary_bottom,
            boundary_top=boundary_top,
            vtmax=vtmax_mean,
            dt=dt
        )
    else:
        pde_loss, loss_time, loss_advection, _ = flow_mixing_pde_loss(
            pred=pred_u,
            boundary_left=boundary_left,
            boundary_right=boundary_right,
            boundary_bottom=boundary_bottom,
            boundary_top=boundary_top,
            vtmax=vtmax_mean,
            dt=dt
        )

    return pde_loss, loss_time, loss_advection


def compute_rmse_loss(output, target, channel_mask=None):
    """Compute RMSE loss."""
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
        mse = torch.mean((output[..., 0] - target[..., 0]) ** 2)

    return torch.sqrt(mse + 1e-8)


@torch.no_grad()
def validate(model, val_loader, config, device, pde_version="v1", use_bf16=True):
    """
    Exact copy of validate() from train_flow_lora.py
    """
    model.eval()

    total_pde_loss = torch.zeros(1, device=device)
    total_loss_time = torch.zeros(1, device=device)
    total_loss_adv = torch.zeros(1, device=device)
    total_rmse_loss = torch.zeros(1, device=device)
    num_batches = torch.zeros(1, device=device)

    for batch_idx, batch in enumerate(val_loader):
        # LoRA uses bf16 for data (same as training)
        if use_bf16:
            data = batch['data'].to(device=device, dtype=torch.bfloat16)
        else:
            data = batch['data'].to(device=device, dtype=torch.float32)

        channel_mask = batch['channel_mask'].to(device=device)

        input_data = data[:, :-1]  # [B, 16, H, W, 6]
        target = data[:, 1:]       # [B, 16, H, W, 6]

        output = model(input_data)

        pde_loss, loss_time, loss_adv = compute_pde_loss(
            output, input_data, batch, config, device, pde_version
        )
        rmse_loss = compute_rmse_loss(output.float(), target.float(), channel_mask)

        total_pde_loss += pde_loss.detach()
        total_loss_time += loss_time.detach()
        total_loss_adv += loss_adv.detach()
        total_rmse_loss += rmse_loss.detach()
        num_batches += 1

        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx:3d}/{len(val_loader)}: pde={pde_loss.item():.4f}, time={loss_time.item():.4f}, adv={loss_adv.item():.4f}")

    n = num_batches.item()
    return (
        (total_pde_loss / n).item() if n > 0 else 0,
        (total_loss_time / n).item() if n > 0 else 0,
        (total_loss_adv / n).item() if n > 0 else 0,
        (total_rmse_loss / n).item() if n > 0 else 0,
        n,
    )


def main():
    parser = argparse.ArgumentParser(description="Reproduce LoRA training validation")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--fp32', action='store_true', help='Use fp32 instead of bf16')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    config = load_config(args.config)
    pde_version = config.get('training', {}).get('pde_version', 'v1')
    batch_size = config['dataloader']['batch_size']

    # Check checkpoint info
    print(f"\n{'='*60}")
    print("Checkpoint Info")
    print(f"{'='*60}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'metrics' in checkpoint:
        print(f"Saved metrics: {checkpoint['metrics']}")
    if 'global_step' in checkpoint:
        print(f"Global step: {checkpoint['global_step']}")
    print(f"{'='*60}\n")

    print(f"Loading LoRA model from: {args.checkpoint}")
    model = load_lora_model(config, args.checkpoint, device)

    # Create val_loader exactly as in training
    val_dataset = FlowMixingDataset(
        data_path=config['dataset']['path'],
        temporal_length=config['dataset']['temporal_length'],
        split='val',
        train_ratio=config['dataset']['train_ratio'],
        seed=config['dataset']['seed'],
        clips_per_sample=None,
    )

    val_sampler = FlowMixingSampler(val_dataset, batch_size, shuffle=False, seed=config['dataset']['seed'])

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=flow_mixing_collate_fn,
        num_workers=config['dataloader']['num_workers'],
        pin_memory=config['dataloader']['pin_memory']
    )

    use_bf16 = not args.fp32
    print(f"\n{'='*60}")
    print(f"Validation Configuration")
    print(f"{'='*60}")
    print(f"PDE version: {pde_version} ({'central diff' if pde_version == 'v2' else '2nd upwind'})")
    print(f"Batch size: {batch_size}")
    print(f"Val samples: {len(val_dataset.samples)}")
    print(f"Val clips: {len(val_dataset)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Data dtype: {'bf16' if use_bf16 else 'fp32'}")
    print(f"{'='*60}\n")

    print("Running validation (same as training)...")
    val_pde, val_time, val_adv, val_rmse, n_batches = validate(
        model, val_loader, config, device, pde_version, use_bf16=use_bf16
    )

    print(f"\n{'='*60}")
    print(f"Validation Results ({int(n_batches)} batches)")
    print(f"{'='*60}")
    print(f"  val_pde:       {val_pde:.6f}")
    print(f"  val_loss_time: {val_time:.6f}")
    print(f"  val_loss_adv:  {val_adv:.6f}")
    print(f"  val_rmse:      {val_rmse:.6f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
