"""
Reproduce training validation process exactly.

This script replicates the validate() function from train_fno_baseline_flow.py
to debug PDE loss discrepancy.

Usage:
    python validate_pde.py --checkpoint checkpoints_mlp_baseline_flow_v2/best.pt --config configs/mlp_baseline_flow_v2.yaml
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


def load_model(config: dict, checkpoint_path: str, device: str = 'cuda'):
    """Load baseline model from checkpoint."""
    model_type = config.get('model', {}).get('type', 'unet')

    if model_type == 'mlp':
        from model_mlp_baseline import create_mlp_baseline
        model = create_mlp_baseline(config)
    elif model_type == 'unet':
        from model_unet_baseline import create_unet_baseline
        model = create_unet_baseline(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def compute_pde_loss(output, input_data, batch, config, device, pde_version="v1"):
    """
    Exact copy of compute_pde_loss from train_fno_baseline_flow.py
    but returns loss_time and loss_advection as well.
    """
    # Use only first channel for PDE loss
    t0_frame = input_data[:, 0:1, ..., :1]  # [B, 1, H, W, 1]
    output_u = output[..., :1]  # [B, T-1, H, W, 1]
    pred_with_t0 = torch.cat([t0_frame, output_u], dim=1)
    pred_u = pred_with_t0.float()

    # Boundaries only have 1 channel
    boundary_left = batch['boundary_left'].to(device).float()
    boundary_right = batch['boundary_right'].to(device).float()
    boundary_bottom = batch['boundary_bottom'].to(device).float()
    boundary_top = batch['boundary_top'].to(device).float()
    vtmax = batch['vtmax'].to(device).float()

    dt = config.get('physics', {}).get('dt', 1/999)
    vtmax_mean = vtmax.mean().item()

    # Select PDE loss version
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


@torch.no_grad()
def validate(model, val_loader, config, device, pde_version="v1", use_bf16=False):
    """
    Exact copy of validate() from train_fno_baseline_flow.py
    """
    model.eval()

    total_pde_loss = torch.zeros(1, device=device)
    total_loss_time = torch.zeros(1, device=device)
    total_loss_adv = torch.zeros(1, device=device)
    num_batches = torch.zeros(1, device=device)

    for batch_idx, batch in enumerate(val_loader):
        # Only use first channel (Flow Mixing has 1 real channel)
        # Training uses float32 for data even with bf16 model
        data = batch['data'][..., :1].to(device=device, dtype=torch.float32)
        input_data = data[:, :-1]

        # But if model is bf16, forward happens in bf16
        if use_bf16:
            data_model = data.to(torch.bfloat16)
            output = model(data_model).float()  # Convert back to float32
        else:
            output = model(data)

        pde_loss, loss_time, loss_adv = compute_pde_loss(
            output, input_data, batch, config, device, pde_version
        )

        total_pde_loss += pde_loss.detach()
        total_loss_time += loss_time.detach()
        total_loss_adv += loss_adv.detach()
        num_batches += 1

        # Print progress every 50 batches
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx:3d}/{len(val_loader)}: pde={pde_loss.item():.4f}, time={loss_time.item():.4f}, adv={loss_adv.item():.4f}")

    n = num_batches.item()
    return (
        (total_pde_loss / n).item() if n > 0 else 0,
        (total_loss_time / n).item() if n > 0 else 0,
        (total_loss_adv / n).item() if n > 0 else 0,
        n,
    )


def main():
    parser = argparse.ArgumentParser(description="Reproduce training validation")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--bf16', action='store_true', help='Use bf16 precision like training')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    config = load_config(args.config)
    pde_version = config.get('training', {}).get('pde_version', 'v1')
    batch_size = config['dataloader']['batch_size']

    # Check checkpoint metrics
    print(f"\n{'='*60}")
    print("Checkpoint Info")
    print(f"{'='*60}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'metrics' in checkpoint:
        print(f"Saved metrics: {checkpoint['metrics']}")
    if 'global_step' in checkpoint:
        print(f"Global step: {checkpoint['global_step']}")
    print(f"{'='*60}\n")

    print(f"Loading model from: {args.checkpoint}")
    model = load_model(config, args.checkpoint, device)

    # Convert to bf16 if requested
    if args.bf16:
        print("Converting model to bf16...")
        model = model.to(torch.bfloat16)

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

    print(f"\n{'='*60}")
    print(f"Validation Configuration")
    print(f"{'='*60}")
    print(f"PDE version: {pde_version} ({'central diff' if pde_version == 'v2' else '2nd upwind'})")
    print(f"Batch size: {batch_size}")
    print(f"Val samples: {len(val_dataset.samples)}")
    print(f"Val clips: {len(val_dataset)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    print(f"{'='*60}\n")

    print("Running validation (same as training)...")
    val_pde, val_time, val_adv, n_batches = validate(model, val_loader, config, device, pde_version, use_bf16=args.bf16)

    print(f"\n{'='*60}")
    print(f"Validation Results ({int(n_batches)} batches)")
    print(f"{'='*60}")
    print(f"  val_pde:       {val_pde:.6f}")
    print(f"  val_loss_time: {val_time:.6f}")
    print(f"  val_loss_adv:  {val_adv:.6f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
