"""
Rollout Evaluation Script for APEBench datasets.

Computes two types of nRMSE:
1. One-step: always use GT as input, predict 1 step
2. Rollout: autoregressive, feed predictions back as input

Metrics:
- Per-timestep nRMSE (per channel): sqrt(sum_spatial(pred-gt)^2 / sum_spatial(gt^2))
- Aggregate: geometric mean over timesteps
- Per-channel then mean for overall

Usage:
    python tools/eval_rollout.py \
        --config configs/finetune_burgers_2d_apebench_rescaled_norm.yaml \
        --checkpoint checkpoints_burgers_2d_apebench_rescaled_norm/best_lora.pt \
        --model_type lora
"""

import argparse
import yaml
import torch
import numpy as np
import h5py
from pathlib import Path

torch.set_float32_matmul_precision('high')


def load_model(config: dict, checkpoint_path: str, model_type: str = 'lora'):
    """Load model from checkpoint."""
    if model_type == 'lora':
        from finetune.model_lora_v3 import PDELoRAModelV3, load_lora_checkpoint
        pretrained_path = config['model'].get('pretrained_path')
        model = PDELoRAModelV3(
            config=config,
            pretrained_path=pretrained_path,
            freeze_encoder=config['model'].get('freeze_encoder', False),
            freeze_decoder=config['model'].get('freeze_decoder', False),
        )
        load_lora_checkpoint(model, checkpoint_path)
    elif model_type == 'scratch':
        from pretrain.model_v3 import PDEModelV3
        model = PDEModelV3(config)
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state_dict, strict=False)
    elif model_type == 'pretrain':
        from pretrain.model_v3 import PDEModelV3
        model = PDEModelV3(config)
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state_dict, strict=False)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model.float()


def nrmse_per_sample(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Compute nRMSE for each sample.
    pred, gt: [M, *spatial, C] — one timestep, M samples.
    Returns: [M] nRMSE per sample.
    """
    # Flatten spatial dims: [M, -1]
    pred_flat = pred.reshape(pred.shape[0], -1)
    gt_flat = gt.reshape(gt.shape[0], -1)

    num = torch.sum((pred_flat - gt_flat) ** 2, dim=-1)  # [M]
    den = torch.sum(gt_flat ** 2, dim=-1)  # [M]
    return torch.sqrt(num / (den + 1e-12))  # [M]


def nrmse_per_channel(pred: torch.Tensor, gt: torch.Tensor, valid_channels: list) -> dict:
    """
    Compute per-channel nRMSE for each sample.
    pred, gt: [M, *spatial, C_total]
    Returns: dict of channel_name -> [M] nRMSE
    """
    results = {}
    for ch_idx, ch_name in valid_channels:
        p = pred[..., ch_idx]  # [M, *spatial]
        g = gt[..., ch_idx]
        p_flat = p.reshape(p.shape[0], -1)
        g_flat = g.reshape(g.shape[0], -1)
        num = torch.sum((p_flat - g_flat) ** 2, dim=-1)
        den = torch.sum(g_flat ** 2, dim=-1)
        results[ch_name] = torch.sqrt(num / (den + 1e-12))  # [M]
    return results


@torch.no_grad()
def evaluate(model, data: torch.Tensor, t_input: int, valid_channels: list, device: torch.device):
    """
    Run both one-step and rollout evaluation.

    Args:
        model: trained model
        data: [M, T_total, *spatial, C] full trajectories
        t_input: number of input frames (e.g. 8)
        valid_channels: list of (ch_idx, ch_name) for active channels
        device: cuda device

    Returns:
        one_step_nrmse: dict of ch_name -> [T_pred] mean nRMSE
        rollout_nrmse: dict of ch_name -> [T_pred] mean nRMSE
    """
    model.eval()
    M, T_total = data.shape[0], data.shape[1]
    T_pred = T_total - t_input  # number of prediction steps

    # ---- One-step evaluation ----
    one_step = {name: [] for _, name in valid_channels}
    for t in range(T_pred):
        input_frames = data[:, t:t + t_input].to(device)  # [M, t_input, *spatial, C]
        gt_frame = data[:, t + t_input]  # [M, *spatial, C]

        output_norm, mean, std = model(input_frames, return_normalized=True)
        output = output_norm * std + mean  # [M, t_input, *spatial, C]
        pred_frame = output[:, -1].cpu()  # last predicted frame [M, *spatial, C]

        ch_nrmse = nrmse_per_channel(pred_frame, gt_frame, valid_channels)
        for name, vals in ch_nrmse.items():
            one_step[name].append(vals.mean().item())  # mean over M samples

    # ---- Rollout evaluation ----
    rollout = {name: [] for _, name in valid_channels}
    # Start with GT input
    current_input = data[:, :t_input].clone().to(device)  # [M, t_input, *spatial, C]

    for t in range(T_pred):
        gt_frame = data[:, t_input + t]  # [M, *spatial, C]

        output_norm, mean, std = model(current_input, return_normalized=True)
        output = output_norm * std + mean
        pred_frame = output[:, -1]  # [M, *spatial, C] on device

        ch_nrmse = nrmse_per_channel(pred_frame.cpu(), gt_frame, valid_channels)
        for name, vals in ch_nrmse.items():
            rollout[name].append(vals.mean().item())

        # Slide window: drop first, append prediction
        current_input = torch.cat([current_input[:, 1:], pred_frame.unsqueeze(1)], dim=1)

    return one_step, rollout


def main():
    parser = argparse.ArgumentParser(description="Rollout Evaluation")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--model_type', type=str, default='lora', choices=['lora', 'scratch', 'pretrain'])
    parser.add_argument('--data_path', type=str, default=None, help='Override data path')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    t_input = config['dataset'].get('t_input', 8)
    vector_dim = config['dataset'].get('vector_dim', 2)
    data_path = args.data_path or config['dataset']['path']

    # Load full dataset
    with h5py.File(data_path, 'r') as f:
        vector = torch.tensor(f['vector'][:], dtype=torch.float32)
        scalar_data = f.get('scalar')
        if scalar_data is not None and scalar_data.shape[-1] > 0:
            scalar = torch.tensor(scalar_data[:], dtype=torch.float32)
            scalar_indices = f['scalar_indices'][:].tolist()
        else:
            scalar = None
            scalar_indices = []

    # Build 18-channel data
    M, T_total = vector.shape[0], vector.shape[1]
    spatial_shape = vector.shape[2:-1]
    data = torch.zeros(M, T_total, *spatial_shape, 18, dtype=torch.float32)
    C_vec = min(vector.shape[-1], 3)
    data[..., :C_vec] = vector[..., :C_vec]
    if scalar is not None:
        for i, idx in enumerate(scalar_indices):
            if idx < 15:
                data[..., 3 + idx] = scalar[..., i]

    # Determine valid channels
    valid_channels = []
    for ch in range(vector_dim):
        valid_channels.append((ch, f'v{ch}'))
    for idx in scalar_indices:
        valid_channels.append((3 + idx, f's{idx}'))

    print(f"Data: M={M}, T={T_total}, spatial={spatial_shape}")
    print(f"Valid channels: {valid_channels}")
    print(f"t_input={t_input}, T_pred={T_total - t_input}")

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(config, args.checkpoint, args.model_type).to(device)
    print(f"Model loaded: {args.model_type}")

    # Evaluate
    one_step, rollout = evaluate(model, data, t_input, valid_channels, device)

    T_pred = T_total - t_input

    # Compute geometric mean per channel
    print(f"\n{'='*70}")
    print(f"Results (T_pred={T_pred} steps, M={M} samples)")
    print(f"{'='*70}")

    one_step_gmeans = []
    rollout_gmeans = []

    for _, name in valid_channels:
        os_arr = np.array(one_step[name])
        ro_arr = np.array(rollout[name])

        os_gmean = np.exp(np.mean(np.log(os_arr + 1e-12)))
        ro_gmean = np.exp(np.mean(np.log(ro_arr + 1e-12)))

        one_step_gmeans.append(os_gmean)
        rollout_gmeans.append(ro_gmean)

        print(f"\n  Channel: {name}")
        print(f"    One-step  nRMSE: gmean={os_gmean:.6f}, t1={os_arr[0]:.6f}, t_last={os_arr[-1]:.6f}")
        print(f"    Rollout   nRMSE: gmean={ro_gmean:.6f}, t1={ro_arr[0]:.6f}, t_last={ro_arr[-1]:.6f}")

    os_overall = np.mean(one_step_gmeans)
    ro_overall = np.mean(rollout_gmeans)

    print(f"\n  Overall (mean of channel gmeans):")
    print(f"    One-step  nRMSE: {os_overall:.6f}")
    print(f"    Rollout   nRMSE: {ro_overall:.6f}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
