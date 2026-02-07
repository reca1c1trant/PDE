"""
Autoregressive Evaluation Script for Pretrained PDE Foundation Model.

Performs autoregressive rollout prediction:
1. Input initial `input_steps` timesteps
2. Predict next step
3. Sliding window: if window full, drop oldest and append prediction; otherwise append
4. Repeat for `rollout_steps` rounds
5. Evaluate with Relative Squared Error (RSE): (pred - gt)² / (gt² + 1e-8)

Usage:
    # Single GPU
    python eval_autoregressive.py --config configs/eval_autoregressive.yaml

    # Multi-GPU
    torchrun --nproc_per_node=4 eval_autoregressive.py --config configs/eval_autoregressive.yaml
"""

import os
import sys
import warnings
import argparse
import yaml
import random
import numpy as np
import torch
import h5py
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader

from pipeline import PDECausalModel

# Suppress warnings on non-main processes
if os.environ.get('LOCAL_RANK', '0') != '0':
    warnings.filterwarnings('ignore')

# Constants
NUM_VECTOR_CHANNELS = 3
NUM_SCALAR_CHANNELS = 15
TOTAL_CHANNELS = NUM_VECTOR_CHANNELS + NUM_SCALAR_CHANNELS  # 18

# Scalar channel indices
SCALAR_INDICES = {
    'buoyancy': 0,
    'concentration_rho': 1,
    'concentration_u': 2,
    'concentration_v': 3,
    'density': 4,
    'electron_fraction': 5,
    'energy': 6,
    'entropy': 7,
    'geometry': 8,
    'gravitational_potential': 9,
    'height': 10,
    'passive_tracer': 11,
    'pressure': 12,
    'speed_of_sound': 13,
    'temperature': 14,
}


@dataclass
class ARDatasetConfig:
    """Configuration for autoregressive evaluation dataset."""
    name: str
    path: str
    Lx: float = 1.0
    Ly: float = 1.0
    channels: List[str] = None


class AutoregressiveDataset(Dataset):
    """
    Dataset for autoregressive evaluation.

    Returns full trajectories for rollout evaluation.
    """

    def __init__(
        self,
        config: ARDatasetConfig,
        split: str = 'val',
        train_ratio: float = 0.9,
        seed: int = 42,
        input_steps: int = 16,
        rollout_steps: int = 10,
        max_window_size: int = 16,
    ):
        self.config = config
        self.path = Path(config.path)
        self.split = split
        self.train_ratio = train_ratio
        self.seed = seed
        self.input_steps = input_steps
        self.rollout_steps = rollout_steps
        self.max_window_size = max_window_size

        # Total timesteps needed: initial input + rollout predictions
        self.total_timesteps = input_steps + rollout_steps

        self._load_metadata()
        self._split_samples()

    def _load_metadata(self):
        """Load dataset metadata."""
        with h5py.File(self.path, 'r') as f:
            if 'scalar' in f:
                self.n_samples, self.n_timesteps = f['scalar'].shape[:2]
                self.scalar_indices = f['scalar_indices'][:]
            else:
                self.n_samples = 0
                self.n_timesteps = 0
                self.scalar_indices = np.array([], dtype=np.int32)

            self.has_vector = 'vector' in f
            if self.has_vector:
                self.n_samples, self.n_timesteps = f['vector'].shape[:2]

            if 'scalar' in f:
                self.spatial_size = f['scalar'].shape[2]
            elif self.has_vector:
                self.spatial_size = f['vector'].shape[2]
            else:
                raise ValueError(f"No data found in {self.path}")

        # Check if we have enough timesteps
        if self.n_timesteps < self.total_timesteps:
            raise ValueError(
                f"Dataset {self.config.name} has {self.n_timesteps} timesteps, "
                f"but need {self.total_timesteps} (input={self.input_steps} + rollout={self.rollout_steps})"
            )

    def _split_samples(self):
        """Split samples into train/val sets."""
        rng = np.random.RandomState(self.seed)
        indices = rng.permutation(self.n_samples)
        split_idx = int(self.n_samples * self.train_ratio)

        if self.split == 'train':
            self.sample_indices = indices[:split_idx].tolist()
        else:
            self.sample_indices = indices[split_idx:].tolist()

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_idx = self.sample_indices[idx]

        # Load full trajectory for rollout
        with h5py.File(self.path, 'r') as f:
            if self.has_vector:
                vector = f['vector'][sample_idx, :self.total_timesteps]
            else:
                T = self.total_timesteps
                H = W = self.spatial_size
                vector = np.zeros((T, H, W, NUM_VECTOR_CHANNELS), dtype=np.float32)

            if 'scalar' in f:
                scalar_compact = f['scalar'][sample_idx, :self.total_timesteps]
                scalar_indices = f['scalar_indices'][:]
            else:
                scalar_compact = None
                scalar_indices = np.array([], dtype=np.int32)

        T, H, W = vector.shape[:3]
        scalar_full = np.zeros((T, H, W, NUM_SCALAR_CHANNELS), dtype=np.float32)
        if scalar_compact is not None:
            for i, idx_val in enumerate(scalar_indices):
                scalar_full[..., idx_val] = scalar_compact[..., i]

        data = np.concatenate([vector, scalar_full], axis=-1).astype(np.float32)

        # Channel mask
        channel_mask = np.zeros(TOTAL_CHANNELS, dtype=np.float32)
        if self.has_vector:
            channel_mask[:NUM_VECTOR_CHANNELS] = 1.0
        for idx_val in scalar_indices:
            channel_mask[NUM_VECTOR_CHANNELS + idx_val] = 1.0

        return {
            'data': torch.from_numpy(data),  # [T_total, H, W, C]
            'channel_mask': torch.from_numpy(channel_mask),
            'sample_idx': sample_idx,
        }


def compute_rse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute Relative Squared Error (RSE).

    RSE = (pred - target)² / (target² + eps)

    Args:
        pred: Predicted values [...]
        target: Ground truth values [...]
        eps: Small constant for numerical stability

    Returns:
        RSE per element, same shape as input
    """
    residual_sq = (pred - target) ** 2
    target_sq = target ** 2 + eps
    return residual_sq / target_sq


def compute_mean_rse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Compute mean RSE over all elements."""
    rse = compute_rse(pred, target, eps)
    return rse.mean().item()


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model(
    checkpoint_path: str,
    model_config: dict,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> PDECausalModel:
    """Load pretrained model from checkpoint."""
    model = PDECausalModel(model_config)

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Handle DDP/FSDP wrapped state dict
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        if k.startswith('_orig_mod.'):
            k = k[10:]
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model = model.to(device=device, dtype=dtype)
    model.eval()

    return model


@torch.no_grad()
def autoregressive_rollout(
    model: PDECausalModel,
    initial_input: torch.Tensor,
    rollout_steps: int,
    max_window_size: int = 16,
) -> torch.Tensor:
    """
    Perform autoregressive rollout prediction.

    Args:
        model: Trained PDE model
        initial_input: Initial input tensor [B, input_steps, H, W, C]
        rollout_steps: Number of steps to predict
        max_window_size: Maximum window size for sliding window

    Returns:
        predictions: All predicted steps [B, rollout_steps, H, W, C]
    """
    B = initial_input.shape[0]
    device = initial_input.device
    dtype = initial_input.dtype

    # Current window for prediction
    window = initial_input.clone()  # [B, current_len, H, W, C]

    predictions = []

    for step in range(rollout_steps):
        # Forward pass: predict next step
        output = model(window)  # [B, window_len, H, W, C]

        # Get the last predicted timestep
        pred_next = output[:, -1:]  # [B, 1, H, W, C]
        predictions.append(pred_next)

        # Update sliding window
        current_len = window.shape[1]
        if current_len >= max_window_size:
            # Window is full: drop oldest, append new prediction
            window = torch.cat([window[:, 1:], pred_next], dim=1)
        else:
            # Window not full: just append
            window = torch.cat([window, pred_next], dim=1)

    # Stack all predictions
    predictions = torch.cat(predictions, dim=1)  # [B, rollout_steps, H, W, C]

    return predictions


@torch.no_grad()
def evaluate_dataset(
    model: PDECausalModel,
    dataset_config: ARDatasetConfig,
    eval_config: dict,
    accelerator: Accelerator,
) -> Dict:
    """
    Evaluate autoregressive rollout on a dataset.

    Returns:
        Dict with RSE metrics per rollout step and overall
    """
    if accelerator.is_main_process:
        print(f"\n{'='*70}")
        print(f"Autoregressive Evaluation: {dataset_config.name}")
        print(f"{'='*70}")

    input_steps = eval_config.get('input_steps', 16)
    rollout_steps = eval_config.get('rollout_steps', 10)
    max_window_size = eval_config.get('max_window_size', 16)
    batch_size = eval_config.get('batch_size', 4)

    if accelerator.is_main_process:
        print(f"Input steps: {input_steps}")
        print(f"Rollout steps: {rollout_steps}")
        print(f"Max window size: {max_window_size}")

    # Create dataset
    dataset = AutoregressiveDataset(
        config=dataset_config,
        split='val',
        train_ratio=0.9,
        seed=eval_config.get('seed', 42),
        input_steps=input_steps,
        rollout_steps=rollout_steps,
        max_window_size=max_window_size,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    dataloader = accelerator.prepare(dataloader)

    if accelerator.is_main_process:
        print(f"Samples: {len(dataset)}")

    # Collect per-step RSE
    per_step_rse = [[] for _ in range(rollout_steps)]
    all_rse = []
    channel_mask_ref = None
    nonzero_mask = None

    iterator = tqdm(dataloader, desc=f"Evaluating {dataset_config.name}",
                    disable=not accelerator.is_main_process)

    for batch in iterator:
        data = batch['data'].to(device=accelerator.device, dtype=torch.float32)
        channel_mask = batch['channel_mask'].to(device=accelerator.device)

        # Split into initial input and ground truth for rollout
        initial_input = data[:, :input_steps]  # [B, input_steps, H, W, C]
        gt_rollout = data[:, input_steps:]      # [B, rollout_steps, H, W, C]

        # Perform autoregressive rollout
        pred_rollout = autoregressive_rollout(
            model, initial_input, rollout_steps, max_window_size
        )  # [B, rollout_steps, H, W, C]

        # Get channel mask (first sample)
        if channel_mask_ref is None:
            channel_mask_ref = channel_mask[0].cpu()
            valid_mask = channel_mask[0].bool()

            # Filter out all-zero channels
            gt_first = gt_rollout[0, 0, ..., valid_mask]  # [H, W, C_valid]
            channel_norms = gt_first.abs().mean(dim=(0, 1))  # [C_valid]
            nonzero_mask = channel_norms > 1e-10

        # Apply masks
        valid_mask = channel_mask[0].bool()
        pred_valid = pred_rollout[..., valid_mask][..., nonzero_mask]
        gt_valid = gt_rollout[..., valid_mask][..., nonzero_mask]

        # Compute RSE for each rollout step
        for step in range(rollout_steps):
            step_pred = pred_valid[:, step]  # [B, H, W, C_valid]
            step_gt = gt_valid[:, step]      # [B, H, W, C_valid]

            step_rse = compute_mean_rse(step_pred, step_gt)
            per_step_rse[step].append(step_rse)

        # Overall RSE
        overall_rse = compute_mean_rse(pred_valid, gt_valid)
        all_rse.append(overall_rse)

    # Gather and compute final metrics
    accelerator.wait_for_everyone()

    results = {}

    if accelerator.is_main_process:
        # Average per-step RSE
        avg_per_step = []
        for step in range(rollout_steps):
            if per_step_rse[step]:
                avg_step_rse = np.mean(per_step_rse[step])
                avg_per_step.append(avg_step_rse)
            else:
                avg_per_step.append(0.0)

        # Overall average RSE
        avg_overall = np.mean(all_rse) if all_rse else 0.0

        results = {
            'name': dataset_config.name,
            'input_steps': input_steps,
            'rollout_steps': rollout_steps,
            'max_window_size': max_window_size,
            'num_samples': len(dataset),
            'overall_rse': avg_overall,
            'per_step_rse': avg_per_step,
        }

        # Print results
        print(f"\n{'-'*60}")
        print(f"Results for {dataset_config.name}:")
        print(f"{'-'*60}")
        print(f"{'Overall RSE':<20}: {avg_overall:.6e}")
        print(f"\nPer-step RSE:")
        for step, rse in enumerate(avg_per_step):
            print(f"  Step {step+1:>2}: {rse:.6e}")
        print(f"{'-'*60}")

    accelerator.wait_for_everyone()
    return results


@torch.no_grad()
def visualize_rollout(
    model: PDECausalModel,
    dataset_config: ARDatasetConfig,
    eval_config: dict,
    output_dir: Path,
    device: torch.device,
):
    """Generate visualization of autoregressive rollout."""
    print(f"\nGenerating rollout visualization for {dataset_config.name}...")

    input_steps = eval_config.get('input_steps', 16)
    rollout_steps = eval_config.get('rollout_steps', 10)
    max_window_size = eval_config.get('max_window_size', 16)

    dataset = AutoregressiveDataset(
        config=dataset_config,
        split='val',
        train_ratio=0.9,
        seed=eval_config.get('seed', 42),
        input_steps=input_steps,
        rollout_steps=rollout_steps,
        max_window_size=max_window_size,
    )

    num_vis = min(eval_config.get('num_vis_samples', 2), len(dataset))
    rng = np.random.RandomState(eval_config.get('seed', 42) + 456)
    vis_indices = rng.choice(len(dataset), num_vis, replace=False)

    # Channel name mapping
    vector_names = ['Vx', 'Vy', 'Vz']
    scalar_names_map = {v: k for k, v in SCALAR_INDICES.items()}

    def get_channel_name(global_idx: int) -> str:
        if global_idx < NUM_VECTOR_CHANNELS:
            return vector_names[global_idx]
        else:
            scalar_idx = global_idx - NUM_VECTOR_CHANNELS
            return scalar_names_map.get(scalar_idx, f"scalar_{scalar_idx}")

    for vis_idx in vis_indices:
        sample = dataset[vis_idx]
        data = sample['data'].unsqueeze(0).to(device=device, dtype=torch.float32)
        channel_mask = sample['channel_mask']
        sample_idx = sample['sample_idx']

        initial_input = data[:, :input_steps]
        gt_rollout = data[:, input_steps:]

        # Perform rollout
        pred_rollout = autoregressive_rollout(
            model, initial_input, rollout_steps, max_window_size
        )

        # Apply masks
        valid_mask = channel_mask.bool()
        gt_valid = gt_rollout[0, ..., valid_mask].cpu()
        pred_valid = pred_rollout[0, ..., valid_mask].cpu()

        # Filter non-zero channels
        channel_norms = gt_valid[0].abs().mean(dim=(0, 1))
        nonzero_mask = channel_norms > 1e-10

        gt_valid = gt_valid[..., nonzero_mask].numpy()
        pred_valid = pred_valid[..., nonzero_mask].numpy()

        # Get channel names
        valid_indices = torch.where(valid_mask)[0]
        kept_indices = valid_indices[nonzero_mask].tolist()
        channel_names = [get_channel_name(i) for i in kept_indices]

        n_channels = len(channel_names)
        n_steps_to_show = min(5, rollout_steps)  # Show first 5 steps
        step_indices = [0, rollout_steps // 4, rollout_steps // 2, 3 * rollout_steps // 4, rollout_steps - 1]
        step_indices = [s for s in step_indices if s < rollout_steps][:n_steps_to_show]

        # Create figure: rows = channels, cols = steps * 3 (GT, Pred, Error)
        fig, axes = plt.subplots(n_channels, len(step_indices) * 3,
                                  figsize=(4 * len(step_indices) * 3, 4 * n_channels))

        if n_channels == 1:
            axes = axes.reshape(1, -1)

        extent = [0, dataset_config.Lx, 0, dataset_config.Ly]

        for ch_idx in range(n_channels):
            ch_name = channel_names[ch_idx]

            for col_idx, step in enumerate(step_indices):
                gt = gt_valid[step, :, :, ch_idx]
                pr = pred_valid[step, :, :, ch_idx]
                error = pr - gt
                rse = compute_mean_rse(
                    torch.from_numpy(pr), torch.from_numpy(gt)
                )

                vmin = min(gt.min(), pr.min())
                vmax = max(gt.max(), pr.max())

                col_base = col_idx * 3

                # Ground truth
                ax_gt = axes[ch_idx, col_base]
                im_gt = ax_gt.imshow(gt.T, origin='lower', extent=extent,
                                     cmap='jet', vmin=vmin, vmax=vmax)
                ax_gt.set_title(f'GT t={input_steps + step + 1}', fontsize=10)
                if col_idx == 0:
                    ax_gt.set_ylabel(ch_name, fontsize=11)
                plt.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)

                # Prediction
                ax_pred = axes[ch_idx, col_base + 1]
                im_pred = ax_pred.imshow(pr.T, origin='lower', extent=extent,
                                         cmap='jet', vmin=vmin, vmax=vmax)
                ax_pred.set_title(f'Pred (RSE={rse:.2e})', fontsize=10)
                plt.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)

                # Error
                ax_err = axes[ch_idx, col_base + 2]
                err_max = np.percentile(np.abs(error), 95)
                err_max = err_max if err_max > 0 else 1.0
                im_err = ax_err.imshow(error.T, origin='lower', extent=extent,
                                       cmap='RdBu_r', vmin=-err_max, vmax=err_max)
                ax_err.set_title('Error', fontsize=10)
                plt.colorbar(im_err, ax=ax_err, fraction=0.046, pad=0.04)

        plt.suptitle(f'{dataset_config.name} Sample {sample_idx} - Rollout ({input_steps} input, {rollout_steps} pred)',
                     fontsize=14)
        plt.tight_layout()

        save_path = output_dir / f"ar_vis_{dataset_config.name}_sample{sample_idx}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")

    # Plot RSE vs rollout step
    print(f"  Generating RSE curve...")

    # Compute RSE for all steps on a few samples
    n_eval = min(20, len(dataset))
    all_step_rse = [[] for _ in range(rollout_steps)]

    for idx in range(n_eval):
        sample = dataset[idx]
        data = sample['data'].unsqueeze(0).to(device=device, dtype=torch.float32)
        channel_mask = sample['channel_mask']

        initial_input = data[:, :input_steps]
        gt_rollout = data[:, input_steps:]
        pred_rollout = autoregressive_rollout(model, initial_input, rollout_steps, max_window_size)

        valid_mask = channel_mask.bool()
        gt_valid = gt_rollout[0, ..., valid_mask]
        pred_valid = pred_rollout[0, ..., valid_mask]

        # Filter non-zero
        channel_norms = gt_valid[0].abs().mean(dim=(0, 1))
        nonzero_mask = channel_norms > 1e-10
        gt_valid = gt_valid[..., nonzero_mask]
        pred_valid = pred_valid[..., nonzero_mask]

        for step in range(rollout_steps):
            step_rse = compute_mean_rse(pred_valid[step], gt_valid[step])
            all_step_rse[step].append(step_rse)

    # Plot
    mean_rse = [np.mean(rses) for rses in all_step_rse]
    std_rse = [np.std(rses) for rses in all_step_rse]
    steps = list(range(1, rollout_steps + 1))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, mean_rse, 'b-o', linewidth=2, markersize=6, label='Mean RSE')
    ax.fill_between(steps,
                    np.array(mean_rse) - np.array(std_rse),
                    np.array(mean_rse) + np.array(std_rse),
                    alpha=0.3, color='blue', label='Std')
    ax.set_xlabel('Rollout Step', fontsize=12)
    ax.set_ylabel('Relative Squared Error (RSE)', fontsize=12)
    ax.set_title(f'{dataset_config.name} - RSE vs Rollout Step\n(input={input_steps}, window={max_window_size})',
                 fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_path = output_dir / f"ar_rse_curve_{dataset_config.name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Autoregressive Evaluation for PDE Model")
    parser.add_argument('--config', type=str, default='configs/eval_autoregressive.yaml',
                        help='Path to evaluation config file')
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    # Initialize accelerator
    accelerator = Accelerator()

    # Set seed
    seed = config.get('eval', {}).get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load model config
    model_config_path = config['model_config']
    model_config = load_config(model_config_path)

    # Load model
    checkpoint_path = config['checkpoint']

    eval_config = config.get('eval', {})
    input_steps = eval_config.get('input_steps', 16)
    rollout_steps = eval_config.get('rollout_steps', 10)
    max_window_size = eval_config.get('max_window_size', 16)

    if accelerator.is_main_process:
        print(f"\n{'='*70}")
        print("PDE Foundation Model - Autoregressive Evaluation")
        print(f"{'='*70}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Model config: {model_config_path}")
        print(f"Input steps: {input_steps}")
        print(f"Rollout steps: {rollout_steps}")
        print(f"Max window size: {max_window_size}")
        print(f"Device: {accelerator.device}")
        print(f"Num processes: {accelerator.num_processes}")

    model = load_model(checkpoint_path, model_config, accelerator.device)
    model = accelerator.prepare(model)

    if accelerator.is_main_process:
        print("Model loaded successfully")

    # Parse dataset configs
    dataset_configs = []
    for ds in config.get('datasets', []):
        if not Path(ds['path']).exists():
            if accelerator.is_main_process:
                print(f"WARNING: Dataset not found: {ds['path']}")
            continue
        dataset_configs.append(ARDatasetConfig(
            name=ds['name'],
            path=ds['path'],
            Lx=ds.get('Lx', 1.0),
            Ly=ds.get('Ly', 1.0),
            channels=ds.get('channels', []),
        ))

    if not dataset_configs:
        if accelerator.is_main_process:
            print("ERROR: No datasets found!")
        return

    # Evaluate each dataset
    all_results = []
    for ds_config in dataset_configs:
        try:
            results = evaluate_dataset(model, ds_config, eval_config, accelerator)
            if accelerator.is_main_process and results:
                all_results.append(results)
        except ValueError as e:
            if accelerator.is_main_process:
                print(f"Skipping {ds_config.name}: {e}")

    # Print summary
    if accelerator.is_main_process and all_results:
        print("\n" + "=" * 80)
        print("                    AUTOREGRESSIVE EVALUATION SUMMARY")
        print("=" * 80)
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Input steps: {input_steps} | Rollout steps: {rollout_steps} | Max window: {max_window_size}")
        print()

        # Summary table
        header = f"{'Dataset':<25} {'Overall RSE':<15} {'Step 1':<12} {'Step 5':<12} {'Step 10':<12} {'Last Step':<12}"
        print(header)
        print("-" * 80)

        for r in all_results:
            step_rses = r['per_step_rse']
            step1 = step_rses[0] if len(step_rses) > 0 else 0
            step5 = step_rses[4] if len(step_rses) > 4 else step_rses[-1] if step_rses else 0
            step10 = step_rses[9] if len(step_rses) > 9 else step_rses[-1] if step_rses else 0
            last = step_rses[-1] if step_rses else 0

            row = f"{r['name']:<25} {r['overall_rse']:<15.4e} {step1:<12.4e} {step5:<12.4e} {step10:<12.4e} {last:<12.4e}"
            print(row)

        print("-" * 80)

        # Generate visualizations
        output_dir = Path(config.get('output_dir', './eval_ar_results'))
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating visualizations to {output_dir}...")
        model_cpu = accelerator.unwrap_model(model)

        for ds_config in dataset_configs:
            try:
                visualize_rollout(
                    model_cpu,
                    ds_config,
                    eval_config,
                    output_dir,
                    accelerator.device,
                )
            except ValueError as e:
                print(f"  Skipping visualization for {ds_config.name}: {e}")

        print(f"\nVisualization complete! Results saved to: {output_dir}")
        print("=" * 80)


if __name__ == "__main__":
    main()
