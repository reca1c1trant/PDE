"""
Evaluation Script for Pretrained PDE Foundation Model.

Evaluates each dataset independently and reports 8 metrics:
1. RMSE - Root Mean Square Error
2. nRMSE - Normalized RMSE
3. CSV - Conserved Variables RMSE
4. Max - Maximum Error
5. BD - Boundary RMSE
6. F_low - Fourier Space RMSE (Low Frequency)
7. F_mid - Fourier Space RMSE (Mid Frequency)
8. F_high - Fourier Space RMSE (High Frequency)

Usage:
    # Single GPU
    python eval_pretrain.py --config configs/eval_pretrain.yaml

    # Multi-GPU (recommended for faster evaluation)
    torchrun --nproc_per_node=4 eval_pretrain.py --config configs/eval_pretrain.yaml
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
from torch.utils.data import Dataset, DataLoader, Sampler

from pipeline import PDECausalModel
from metrics import metric_func

# Suppress warnings on non-main processes
if os.environ.get('LOCAL_RANK', '0') != '0':
    warnings.filterwarnings('ignore')

# Constants from dataset_pretrain.py
NUM_VECTOR_CHANNELS = 3
NUM_SCALAR_CHANNELS = 15
TOTAL_CHANNELS = NUM_VECTOR_CHANNELS + NUM_SCALAR_CHANNELS  # 18
DEFAULT_TEMPORAL_LENGTH = 17  # Default: 16 input + 1 target
DEFAULT_INPUT_STEPS = 16

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

# Reverse mapping for display
SCALAR_NAMES = {v: k for k, v in SCALAR_INDICES.items()}


@dataclass
class EvalDatasetConfig:
    """Configuration for a single evaluation dataset."""
    name: str
    path: str
    Lx: float
    Ly: float
    channels: List[str]


class SingleEvalDataset(Dataset):
    """Dataset for evaluating a single pretrained dataset."""

    def __init__(
        self,
        config: EvalDatasetConfig,
        split: str = 'val',
        train_ratio: float = 0.9,
        seed: int = 42,
        sample_indices: Optional[List[int]] = None,  # Override sample indices for subset evaluation
        input_steps: int = DEFAULT_INPUT_STEPS,  # Number of input timesteps (1-16)
    ):
        self.config = config
        self.path = Path(config.path)
        self.split = split
        self.train_ratio = train_ratio
        self.seed = seed
        self.override_indices = sample_indices
        self.input_steps = input_steps
        self.temporal_length = input_steps + 1  # input + 1 target

        self._load_metadata()
        self._split_samples()
        self._generate_clips()

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

        self.max_start = self.n_timesteps - self.temporal_length

    def _split_samples(self):
        """Split samples into train/val sets."""
        # If override indices provided, use them directly
        if self.override_indices is not None:
            self.sample_indices = self.override_indices
            return

        rng = np.random.RandomState(self.seed)
        indices = rng.permutation(self.n_samples)
        split_idx = int(self.n_samples * self.train_ratio)

        if self.split == 'train':
            self.sample_indices = indices[:split_idx].tolist()
        else:
            self.sample_indices = indices[split_idx:].tolist()

    def _generate_clips(self):
        """Generate evaluation clips (all time windows for each sample)."""
        self.clips = []
        for sample_idx in self.sample_indices:
            if self.max_start <= 0:
                if self.n_timesteps >= self.temporal_length:
                    self.clips.append((sample_idx, 0))
            else:
                # For validation, sample every 5 timesteps
                for start_t in range(0, self.max_start + 1, 5):
                    self.clips.append((sample_idx, start_t))

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_idx, start_t = self.clips[idx]
        end_t = start_t + self.temporal_length

        with h5py.File(self.path, 'r') as f:
            if self.has_vector:
                vector = f['vector'][sample_idx, start_t:end_t]
            else:
                T = self.temporal_length
                H = W = self.spatial_size
                vector = np.zeros((T, H, W, NUM_VECTOR_CHANNELS), dtype=np.float32)

            if 'scalar' in f:
                scalar_compact = f['scalar'][sample_idx, start_t:end_t]
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

        channel_mask = np.zeros(TOTAL_CHANNELS, dtype=np.float32)
        if self.has_vector:
            channel_mask[:NUM_VECTOR_CHANNELS] = 1.0
        for idx_val in scalar_indices:
            channel_mask[NUM_VECTOR_CHANNELS + idx_val] = 1.0

        return {
            'data': torch.from_numpy(data),
            'channel_mask': torch.from_numpy(channel_mask),
            'sample_idx': sample_idx,
            'start_t': start_t,
        }


class EvalSampler(Sampler):
    """Simple distributed sampler for evaluation."""

    def __init__(
        self,
        dataset: SingleEvalDataset,
        batch_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size

        if num_replicas is None:
            num_replicas = int(os.environ.get('WORLD_SIZE', 1))
        if rank is None:
            rank = int(os.environ.get('RANK', 0))

        self.num_replicas = num_replicas
        self.rank = rank

        # Create batches
        indices = list(range(len(dataset)))
        batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]

        # Pad to ensure equal batches per rank
        remainder = len(batches) % num_replicas
        if remainder > 0:
            need = num_replicas - remainder
            for i in range(need):
                batches.append(batches[i % len(batches)])

        self.num_batches_per_rank = len(batches) // num_replicas
        self._all_batches = batches

    def __iter__(self):
        start_idx = self.rank * self.num_batches_per_rank
        end_idx = start_idx + self.num_batches_per_rank
        for batch in self._all_batches[start_idx:end_idx]:
            yield batch

    def __len__(self) -> int:
        return self.num_batches_per_rank


def eval_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for evaluation."""
    return {
        'data': torch.stack([item['data'] for item in batch], dim=0),
        'channel_mask': torch.stack([item['channel_mask'] for item in batch], dim=0),
        'sample_idx': [item['sample_idx'] for item in batch],
        'start_t': [item['start_t'] for item in batch],
    }


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


def get_channel_names(config: EvalDatasetConfig) -> List[str]:
    """Get display names for active channels."""
    return config.channels if config.channels else []


def get_valid_channel_indices(channel_mask: torch.Tensor) -> torch.Tensor:
    """Get indices of valid (non-padded) channels."""
    return torch.where(channel_mask.bool())[0]


@torch.no_grad()
def evaluate_dataset(
    model: PDECausalModel,
    dataset_config: EvalDatasetConfig,
    eval_config: dict,
    accelerator: Accelerator,
    sample_indices: Optional[List[int]] = None,
    subset_name: Optional[str] = None,
) -> Dict:
    """
    Evaluate model on a single dataset or subset.

    Args:
        sample_indices: If provided, only evaluate on these sample indices
        subset_name: If provided, use this name instead of dataset_config.name

    Returns:
        Dict with all 8 metrics and per-channel nRMSE
    """
    eval_name = subset_name or dataset_config.name

    if accelerator.is_main_process:
        print(f"\n{'='*70}")
        print(f"Evaluating: {eval_name}")
        print(f"{'='*70}")
        print(f"Path: {dataset_config.path}")
        print(f"Domain: Lx={dataset_config.Lx}, Ly={dataset_config.Ly}")

    # Get input_steps from config (default 16)
    input_steps = eval_config.get('input_steps', DEFAULT_INPUT_STEPS)

    # Create dataset and dataloader
    dataset = SingleEvalDataset(
        config=dataset_config,
        split='val',
        train_ratio=0.9,
        seed=eval_config.get('seed', 42),
        sample_indices=sample_indices,
        input_steps=input_steps,
    )

    batch_size = eval_config.get('batch_size', 8)
    sampler = EvalSampler(
        dataset,
        batch_size=batch_size,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
    )

    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=eval_collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    if accelerator.is_main_process:
        print(f"Samples: {len(dataset.sample_indices)} | Clips: {len(dataset)} | Batches/rank: {len(sampler)}")
        print(f"Input steps: {input_steps} | Temporal length: {input_steps + 1}")

    # Collect predictions
    all_preds = []
    all_targets = []
    channel_mask_ref = None
    channel_mask_nonzero = None  # Filter for non-zero channels (e.g., exclude Vz=0)
    filtered_channel_indices = None  # Track which original channels are kept

    iterator = tqdm(dataloader, desc=f"Evaluating {dataset_config.name}",
                    disable=not accelerator.is_main_process)

    for batch in iterator:
        data = batch['data'].to(device=accelerator.device, dtype=torch.float32)
        channel_mask = batch['channel_mask'].to(device=accelerator.device)

        # input: t[0:input_steps], target: t[1:input_steps+1]
        input_data = data[:, :-1]   # [B, input_steps, H, W, 18]
        target_data = data[:, 1:]   # [B, input_steps, H, W, 18]

        output = model(input_data)  # [B, 16, H, W, 18]

        # Get last timestep prediction
        pred_last = output[:, -1]       # [B, H, W, 18]
        target_last = target_data[:, -1]  # [B, H, W, 18]

        # Apply channel mask
        if channel_mask_ref is None:
            channel_mask_ref = channel_mask[0].cpu()

        valid_mask = channel_mask[0].bool()
        pred_valid = pred_last[..., valid_mask].float()
        target_valid = target_last[..., valid_mask].float()

        # Additionally filter out channels that are all zeros in target
        # This handles cases like Vz=0 in 2D simulations
        if channel_mask_nonzero is None:
            # Check which channels have non-zero values
            target_norms = target_valid.abs().mean(dim=(0, 1, 2))  # [C_valid]
            channel_mask_nonzero = target_norms > 1e-10  # True for non-zero channels
            # Track which channels are kept for naming
            valid_indices = torch.where(valid_mask)[0]
            filtered_channel_indices = valid_indices[channel_mask_nonzero].cpu().tolist()

        pred_valid = pred_valid[..., channel_mask_nonzero]
        target_valid = target_valid[..., channel_mask_nonzero]

        all_preds.append(pred_valid.cpu())
        all_targets.append(target_valid.cpu())

    # Gather from all ranks
    accelerator.wait_for_everyone()

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    all_preds_gathered = accelerator.gather(all_preds.to(accelerator.device))
    all_targets_gathered = accelerator.gather(all_targets.to(accelerator.device))

    results = {}

    if accelerator.is_main_process:
        # Add time dimension for metric_func: [N, H, W, T=1, C]
        all_preds_gathered = all_preds_gathered.unsqueeze(-2)
        all_targets_gathered = all_targets_gathered.unsqueeze(-2)

        N = all_preds_gathered.shape[0]
        print(f"Computing metrics on {N} predictions...")

        # Compute metrics in batches to avoid OOM
        metrics_batch_size = 1000
        total_RMSE = 0.0
        total_nRMSE = 0.0
        total_CSV = 0.0
        total_Max = 0.0
        total_BD = 0.0
        total_F = None
        total_MPPnRMSE = 0.0
        num_batches = 0

        for i in range(0, N, metrics_batch_size):
            batch_preds = all_preds_gathered[i:i+metrics_batch_size]
            batch_targets = all_targets_gathered[i:i+metrics_batch_size]

            err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F, err_MPPnRMSE = metric_func(
                batch_preds.float(),
                batch_targets.float(),
                if_mean=True,
                Lx=dataset_config.Lx,
                Ly=dataset_config.Ly,
                Lz=1.0,
                iLow=4,
                iHigh=12,
                initial_step=0,
            )

            total_RMSE += err_RMSE.item()
            total_nRMSE += err_nRMSE.item()
            total_CSV += err_CSV.item()
            total_Max = max(total_Max, err_Max.item())
            total_BD += err_BD.item()
            total_MPPnRMSE += err_MPPnRMSE.item()
            if total_F is None:
                total_F = err_F.cpu().clone()
            else:
                total_F += err_F.cpu()

            num_batches += 1

        # Average over batches
        results = {
            'name': eval_name,
            'num_samples': N,
            'RMSE': total_RMSE / num_batches,
            'nRMSE': total_nRMSE / num_batches,
            'CSV': total_CSV / num_batches,
            'Max': total_Max,
            'BD': total_BD / num_batches,
            'F_low': (total_F[0] / num_batches).item(),
            'F_mid': (total_F[1] / num_batches).item(),
            'F_high': (total_F[2] / num_batches).item(),
            'MPPnRMSE': total_MPPnRMSE / num_batches,
        }

        # Per-channel nRMSE
        C_valid = all_preds_gathered.shape[-1]
        channel_names_config = get_channel_names(dataset_config)

        # Map filtered channel indices to names
        # filtered_channel_indices contains the original 18-channel indices that were kept
        # We need to map these to user-friendly names
        vector_names = ['Vx', 'Vy', 'Vz']
        scalar_names_map = {v: k for k, v in SCALAR_INDICES.items()}

        def get_channel_name(global_idx: int) -> str:
            if global_idx < NUM_VECTOR_CHANNELS:
                return vector_names[global_idx]
            else:
                scalar_idx = global_idx - NUM_VECTOR_CHANNELS
                return scalar_names_map.get(scalar_idx, f"scalar_{scalar_idx}")

        per_channel_nRMSE = {}

        for ch_idx in range(C_valid):
            ch_nRMSE_sum = 0.0
            ch_batches = 0
            for i in range(0, N, metrics_batch_size):
                batch_preds = all_preds_gathered[i:i+metrics_batch_size, ..., ch_idx:ch_idx+1]
                batch_targets = all_targets_gathered[i:i+metrics_batch_size, ..., ch_idx:ch_idx+1]

                _, ch_nRMSE, _, _, _, _, _ = metric_func(
                    batch_preds.float(),
                    batch_targets.float(),
                    if_mean=True,
                    Lx=dataset_config.Lx,
                    Ly=dataset_config.Ly,
                    Lz=1.0,
                    iLow=4,
                    iHigh=12,
                    initial_step=0,
                )
                ch_nRMSE_sum += ch_nRMSE.item()
                ch_batches += 1

            # Get channel name from filtered indices
            if filtered_channel_indices is not None and ch_idx < len(filtered_channel_indices):
                global_idx = filtered_channel_indices[ch_idx]
                ch_name = get_channel_name(global_idx)
            elif ch_idx < len(channel_names_config):
                ch_name = channel_names_config[ch_idx]
            else:
                ch_name = f"ch_{ch_idx}"
            per_channel_nRMSE[ch_name] = ch_nRMSE_sum / ch_batches

        results['per_channel_nRMSE'] = per_channel_nRMSE

        # Print results
        print(f"\n{'-'*50}")
        print(f"Results for {eval_name}:")
        print(f"{'-'*50}")
        print(f"{'Metric':<25} {'Value':<15}")
        print(f"{'-'*50}")
        print(f"{'RMSE':<25} {results['RMSE']:.6e}")
        print(f"{'nRMSE':<25} {results['nRMSE']:.6e}")
        print(f"{'MPPnRMSE':<25} {results['MPPnRMSE']:.6e}")
        print(f"{'Max Error':<25} {results['Max']:.6e}")
        print(f"{'Conserved Vars RMSE':<25} {results['CSV']:.6e}")
        print(f"{'Boundary RMSE':<25} {results['BD']:.6e}")
        print(f"{'-'*50}")
        print("Fourier Space RMSE:")
        print(f"{'  Low Frequency':<25} {results['F_low']:.6e}")
        print(f"{'  Mid Frequency':<25} {results['F_mid']:.6e}")
        print(f"{'  High Frequency':<25} {results['F_high']:.6e}")
        print(f"{'-'*50}")
        print("Per-channel nRMSE:")
        for ch_name, ch_nrmse in per_channel_nRMSE.items():
            print(f"  {ch_name:<20} {ch_nrmse:.6e}")
        print(f"{'-'*50}")

    accelerator.wait_for_everyone()
    return results


@torch.no_grad()
def visualize_dataset(
    model: PDECausalModel,
    dataset_config: EvalDatasetConfig,
    eval_config: dict,
    output_dir: Path,
    device: torch.device,
):
    """
    Generate visualization for a dataset.

    Layout per sample: GT | Pred | Error for each channel
    """
    print(f"\nGenerating visualization for {dataset_config.name}...")

    input_steps = eval_config.get('input_steps', DEFAULT_INPUT_STEPS)

    dataset = SingleEvalDataset(
        config=dataset_config,
        split='val',
        train_ratio=0.9,
        seed=eval_config.get('seed', 42),
        input_steps=input_steps,
    )

    num_vis = eval_config.get('num_vis_samples', 3)
    num_vis = min(num_vis, len(dataset))

    # Random sample indices
    rng = np.random.RandomState(eval_config.get('seed', 42) + 123)
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

    results_list = []
    filtered_channel_names = None

    for idx in vis_indices:
        sample = dataset[idx]
        data = sample['data'].unsqueeze(0).to(device=device, dtype=torch.float32)
        channel_mask = sample['channel_mask']

        input_data = data[:, :-1]
        target_data = data[:, 1:]

        output = model(input_data)

        # Get last timestep
        pred_last = output[0, -1].float().cpu()
        target_last = target_data[0, -1].float().cpu()

        valid_mask = channel_mask.bool()
        pred_valid = pred_last[..., valid_mask]
        target_valid = target_last[..., valid_mask]

        # Filter out all-zero channels (e.g., Vz in 2D)
        target_norms = target_valid.abs().mean(dim=(0, 1))  # [C_valid]
        nonzero_mask = target_norms > 1e-10

        pred_valid = pred_valid[..., nonzero_mask].numpy()
        target_valid = target_valid[..., nonzero_mask].numpy()

        # Get filtered channel names (only once)
        if filtered_channel_names is None:
            valid_indices = torch.where(valid_mask)[0]
            kept_indices = valid_indices[nonzero_mask].tolist()
            filtered_channel_names = [get_channel_name(i) for i in kept_indices]

        results_list.append({
            'pred': pred_valid,
            'target': target_valid,
            'sample_idx': sample['sample_idx'],
            'start_t': sample['start_t'],
        })

    # Plot: 3 columns per channel (GT, Pred, Error)
    n_samples = len(results_list)
    channel_names = filtered_channel_names if filtered_channel_names else []
    n_channels = min(len(channel_names), results_list[0]['pred'].shape[-1]) if results_list else 0

    if n_channels == 0:
        print(f"  No channels to visualize for {dataset_config.name}")
        return

    fig, axes = plt.subplots(n_samples, n_channels * 3, figsize=(4 * n_channels * 3, 4 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)
    if n_channels * 3 == 1:
        axes = axes.reshape(-1, 1)

    extent = [0, dataset_config.Lx, 0, dataset_config.Ly]

    for row, res in enumerate(results_list):
        pred = res['pred']
        target = res['target']
        sample_idx = res['sample_idx']

        for ch_idx in range(n_channels):
            ch_name = channel_names[ch_idx] if ch_idx < len(channel_names) else f"ch_{ch_idx}"

            gt = target[:, :, ch_idx]
            pr = pred[:, :, ch_idx]
            error = pr - gt

            vmin = min(gt.min(), pr.min())
            vmax = max(gt.max(), pr.max())

            col_base = ch_idx * 3

            # Ground truth
            ax_gt = axes[row, col_base]
            im_gt = ax_gt.imshow(gt.T, origin='lower', extent=extent, cmap='jet', vmin=vmin, vmax=vmax)
            ax_gt.set_title(f'GT: {ch_name}', fontsize=11)
            if ch_idx == 0:
                ax_gt.set_ylabel(f'Sample {sample_idx}', fontsize=10)
            plt.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)

            # Prediction
            ax_pred = axes[row, col_base + 1]
            im_pred = ax_pred.imshow(pr.T, origin='lower', extent=extent, cmap='jet', vmin=vmin, vmax=vmax)
            rmse = np.sqrt(np.mean((pr - gt) ** 2))
            ax_pred.set_title(f'Pred (RMSE={rmse:.4f})', fontsize=11)
            plt.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)

            # Error (using RdBu_r colormap, symmetric around 0)
            ax_err = axes[row, col_base + 2]
            err_max = np.percentile(np.abs(error), 95)
            err_max = err_max if err_max > 0 else 1.0
            im_err = ax_err.imshow(error.T, origin='lower', extent=extent, cmap='RdBu_r',
                                   vmin=-err_max, vmax=err_max)
            mae = np.mean(np.abs(error))
            ax_err.set_title(f'Error (MAE={mae:.4f})', fontsize=11)
            plt.colorbar(im_err, ax=ax_err, fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path = output_dir / f"vis_{dataset_config.name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Pretrained PDE Foundation Model")
    parser.add_argument('--config', type=str, default='configs/eval_pretrain.yaml',
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
    input_steps = config.get('eval', {}).get('input_steps', DEFAULT_INPUT_STEPS)

    if accelerator.is_main_process:
        print(f"\n{'='*70}")
        print("PDE Foundation Model Evaluation")
        print(f"{'='*70}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Model config: {model_config_path}")
        print(f"Input steps: {input_steps}")
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
        dataset_configs.append(EvalDatasetConfig(
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

    eval_config = config.get('eval', {})

    # Evaluate each dataset
    all_results = []
    for ds_config in dataset_configs:
        # Special handling for 2D CFD: split M0.1 and M1.0
        if ds_config.name == '2d_cfd':
            # Get val indices using same seed as normal evaluation
            rng = np.random.RandomState(eval_config.get('seed', 42))
            n_samples = 4000  # 4 files × 1000 samples
            indices = rng.permutation(n_samples)
            val_indices = indices[int(n_samples * 0.9):].tolist()

            # Split by Mach number: M0.1 = samples 0-1999, M1.0 = samples 2000-3999
            m01_indices = [i for i in val_indices if i < 2000]
            m10_indices = [i for i in val_indices if i >= 2000]

            if accelerator.is_main_process:
                print(f"\n2D CFD split: M0.1={len(m01_indices)} val samples, M1.0={len(m10_indices)} val samples")

            # Evaluate M0.1
            if m01_indices:
                results = evaluate_dataset(model, ds_config, eval_config, accelerator,
                                          sample_indices=m01_indices, subset_name='2d_cfd_M0.1')
                if accelerator.is_main_process and results:
                    all_results.append(results)

            # Evaluate M1.0
            if m10_indices:
                results = evaluate_dataset(model, ds_config, eval_config, accelerator,
                                          sample_indices=m10_indices, subset_name='2d_cfd_M1.0')
                if accelerator.is_main_process and results:
                    all_results.append(results)

            # Also evaluate combined (all)
            results = evaluate_dataset(model, ds_config, eval_config, accelerator,
                                      subset_name='2d_cfd_all')
            if accelerator.is_main_process and results:
                all_results.append(results)
        else:
            results = evaluate_dataset(model, ds_config, eval_config, accelerator)
            if accelerator.is_main_process and results:
                all_results.append(results)

    # Print summary
    if accelerator.is_main_process and all_results:
        print("\n" + "=" * 100)
        print("                              EVALUATION SUMMARY")
        print("=" * 100)
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Datasets evaluated: {len(all_results)}")
        print()

        # Main metrics table
        header = f"{'Dataset':<20} {'nRMSE':<12} {'MPPnRMSE':<12} {'RMSE':<12} {'Max':<12} {'CSV':<12} {'BD':<12} {'Samples':<8}"
        print(header)
        print("-" * 100)
        for r in all_results:
            row = f"{r['name']:<20} {r['nRMSE']:<12.4e} {r['MPPnRMSE']:<12.4e} {r['RMSE']:<12.4e} {r['Max']:<12.4e} {r['CSV']:<12.4e} {r['BD']:<12.4e} {r['num_samples']:<8}"
            print(row)
        print("-" * 100)

        # Average
        if len(all_results) > 1:
            avg_nRMSE = np.mean([r['nRMSE'] for r in all_results])
            avg_MPPnRMSE = np.mean([r['MPPnRMSE'] for r in all_results])
            avg_RMSE = np.mean([r['RMSE'] for r in all_results])
            avg_CSV = np.mean([r['CSV'] for r in all_results])
            avg_BD = np.mean([r['BD'] for r in all_results])
            print(f"{'Average':<20} {avg_nRMSE:<12.4e} {avg_MPPnRMSE:<12.4e} {avg_RMSE:<12.4e} {'-':<12} {avg_CSV:<12.4e} {avg_BD:<12.4e}")
            print("-" * 100)

        # Fourier metrics
        print()
        print("Fourier Space RMSE (Low / Mid / High):")
        print("-" * 100)
        for r in all_results:
            print(f"  {r['name']:<18} {r['F_low']:<12.4e} {r['F_mid']:<12.4e} {r['F_high']:<12.4e}")
        print("-" * 100)
        print("=" * 100)

        # Generate visualizations (only on main process, after metrics)
        output_dir = Path(config.get('output_dir', './eval_results'))
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating visualizations to {output_dir}...")
        model_cpu = accelerator.unwrap_model(model)

        for ds_config in dataset_configs:
            visualize_dataset(
                model_cpu,
                ds_config,
                eval_config,
                output_dir,
                accelerator.device,
            )

        print(f"\nVisualization complete! Results saved to: {output_dir}")
        print("=" * 100)


if __name__ == "__main__":
    main()
