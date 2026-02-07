"""
PDEBench Baseline Evaluation Script.

Computes the official PDEBench metric L_sim:
    L_sim^u = (1/|B|) * sqrt( sum_{(x,t) in B} ((u_pred - u_gt) / sigma_u)^2 )

Where:
    - sigma_u is computed per channel over the entire validation set
    - B is the set of all spatial and temporal points
    - Final metric is averaged across channels

Usage:
    python eval_omni.py --config configs/eval_pretrain.yaml
    torchrun --nproc_per_node=8 eval_omni.py --config configs/eval_pretrain.yaml
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

# Suppress warnings on non-main processes
if os.environ.get('LOCAL_RANK', '0') != '0':
    warnings.filterwarnings('ignore')

# Constants
NUM_VECTOR_CHANNELS = 3
NUM_SCALAR_CHANNELS = 15
TOTAL_CHANNELS = NUM_VECTOR_CHANNELS + NUM_SCALAR_CHANNELS
TEMPORAL_LENGTH = 17

SCALAR_INDICES = {
    'buoyancy': 0, 'concentration_rho': 1, 'concentration_u': 2, 'concentration_v': 3,
    'density': 4, 'electron_fraction': 5, 'energy': 6, 'entropy': 7, 'geometry': 8,
    'gravitational_potential': 9, 'height': 10, 'passive_tracer': 11, 'pressure': 12,
    'speed_of_sound': 13, 'temperature': 14,
}
SCALAR_NAMES = {v: k for k, v in SCALAR_INDICES.items()}


@dataclass
class EvalDatasetConfig:
    name: str
    path: str
    Lx: float
    Ly: float
    channels: List[str]


class SingleEvalDataset(Dataset):
    """Dataset for evaluation."""

    def __init__(
        self,
        config: EvalDatasetConfig,
        split: str = 'val',
        train_ratio: float = 0.9,
        seed: int = 42,
        sample_indices: Optional[List[int]] = None,  # Override sample indices
    ):
        self.config = config
        self.path = Path(config.path)
        self.split = split
        self.train_ratio = train_ratio
        self.seed = seed
        self.override_indices = sample_indices

        self._load_metadata()
        self._split_samples()
        self._generate_clips()

    def _load_metadata(self):
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

        self.max_start = self.n_timesteps - TEMPORAL_LENGTH

    def _split_samples(self):
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
        self.clips = []
        for sample_idx in self.sample_indices:
            if self.max_start <= 0:
                if self.n_timesteps >= TEMPORAL_LENGTH:
                    self.clips.append((sample_idx, 0))
            else:
                for start_t in range(0, self.max_start + 1, 5):
                    self.clips.append((sample_idx, start_t))

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_idx, start_t = self.clips[idx]
        end_t = start_t + TEMPORAL_LENGTH

        with h5py.File(self.path, 'r') as f:
            if self.has_vector:
                vector = f['vector'][sample_idx, start_t:end_t]
            else:
                T = TEMPORAL_LENGTH
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
            # Only mark actual 2D vector channels (not Vz)
            channel_mask[:2] = 1.0  # Vx, Vy
        for idx_val in scalar_indices:
            channel_mask[NUM_VECTOR_CHANNELS + idx_val] = 1.0

        return {
            'data': torch.from_numpy(data),
            'channel_mask': torch.from_numpy(channel_mask),
            'sample_idx': sample_idx,
            'start_t': start_t,
        }


class EvalSampler(Sampler):
    def __init__(self, dataset, batch_size, num_replicas=None, rank=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas or int(os.environ.get('WORLD_SIZE', 1))
        self.rank = rank or int(os.environ.get('RANK', 0))

        indices = list(range(len(dataset)))
        batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]

        remainder = len(batches) % self.num_replicas
        if remainder > 0:
            for i in range(self.num_replicas - remainder):
                batches.append(batches[i % len(batches)])

        self.num_batches_per_rank = len(batches) // self.num_replicas
        self._all_batches = batches

    def __iter__(self):
        start = self.rank * self.num_batches_per_rank
        end = start + self.num_batches_per_rank
        for batch in self._all_batches[start:end]:
            yield batch

    def __len__(self):
        return self.num_batches_per_rank


def eval_collate_fn(batch):
    return {
        'data': torch.stack([item['data'] for item in batch], dim=0),
        'channel_mask': torch.stack([item['channel_mask'] for item in batch], dim=0),
        'sample_idx': [item['sample_idx'] for item in batch],
        'start_t': [item['start_t'] for item in batch],
    }


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path: str, model_config: dict, device: torch.device) -> PDECausalModel:
    model = PDECausalModel(model_config)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        if k.startswith('_orig_mod.'):
            k = k[10:]
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model = model.to(device=device, dtype=torch.float32)
    model.eval()
    return model


def compute_pdebench_nrmse(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[float, List[float]]:
    """
    Compute PDEBench L_sim metric.

    L_sim^u = (1/|B|) * sqrt( sum_{(x,t) in B} ((u_pred - u_gt) / sigma_u)^2 )

    Args:
        pred: [N, H, W, C] predictions
        target: [N, H, W, C] ground truth

    Returns:
        avg_nrmse: Average nRMSE across channels
        per_channel_nrmse: List of nRMSE per channel
    """
    C = pred.shape[-1]
    per_channel_nrmse = []

    for c in range(C):
        pred_c = pred[..., c]  # [N, H, W]
        target_c = target[..., c]  # [N, H, W]

        # Compute sigma over all samples (global std)
        sigma_c = target_c.std() + 1e-8

        # Normalized error
        normalized_error = (pred_c - target_c) / sigma_c

        # RMSE over all spatial and temporal points
        nrmse_c = torch.sqrt(torch.mean(normalized_error ** 2)).item()
        per_channel_nrmse.append(nrmse_c)

    avg_nrmse = np.mean(per_channel_nrmse)
    return avg_nrmse, per_channel_nrmse


@torch.no_grad()
def evaluate_dataset(
    model: PDECausalModel,
    dataset_config: EvalDatasetConfig,
    eval_config: dict,
    accelerator: Accelerator,
    sample_indices: Optional[List[int]] = None,
    subset_name: Optional[str] = None,
) -> Dict:
    """Evaluate on a dataset or subset."""
    name = subset_name or dataset_config.name

    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}")

    dataset = SingleEvalDataset(
        config=dataset_config,
        split='val',
        train_ratio=0.9,
        seed=eval_config.get('seed', 42),
        sample_indices=sample_indices,
    )

    batch_size = eval_config.get('batch_size', 8)
    sampler = EvalSampler(dataset, batch_size, accelerator.num_processes, accelerator.process_index)
    dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=eval_collate_fn,
                           num_workers=4, pin_memory=True)

    if accelerator.is_main_process:
        print(f"Samples: {len(dataset.sample_indices)} | Clips: {len(dataset)}")

    all_preds = []
    all_targets = []
    channel_mask_nonzero = None

    for batch in tqdm(dataloader, desc=name, disable=not accelerator.is_main_process):
        data = batch['data'].to(device=accelerator.device, dtype=torch.float32)
        channel_mask = batch['channel_mask'].to(device=accelerator.device)

        input_data = data[:, :-1]
        target_data = data[:, 1:]
        output = model(input_data)

        pred_last = output[:, -1].float()
        target_last = target_data[:, -1].float()

        valid_mask = channel_mask[0].bool()
        pred_valid = pred_last[..., valid_mask]
        target_valid = target_last[..., valid_mask]

        # Filter zero channels
        if channel_mask_nonzero is None:
            target_norms = target_valid.abs().mean(dim=(0, 1, 2))
            channel_mask_nonzero = target_norms > 1e-10

        pred_valid = pred_valid[..., channel_mask_nonzero]
        target_valid = target_valid[..., channel_mask_nonzero]

        all_preds.append(pred_valid.cpu())
        all_targets.append(target_valid.cpu())

    accelerator.wait_for_everyone()

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    all_preds = accelerator.gather(all_preds.to(accelerator.device))
    all_targets = accelerator.gather(all_targets.to(accelerator.device))

    results = {}
    if accelerator.is_main_process:
        avg_nrmse, per_channel = compute_pdebench_nrmse(all_preds, all_targets)

        results = {
            'name': name,
            'num_samples': all_preds.shape[0],
            'L_sim': avg_nrmse,
            'per_channel': per_channel,
        }

        print(f"\nL_sim (PDEBench nRMSE): {avg_nrmse:.6e}")
        print(f"Per-channel: {[f'{x:.4e}' for x in per_channel]}")

    accelerator.wait_for_everyone()
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="PDEBench Baseline Evaluation")
    parser.add_argument('--config', type=str, default='configs/eval_pretrain.yaml')
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    accelerator = Accelerator()

    seed = config.get('eval', {}).get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model_config = load_config(config['model_config'])
    checkpoint_path = config['checkpoint']

    if accelerator.is_main_process:
        print(f"\n{'='*70}")
        print("PDEBench Baseline Evaluation (L_sim metric)")
        print(f"{'='*70}")
        print(f"Checkpoint: {checkpoint_path}")

    model = load_model(checkpoint_path, model_config, accelerator.device)
    model = accelerator.prepare(model)

    eval_config = config.get('eval', {})
    all_results = []

    for ds in config.get('datasets', []):
        if not Path(ds['path']).exists():
            continue

        ds_config = EvalDatasetConfig(
            name=ds['name'],
            path=ds['path'],
            Lx=ds.get('Lx', 1.0),
            Ly=ds.get('Ly', 1.0),
            channels=ds.get('channels', []),
        )

        # Special handling for 2D CFD: split M0.1 and M1.0
        if ds['name'] == '2d_cfd':
            # Get val indices first
            rng = np.random.RandomState(eval_config.get('seed', 42))
            n_samples = 4000  # 4 files × 1000 samples
            indices = rng.permutation(n_samples)
            val_indices = indices[int(n_samples * 0.9):].tolist()

            # Split by Mach number
            m01_indices = [i for i in val_indices if i < 2000]
            m10_indices = [i for i in val_indices if i >= 2000]

            if accelerator.is_main_process:
                print(f"\n2D CFD split: M0.1={len(m01_indices)} samples, M1.0={len(m10_indices)} samples")

            # Evaluate M0.1
            if m01_indices:
                result = evaluate_dataset(model, ds_config, eval_config, accelerator,
                                         sample_indices=m01_indices, subset_name='2d_cfd_M0.1')
                if result:
                    all_results.append(result)

            # Evaluate M1.0
            if m10_indices:
                result = evaluate_dataset(model, ds_config, eval_config, accelerator,
                                         sample_indices=m10_indices, subset_name='2d_cfd_M1.0')
                if result:
                    all_results.append(result)

            # Also evaluate combined
            result = evaluate_dataset(model, ds_config, eval_config, accelerator,
                                     subset_name='2d_cfd_all')
            if result:
                all_results.append(result)
        else:
            result = evaluate_dataset(model, ds_config, eval_config, accelerator)
            if result:
                all_results.append(result)

    # Print summary
    if accelerator.is_main_process and all_results:
        print("\n" + "=" * 80)
        print("                    PDEBench L_sim SUMMARY")
        print("=" * 80)
        print(f"{'Dataset':<25} {'L_sim':<15} {'Samples':<10}")
        print("-" * 80)
        for r in all_results:
            print(f"{r['name']:<25} {r['L_sim']:<15.6e} {r['num_samples']:<10}")
        print("-" * 80)

        # Average (excluding subsets)
        main_results = [r for r in all_results if not r['name'].startswith('2d_cfd_M')]
        if main_results:
            avg = np.mean([r['L_sim'] for r in main_results])
            print(f"{'Average':<25} {avg:<15.6e}")
        print("=" * 80)


if __name__ == "__main__":
    main()
