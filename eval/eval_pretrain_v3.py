"""
Evaluation Script for PDEModelV3 on Pretrain Datasets.

Evaluates each dataset independently with PDEBench metrics.
Supports 1D and 2D datasets (V3 auto-detects from input shape).

Metrics (from metrics.py):
1. RMSE - Root Mean Square Error
2. nRMSE - Normalized RMSE
3. CSV - Conserved Variables RMSE
4. Max - Maximum Error
5. BD - Boundary RMSE
6. F_low/mid/high - Fourier Space RMSE
7. MPPnRMSE - Mean Per-Point nRMSE
8. Per-channel nRMSE

Usage:
    # Single GPU
    python eval/eval_pretrain_v3.py --config configs/pretrain_v3_s.yaml --checkpoint checkpoints_v3_s/best_tf.pt

    # Multi-GPU
    torchrun --nproc_per_node=8 eval/eval_pretrain_v3.py --config configs/pretrain_v3_s.yaml --checkpoint checkpoints_v3_s/best_tf.pt
"""

import os
import argparse
import yaml
import numpy as np
import torch
import h5py
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from tqdm import tqdm

from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader, Sampler

from pretrain.model_v3 import PDEModelV3
import eval.metrics as metrics
from eval.metrics import metric_func

# Constants
NUM_VECTOR_CHANNELS = 3
NUM_SCALAR_CHANNELS = 15
TOTAL_CHANNELS = NUM_VECTOR_CHANNELS + NUM_SCALAR_CHANNELS  # 18

# Scalar channel indices
SCALAR_INDICES = {
    'buoyancy': 0, 'concentration_rho': 1, 'concentration_u': 2, 'concentration_v': 3,
    'density': 4, 'electron_fraction': 5, 'energy': 6, 'entropy': 7, 'geometry': 8,
    'gravitational_potential': 9, 'height': 10, 'passive_tracer': 11, 'pressure': 12,
    'speed_of_sound': 13, 'temperature': 14,
}
SCALAR_NAMES = {v: k for k, v in SCALAR_INDICES.items()}


@dataclass
class EvalDatasetConfig:
    """Configuration for a single evaluation dataset."""
    name: str
    path: str
    Lx: float = 1.0
    Ly: float = 1.0
    vector_dim: int = 0  # 0=no vector, 1=1D velocity, 2=2D velocity


class SingleEvalDataset(Dataset):
    """Dataset for evaluating a single pretrained dataset. Supports 1D and 2D."""

    def __init__(
        self,
        config: EvalDatasetConfig,
        split: str = 'val',
        train_ratio: float = 0.9,
        seed: int = 42,
        temporal_length: int = 11,
        val_time_interval: int = 5,
    ):
        self.config = config
        self.path = Path(config.path)
        self.split = split
        self.train_ratio = train_ratio
        self.seed = seed
        self.temporal_length = temporal_length
        self.val_time_interval = val_time_interval

        self._load_metadata()
        self._split_samples()
        self._generate_clips()

    def _load_metadata(self):
        """Load dataset metadata and detect spatial dimensions."""
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
                vec_shape = f['vector'].shape  # [N, T, X, 3] or [N, T, H, W, 3]
                self.n_samples, self.n_timesteps = vec_shape[:2]
                # Detect spatial ndim: total dims - 3 (N, T, C)
                self.spatial_ndim = len(vec_shape) - 3  # 1 for 1D, 2 for 2D
                self.spatial_shape = vec_shape[2:-1]  # (X,) or (H, W)
            elif 'scalar' in f:
                sc_shape = f['scalar'].shape
                self.spatial_ndim = len(sc_shape) - 3
                self.spatial_shape = sc_shape[2:-1]
            else:
                raise ValueError(f"No data found in {self.path}")

        self.max_start = self.n_timesteps - self.temporal_length

    def _split_samples(self):
        """Split samples into train/val sets."""
        rng = np.random.RandomState(self.seed)
        indices = rng.permutation(self.n_samples)
        split_idx = int(self.n_samples * self.train_ratio)

        if self.split == 'train':
            self.sample_indices = indices[:split_idx].tolist()
        else:
            self.sample_indices = indices[split_idx:].tolist()

    def _generate_clips(self):
        """Generate evaluation clips."""
        self.clips = []
        for sample_idx in self.sample_indices:
            if self.max_start <= 0:
                if self.n_timesteps >= self.temporal_length:
                    self.clips.append((sample_idx, 0))
            else:
                for start_t in range(0, self.max_start + 1, self.val_time_interval):
                    self.clips.append((sample_idx, start_t))

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_idx, start_t = self.clips[idx]
        end_t = start_t + self.temporal_length

        with h5py.File(self.path, 'r') as f:
            if self.has_vector:
                vector = f['vector'][sample_idx, start_t:end_t]  # [T, *spatial, 3]
            else:
                # Create zero vector matching spatial shape
                T = self.temporal_length
                vector = np.zeros((T,) + self.spatial_shape + (NUM_VECTOR_CHANNELS,), dtype=np.float32)

            if 'scalar' in f:
                scalar_compact = f['scalar'][sample_idx, start_t:end_t]  # [T, *spatial, C_s]
                scalar_indices = f['scalar_indices'][:]
            else:
                scalar_compact = None
                scalar_indices = np.array([], dtype=np.int32)

        # Expand scalar to full 15 channels
        spatial_shape = vector.shape[:-1]  # (T, X) for 1D, (T, H, W) for 2D
        scalar_full = np.zeros(spatial_shape + (NUM_SCALAR_CHANNELS,), dtype=np.float32)
        if scalar_compact is not None:
            for i, idx_val in enumerate(scalar_indices):
                scalar_full[..., idx_val] = scalar_compact[..., i]

        # Concatenate: [T, *spatial, 18]
        data = np.concatenate([vector, scalar_full], axis=-1).astype(np.float32)

        # Channel mask
        channel_mask = np.zeros(TOTAL_CHANNELS, dtype=np.float32)
        if self.has_vector and self.config.vector_dim > 0:
            channel_mask[:self.config.vector_dim] = 1.0
        for idx_val in scalar_indices:
            channel_mask[NUM_VECTOR_CHANNELS + idx_val] = 1.0

        return {
            'data': torch.from_numpy(data),
            'channel_mask': torch.from_numpy(channel_mask),
        }


class EvalSampler(Sampler):
    """Distributed sampler for evaluation."""

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

        indices = list(range(len(dataset)))
        batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]

        # Pad for equal distribution
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
    }


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path: str, config: dict, device: torch.device) -> PDEModelV3:
    """Load PDEModelV3 from checkpoint."""
    model = PDEModelV3(config)

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Handle DDP/compile prefixes
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.removeprefix('module.').removeprefix('_orig_mod.')
        new_state_dict[k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

    model = model.float().to(device)
    model.eval()

    return model


def get_channel_name(global_idx: int) -> str:
    """Get channel name from global index."""
    vector_names = ['Vx', 'Vy', 'Vz']
    if global_idx < NUM_VECTOR_CHANNELS:
        return vector_names[global_idx]
    else:
        scalar_idx = global_idx - NUM_VECTOR_CHANNELS
        return SCALAR_NAMES.get(scalar_idx, f"scalar_{scalar_idx}")


@torch.no_grad()
def evaluate_dataset(
    model: PDEModelV3,
    dataset_config: EvalDatasetConfig,
    accelerator: Accelerator,
    t_input: int = 8,
    temporal_length: int = 11,
    batch_size: int = 4,
    seed: int = 42,
) -> Dict:
    """
    Evaluate model on a single dataset.

    Works for both 1D and 2D datasets — metric_func auto-detects from tensor shape.
    """
    if accelerator.is_main_process:
        print(f"\n{'='*70}")
        print(f"Evaluating: {dataset_config.name}")
        print(f"{'='*70}")
        print(f"Path: {dataset_config.path}")
        print(f"Domain: Lx={dataset_config.Lx}, Ly={dataset_config.Ly}")
        print(f"Vector dim: {dataset_config.vector_dim}")

    dataset = SingleEvalDataset(
        config=dataset_config,
        split='val',
        train_ratio=0.9,
        seed=seed,
        temporal_length=temporal_length,
    )

    if accelerator.is_main_process:
        print(f"Spatial ndim: {dataset.spatial_ndim}, shape: {dataset.spatial_shape}")

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
        print(f"t_input: {t_input} | temporal_length: {temporal_length}")

    all_preds = []
    all_targets = []
    channel_mask_nonzero = None
    filtered_channel_indices = None

    iterator = tqdm(dataloader, desc=f"Evaluating {dataset_config.name}",
                    disable=not accelerator.is_main_process)

    for batch in iterator:
        data = batch['data'].to(device=accelerator.device, dtype=torch.float32)
        channel_mask = batch['channel_mask'].to(device=accelerator.device)

        # Input/target split
        # 1D data: [B, T, X, 18], 2D data: [B, T, H, W, 18]
        input_data = data[:, :t_input]
        target_data = data[:, 1:t_input + 1]

        # V3 model auto-detects 1D vs 2D from ndim
        output = model(input_data)

        # Last timestep prediction
        pred_last = output[:, -1]        # [B, *spatial, 18]
        target_last = target_data[:, -1]  # [B, *spatial, 18]

        # Apply channel mask (last dim)
        valid_mask = channel_mask[0].bool()
        pred_valid = pred_last[..., valid_mask].float()
        target_valid = target_last[..., valid_mask].float()

        # Filter out all-zero channels
        if channel_mask_nonzero is None:
            # Flatten spatial dims for norm computation
            target_flat = target_valid.reshape(-1, target_valid.shape[-1])
            target_norms = target_flat.abs().mean(dim=0)
            channel_mask_nonzero = target_norms > 1e-10
            valid_indices = torch.where(valid_mask)[0]
            filtered_channel_indices = valid_indices[channel_mask_nonzero].cpu().tolist()

        pred_valid = pred_valid[..., channel_mask_nonzero]
        target_valid = target_valid[..., channel_mask_nonzero]

        all_preds.append(pred_valid.cpu())
        all_targets.append(target_valid.cpu())

    accelerator.wait_for_everyone()

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    all_preds_gathered = accelerator.gather(all_preds.to(accelerator.device))
    all_targets_gathered = accelerator.gather(all_targets.to(accelerator.device))

    all_preds_gathered = all_preds_gathered.cpu()
    all_targets_gathered = all_targets_gathered.cpu()

    results = {}

    if accelerator.is_main_process:
        # Add time dimension for metric_func:
        # 1D: [N, X, C] → [N, X, 1, C] → metric_func 1D branch
        # 2D: [N, H, W, C] → [N, H, W, 1, C] → metric_func 2D branch
        all_preds_gathered = all_preds_gathered.unsqueeze(-2)
        all_targets_gathered = all_targets_gathered.unsqueeze(-2)

        N = all_preds_gathered.shape[0]
        print(f"Computing metrics on {N} predictions (shape: {all_preds_gathered.shape})...")

        # Force CPU computation
        original_device = metrics.device
        metrics.device = torch.device('cpu')

        # Compute metrics in batches
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
            batch_preds = all_preds_gathered[i:i + metrics_batch_size]
            batch_targets = all_targets_gathered[i:i + metrics_batch_size]

            err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F, err_MPPnRMSE = metric_func(
                batch_preds.float(),
                batch_targets.float(),
                if_mean=True,
                Lx=dataset_config.Lx,
                Ly=dataset_config.Ly,
                Lz=1.0,
                iLow=4, iHigh=12,
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

        results = {
            'name': dataset_config.name,
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
        per_channel_nRMSE = {}

        for ch_idx in range(C_valid):
            ch_nRMSE_sum = 0.0
            ch_batches = 0
            for i in range(0, N, metrics_batch_size):
                batch_preds = all_preds_gathered[i:i + metrics_batch_size, ..., ch_idx:ch_idx + 1]
                batch_targets = all_targets_gathered[i:i + metrics_batch_size, ..., ch_idx:ch_idx + 1]

                _, ch_nRMSE, _, _, _, _, _ = metric_func(
                    batch_preds.float(),
                    batch_targets.float(),
                    if_mean=True,
                    Lx=dataset_config.Lx,
                    Ly=dataset_config.Ly,
                    Lz=1.0,
                    iLow=4, iHigh=12,
                    initial_step=0,
                )
                ch_nRMSE_sum += ch_nRMSE.item()
                ch_batches += 1

            if filtered_channel_indices is not None and ch_idx < len(filtered_channel_indices):
                global_idx = filtered_channel_indices[ch_idx]
                ch_name = get_channel_name(global_idx)
            else:
                ch_name = f"ch_{ch_idx}"
            per_channel_nRMSE[ch_name] = ch_nRMSE_sum / ch_batches

        results['per_channel_nRMSE'] = per_channel_nRMSE

        metrics.device = original_device

        # Print results
        print(f"\n{'-'*50}")
        print(f"Results for {dataset_config.name}:")
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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PDEModelV3")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--datasets', type=str, nargs='*', default=None,
                        help='Specific datasets to evaluate (default: all)')
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    accelerator = Accelerator()

    if accelerator.is_main_process:
        print(f"\n{'='*70}")
        print("PDEModelV3 Evaluation (per-dataset)")
        print(f"{'='*70}")
        print(f"Config: {args.config}")
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Device: {accelerator.device}")
        print(f"Num processes: {accelerator.num_processes}")

    if not Path(args.checkpoint).exists():
        if accelerator.is_main_process:
            print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        return

    if accelerator.is_main_process:
        print("\nLoading model...")

    model = load_model(args.checkpoint, config, accelerator.device)

    if accelerator.is_main_process:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,}")

    t_input = config['dataset'].get('t_input', 8)
    temporal_length = t_input + 1
    seed = config['dataset'].get('seed', 42)
    data_dir = Path(config['dataset']['path'])

    # Dataset configs: 1D + 2D (matching pretrain datasets)
    dataset_configs = [
        EvalDatasetConfig(
            name='1d_cfd',
            path=str(data_dir / 'pretrained' / '1D_CFD_unified.hdf5'),
            Lx=1.0,
            vector_dim=1,  # 1D velocity (Vx)
        ),
        EvalDatasetConfig(
            name='diffusion_reaction',
            path=str(data_dir / 'pretrained' / '2D_diff-react_NA_NA.hdf5'),
            Lx=1.0, Ly=1.0,
            vector_dim=0,
        ),
        EvalDatasetConfig(
            name='2d_cfd',
            path=str(data_dir / 'pretrained' / '2D_CFD_128_merged.hdf5'),
            Lx=1.0, Ly=1.0,
            vector_dim=2,
        ),
        EvalDatasetConfig(
            name='swe',
            path=str(data_dir / 'pretrained' / '2D_rdb_NA_NA.h5'),
            Lx=1.0, Ly=1.0,
            vector_dim=0,
        ),
    ]

    # Filter by --datasets flag if provided
    if args.datasets:
        dataset_configs = [c for c in dataset_configs if c.name in args.datasets]

    # Filter existing datasets
    valid_configs = []
    for ds_config in dataset_configs:
        if Path(ds_config.path).exists():
            valid_configs.append(ds_config)
        elif accelerator.is_main_process:
            print(f"WARNING: Dataset not found: {ds_config.path}")

    if not valid_configs:
        if accelerator.is_main_process:
            print("ERROR: No datasets found!")
        return

    # Evaluate each dataset
    all_results = []
    for ds_config in valid_configs:
        results = evaluate_dataset(
            model=model,
            dataset_config=ds_config,
            accelerator=accelerator,
            t_input=t_input,
            temporal_length=temporal_length,
            batch_size=args.batch_size,
            seed=seed,
        )
        if accelerator.is_main_process and results:
            all_results.append(results)

    # Print summary
    if accelerator.is_main_process and all_results:
        print("\n" + "=" * 100)
        print("                              EVALUATION SUMMARY")
        print("=" * 100)
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Datasets evaluated: {len(all_results)}")
        print()

        header = f"{'Dataset':<20} {'nRMSE':<12} {'MPPnRMSE':<12} {'RMSE':<12} {'Max':<12} {'CSV':<12} {'BD':<12} {'Samples':<8}"
        print(header)
        print("-" * 100)
        for r in all_results:
            row = f"{r['name']:<20} {r['nRMSE']:<12.4e} {r['MPPnRMSE']:<12.4e} {r['RMSE']:<12.4e} {r['Max']:<12.4e} {r['CSV']:<12.4e} {r['BD']:<12.4e} {r['num_samples']:<8}"
            print(row)
        print("-" * 100)

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


if __name__ == "__main__":
    main()
