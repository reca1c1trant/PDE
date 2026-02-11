"""
Unified Dataset for PDE Foundation Model Pretraining.

Supports multiple datasets with different formats:
- diffusion-reaction: 128x128, scalar only (concentration_u, concentration_v)
- 2D_CFD: 128x128, vector (Vx, Vy) + scalar (density, pressure)
- SWE: 128x128, scalar only (height)
- NS_incom: 512x512, vector (vx, vy) + scalar (passive_tracer)

Data Format (after preprocessing):
    dataset.hdf5
    ├── vector: [N, T, H, W, 3]          # optional
    ├── scalar: [N, T, H, W, C_actual]   # compact storage
    └── scalar_indices: [C_actual]        # global index mapping

Output Format:
    data: [T, H, W, 18]      # 3 vector + 15 scalar (unified)
    channel_mask: [18]       # valid channel indicator
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
NUM_VECTOR_CHANNELS = 3
NUM_SCALAR_CHANNELS = 15
TOTAL_CHANNELS = NUM_VECTOR_CHANNELS + NUM_SCALAR_CHANNELS  # 18
DEFAULT_TEMPORAL_LENGTH = 11  # V2: T_input=8 + num_steps=3 for AR rollout


@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""
    name: str
    path: str
    val_time_interval: int = 2  # Time interval for validation clips
    vector_dim: int = 0  # Actual vector dimensions (0=no vector, 2=2D, 3=3D)
    clips_per_epoch: Optional[int] = None  # If set, override clips_ratio with fixed number per epoch


# Scalar channel indices (from scalar_channel.csv)
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


class SingleDataset:
    """
    Handler for a single dataset file.

    Manages sample indexing, clip generation, and data loading.
    """

    def __init__(
        self,
        config: DatasetConfig,
        split: str,
        train_ratio: float = 0.9,
        seed: int = 42,
        clips_ratio: float = 0.2,
        temporal_length: int = DEFAULT_TEMPORAL_LENGTH,
    ):
        self.config = config
        self.split = split
        self.train_ratio = train_ratio
        self.seed = seed
        self.temporal_length = temporal_length
        self.clips_ratio = clips_ratio
        self.val_time_interval = config.val_time_interval
        self.path = Path(config.path)

        # Load metadata
        self._load_metadata()
        self._split_samples()

    def _load_metadata(self):
        """Load dataset metadata without loading all data."""
        with h5py.File(self.path, 'r') as f:
            # Get dimensions
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

            # Spatial size
            if 'scalar' in f:
                self.spatial_size = f['scalar'].shape[2]  # H
            elif self.has_vector:
                self.spatial_size = f['vector'].shape[2]
            else:
                raise ValueError(f"No data found in {self.path}")

        self.max_start = self.n_timesteps - self.temporal_length

        logger.info(f"{self.config.name}: {self.n_samples} samples, T={self.n_timesteps}, "
                   f"spatial={self.spatial_size}, temporal_len={self.temporal_length}, max_start={self.max_start}")

    def _split_samples(self):
        """Split samples into train/val sets."""
        rng = np.random.RandomState(self.seed)
        indices = rng.permutation(self.n_samples)
        split_idx = int(self.n_samples * self.train_ratio)

        if self.split == 'train':
            self.sample_indices = indices[:split_idx].tolist()
        else:
            self.sample_indices = indices[split_idx:].tolist()

        logger.info(f"{self.config.name} ({self.split}): {len(self.sample_indices)} samples")

    def generate_clips(self, epoch: int = 0) -> List[Tuple[int, int]]:
        """
        Generate clips for current epoch.

        Returns:
            List of (sample_idx, start_t) tuples.
        """
        rng = np.random.RandomState(self.seed + epoch)
        clips = []

        for sample_idx in self.sample_indices:
            if self.split == 'train':
                time_clips = self._get_train_time_clips(rng)
            else:
                time_clips = self._get_val_time_clips()

            for start_t in time_clips:
                clips.append((sample_idx, start_t))

        return clips

    def _get_train_time_clips(self, rng: np.random.RandomState) -> List[int]:
        """Get time clip starting points for training (20% sampling)."""
        if self.max_start <= 0:
            return [0] if self.n_timesteps >= self.temporal_length else []

        if self.n_timesteps <= 30:
            # Short sequence: use all
            return list(range(self.max_start + 1))
        else:
            # Long sequence: sample 20%
            n_clips = max(1, int(self.clips_ratio * (self.max_start + 1)))
            return rng.choice(self.max_start + 1, n_clips, replace=False).tolist()

    def _get_val_time_clips(self) -> List[int]:
        """Get time clip starting points for validation (fixed interval)."""
        if self.max_start <= 0:
            return [0] if self.n_timesteps >= self.temporal_length else []

        if self.n_timesteps <= 30:
            # Short sequence (e.g., 2D_CFD): use all
            return list(range(self.max_start + 1))
        else:
            # Use per-dataset validation interval
            return list(range(0, self.max_start + 1, self.val_time_interval))

    def load_clip(self, sample_idx: int, start_t: int) -> Dict[str, np.ndarray]:
        """
        Load a single clip from the dataset.

        Returns:
            data: [T, H, W, 18] unified format
            channel_mask: [18] valid channel indicator
        """
        end_t = start_t + self.temporal_length

        with h5py.File(self.path, 'r') as f:
            # Load vector if exists
            if self.has_vector:
                vector = f['vector'][sample_idx, start_t:end_t]  # [T, H, W, 3]
            else:
                # Create zero vector
                if 'scalar' in f:
                    _, _, H, W, _ = f['scalar'].shape
                else:
                    H, W = 128, 128
                vector = np.zeros((self.temporal_length, H, W, NUM_VECTOR_CHANNELS), dtype=np.float32)

            # Load scalar
            if 'scalar' in f:
                scalar_compact = f['scalar'][sample_idx, start_t:end_t]  # [T, H, W, C_actual]
                scalar_indices = f['scalar_indices'][:]
            else:
                scalar_compact = None
                scalar_indices = np.array([], dtype=np.int32)

        # Expand scalar to full 15 channels
        T, H, W = vector.shape[:3]
        scalar_full = np.zeros((T, H, W, NUM_SCALAR_CHANNELS), dtype=np.float32)
        if scalar_compact is not None:
            for i, idx in enumerate(scalar_indices):
                scalar_full[..., idx] = scalar_compact[..., i]

        # Concatenate to unified format: [T, H, W, 18]
        data = np.concatenate([vector, scalar_full], axis=-1).astype(np.float32)

        # Create channel mask
        channel_mask = np.zeros(TOTAL_CHANNELS, dtype=np.float32)
        if self.has_vector and self.config.vector_dim > 0:
            channel_mask[:self.config.vector_dim] = 1.0
        for idx in scalar_indices:
            channel_mask[NUM_VECTOR_CHANNELS + idx] = 1.0

        return {
            'data': data,
            'channel_mask': channel_mask,
            'dataset_name': self.config.name,
        }


class PretrainDataset(Dataset):
    """
    Unified dataset for pretraining across multiple PDE datasets.

    Combines clips from all datasets into a single indexable dataset.
    """

    def __init__(
        self,
        dataset_configs: List[DatasetConfig],
        split: str = 'train',
        train_ratio: float = 0.9,
        seed: int = 42,
        clips_ratio: float = 0.2,
        temporal_length: int = DEFAULT_TEMPORAL_LENGTH,
    ):
        self.split = split
        self.seed = seed
        self.epoch = 0
        self.temporal_length = temporal_length

        # Initialize individual datasets
        self.datasets: Dict[str, SingleDataset] = {}
        for config in dataset_configs:
            self.datasets[config.name] = SingleDataset(
                config=config,
                split=split,
                train_ratio=train_ratio,
                seed=seed,
                clips_ratio=clips_ratio,
                temporal_length=temporal_length,
            )

        # Generate initial clips
        self._generate_all_clips()

        # Log statistics
        total_clips = len(self.clips)
        logger.info(f"PretrainDataset ({split}): {total_clips} total clips")
        for name, ds in self.datasets.items():
            ds_clips = sum(1 for c in self.clips if c[0] == name)
            logger.info(f"  {name}: {ds_clips} clips ({100*ds_clips/total_clips:.1f}%)")

    def _generate_all_clips(self):
        """Generate clips from all datasets."""
        self.clips = []  # List of (dataset_name, sample_idx, start_t)
        self.clips_by_sample: Dict[Tuple[str, int], List[int]] = {}

        clip_idx = 0
        rng = np.random.RandomState(self.seed + self.epoch)

        for name, ds in self.datasets.items():
            ds_clips = ds.generate_clips(self.epoch)

            # Apply clips_per_epoch limit if configured
            clips_per_epoch = ds.config.clips_per_epoch
            if clips_per_epoch is not None and len(ds_clips) > clips_per_epoch:
                # Randomly sample clips_per_epoch clips
                selected_indices = rng.choice(len(ds_clips), clips_per_epoch, replace=False)
                ds_clips = [ds_clips[i] for i in selected_indices]
                logger.info(f"  {name}: limited to {clips_per_epoch} clips per epoch")

            for sample_idx, start_t in ds_clips:
                self.clips.append((name, sample_idx, start_t))

                # Group by sample (dataset, sample)
                sample_key = (name, sample_idx)
                if sample_key not in self.clips_by_sample:
                    self.clips_by_sample[sample_key] = []
                self.clips_by_sample[sample_key].append(clip_idx)

                clip_idx += 1

    def set_epoch(self, epoch: int):
        """Set epoch for reproducible shuffling."""
        self.epoch = epoch
        self._generate_all_clips()

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        dataset_name, sample_idx, start_t = self.clips[idx]
        ds = self.datasets[dataset_name]

        result = ds.load_clip(sample_idx, start_t)

        return {
            'data': torch.from_numpy(result['data']),
            'channel_mask': torch.from_numpy(result['channel_mask']),
            'dataset_name': result['dataset_name'],
        }


class PretrainSampler(Sampler):
    """
    Sampler for pretraining with flexible batching strategy.

    For equivalent samples with enough clips (>= batch_size):
        - Same-equivalent-sample constraint (same physical simulation per batch)

    For equivalent samples with few clips (< batch_size):
        - Merge multiple samples' clips into batches (must be multiple of 4)
        - This handles short-sequence datasets like CFD (only 5 clips per sample)
    """

    def __init__(
        self,
        dataset: PretrainDataset,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        min_batch_unit: int = 4,  # Batches must be multiple of this
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.min_batch_unit = min_batch_unit

        # Distributed settings
        import os
        if num_replicas is None:
            num_replicas = int(os.environ.get('WORLD_SIZE', 1))
        if rank is None:
            rank = int(os.environ.get('RANK', 0))

        self.num_replicas = num_replicas
        self.rank = rank

        self._compute_batches()

    def _compute_batches(self):
        """Compute batches with flexible strategy for small equiv samples."""
        rng = np.random.RandomState(self.seed + self.epoch)

        # Get all equivalent sample keys and shuffle
        equiv_keys = list(self.dataset.clips_by_sample.keys())
        if self.shuffle:
            rng.shuffle(equiv_keys)

        batches = []
        small_clips_pool = []  # Pool for merging small equiv samples

        for equiv_key in equiv_keys:
            clip_indices = list(self.dataset.clips_by_sample[equiv_key])

            if len(clip_indices) == 0:
                continue

            # Shuffle clips within this equivalent sample
            if self.shuffle:
                rng.shuffle(clip_indices)

            if len(clip_indices) >= self.batch_size:
                # Large enough: use same-equiv-sample batching with padding
                for i in range(0, len(clip_indices), self.batch_size):
                    batch = clip_indices[i:i + self.batch_size]

                    # Pad incomplete batch from same equiv sample
                    if len(batch) < self.batch_size:
                        need = self.batch_size - len(batch)
                        pool = clip_indices[:i] if i > 0 else batch
                        if len(pool) >= need:
                            pad = rng.choice(pool, need, replace=False).tolist()
                        else:
                            pad = rng.choice(pool, need, replace=True).tolist()
                        batch = batch + pad

                    batches.append(batch)
            else:
                # Too few clips: add to pool for merging
                small_clips_pool.extend(clip_indices)

        # Process small clips pool: merge into batches
        if small_clips_pool:
            if self.shuffle:
                rng.shuffle(small_clips_pool)

            for i in range(0, len(small_clips_pool), self.batch_size):
                batch = small_clips_pool[i:i + self.batch_size]

                # Only keep if batch size is multiple of min_batch_unit
                if len(batch) >= self.min_batch_unit:
                    # Trim to multiple of min_batch_unit if needed
                    usable_size = (len(batch) // self.min_batch_unit) * self.min_batch_unit
                    batch = batch[:usable_size]

                    # Pad to full batch_size if close enough
                    if len(batch) < self.batch_size and len(batch) >= self.batch_size // 2:
                        need = self.batch_size - len(batch)
                        pad = rng.choice(batch, need, replace=True).tolist()
                        batch = batch + pad

                    if len(batch) > 0:
                        batches.append(batch)

        # Distribute across ranks with padding to ensure ALL ranks get EQUAL batches
        # This is critical for DDP - unequal batch counts cause NCCL collective timeouts
        total_batches = len(batches)

        if total_batches == 0:
            self.num_batches_per_rank = 0
            self._all_batches = []
            if self.rank == 0:
                logger.warning("PretrainSampler: No batches generated!")
            return

        # IMPORTANT: Create a copy to avoid modifying original list during padding
        self._all_batches = list(batches)

        # Pad to make evenly divisible by num_replicas
        remainder = total_batches % self.num_replicas
        padding_count = 0
        if remainder > 0:
            need = self.num_replicas - remainder
            for i in range(need):
                # Repeat batches cyclically for padding (use original batches list)
                self._all_batches.append(batches[i % total_batches])
            padding_count = need

        self.num_batches_per_rank = len(self._all_batches) // self.num_replicas

        if self.rank == 0:
            logger.info(f"PretrainSampler: {total_batches} batches (+{padding_count} padding), "
                       f"{self.num_batches_per_rank}/rank, batch_size={self.batch_size}")

    def set_epoch(self, epoch: int):
        """Set epoch and regenerate batches."""
        self.epoch = epoch
        self.dataset.set_epoch(epoch)
        self._compute_batches()

    def __iter__(self):
        start_idx = self.rank * self.num_batches_per_rank
        end_idx = start_idx + self.num_batches_per_rank

        for batch in self._all_batches[start_idx:end_idx]:
            yield batch

    def __len__(self) -> int:
        return self.num_batches_per_rank


def pretrain_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for pretraining dataset."""
    return {
        'data': torch.stack([item['data'] for item in batch], dim=0),
        'channel_mask': torch.stack([item['channel_mask'] for item in batch], dim=0),
        'dataset_names': [item['dataset_name'] for item in batch],
    }


def create_pretrain_dataloaders(
    data_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
    dataset_overrides: Optional[Dict[str, Dict]] = None,
    temporal_length: int = DEFAULT_TEMPORAL_LENGTH,
    clips_ratio: float = 0.25,
):
    """
    Create train and validation dataloaders for pretraining.

    Args:
        data_dir: Directory containing pretrained/*.hdf5 files
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        seed: Random seed
        dataset_overrides: Optional dict to override dataset configs from YAML
            Example: {'2d_cfd': {'clips_per_epoch': 5000}, 'ns_incom': {'clips_per_epoch': 3000}}
        temporal_length: Number of timesteps per clip (t_input + num_steps)
        clips_ratio: Temporal sampling ratio for training (e.g., 0.25 = 25%)

    Returns:
        train_loader, val_loader, train_sampler, val_sampler
    """
    data_dir = Path(data_dir)
    dataset_overrides = dataset_overrides or {}

    # Define dataset configurations
    # vector_dim: 0=no vector, 2=2D velocity (Vx,Vy), 3=3D velocity (Vx,Vy,Vz)
    configs = [
        DatasetConfig(
            name='diffusion_reaction',
            path=str(data_dir / 'pretrained' / '2D_diff-react_NA_NA.hdf5'),
            val_time_interval=2,
            vector_dim=0,  # No vector, only scalar (concentration_u, concentration_v)
        ),
        DatasetConfig(
            name='2d_cfd',
            path=str(data_dir / 'pretrained' / '2D_CFD_128_merged.hdf5'),
            val_time_interval=2,
            vector_dim=2,  # 2D velocity (Vx, Vy)
        ),
        DatasetConfig(
            name='swe',
            path=str(data_dir / 'pretrained' / '2D_rdb_NA_NA.h5'),
            val_time_interval=2,
            vector_dim=0,  # No vector, only scalar (height)
        ),
        DatasetConfig(
            name='ns_incom',
            path='/scratch-share/SONG0304/pretrained/ns_incom_inhom_2d_512.hdf5',
            val_time_interval=8,
            vector_dim=2,  # 2D velocity (vx, vy)
        ),
    ]

    # Apply overrides from YAML config
    for config in configs:
        if config.name in dataset_overrides:
            overrides = dataset_overrides[config.name]
            for key, value in overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                    logger.info(f"  {config.name}: override {key}={value}")

    # Filter to existing datasets
    existing_configs = []
    for config in configs:
        if Path(config.path).exists():
            existing_configs.append(config)
            logger.info(f"Found dataset: {config.name}")
        else:
            logger.warning(f"Dataset not found: {config.path}")

    if not existing_configs:
        raise ValueError(f"No datasets found in {data_dir}")

    # Create datasets
    train_dataset = PretrainDataset(
        dataset_configs=existing_configs,
        split='train',
        train_ratio=0.9,
        seed=seed,
        clips_ratio=clips_ratio,
        temporal_length=temporal_length,
    )

    val_dataset = PretrainDataset(
        dataset_configs=existing_configs,
        split='val',
        train_ratio=0.9,
        seed=seed,
        clips_ratio=clips_ratio,  # Not used for validation (uses fixed intervals)
        temporal_length=temporal_length,
    )

    # Create samplers
    train_sampler = PretrainSampler(train_dataset, batch_size, shuffle=True, seed=seed)
    val_sampler = PretrainSampler(val_dataset, batch_size, shuffle=False, seed=seed)

    # Create dataloaders
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=pretrain_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=pretrain_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, train_sampler, val_sampler


if __name__ == "__main__":
    """Test the dataset."""
    import sys

    data_dir = "./data"

    print("=" * 60)
    print("Testing PretrainDataset")
    print("=" * 60)

    pretrained_dir = Path(data_dir) / 'pretrained'
    if pretrained_dir.exists():
        try:
            train_loader, val_loader, train_sampler, val_sampler = create_pretrain_dataloaders(
                data_dir=data_dir,
                batch_size=4,
                num_workers=0,
            )

            print(f"\nTrain batches: {len(train_loader)}")
            print(f"Val batches: {len(val_loader)}")

            # Test one batch
            for batch in train_loader:
                print(f"\nBatch shapes:")
                print(f"  data: {batch['data'].shape}")
                print(f"  channel_mask: {batch['channel_mask'].shape}")
                print(f"  dataset_names: {batch['dataset_names']}")
                break

        except Exception as e:
            print(f"Could not load datasets: {e}")
    else:
        print(f"\nNo pretrained data found at {pretrained_dir}")
        print("Run preprocessing scripts first.")

    print("\nDataset test complete!")
