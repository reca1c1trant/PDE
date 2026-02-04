"""
Unified Dataset for PDE Foundation Model Pretraining.

Supports multiple datasets with different formats:
- diffusion-reaction: scalar only (concentration_u, concentration_v)
- 2D_CFD: vector (Vx, Vy) + scalar (density, pressure)
- SWE: scalar only (height)
- NS_incom: vector (vx, vy) + scalar (passive_tracer), 512x512 -> crop to 128x128

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
TEMPORAL_LENGTH = 17  # 16 input + 1 for causal AR


@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""
    name: str
    path: str
    spatial_size: int  # 128 or 512
    needs_crop: bool   # True for 512x512 datasets
    train_spatial_points: int  # number of spatial crop points for training
    val_spatial_points: int    # number of spatial crop points for validation


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


def generate_spatial_points(
    n_points: int,
    max_coord: int = 384,
    rng: np.random.RandomState = None
) -> List[Tuple[int, int]]:
    """
    Generate n spatial crop starting points using deterministic grid.

    For n_points=11: use ~3.3x3.3 grid, take first 11 points
    For n_points=16: use 4x4 grid

    Args:
        n_points: Number of points to generate
        max_coord: Maximum coordinate value (crop starts in [0, max_coord])
        rng: Not used, kept for API compatibility

    Returns:
        List of (x, y) tuples with spacing >= 128
    """
    # Determine grid size
    grid_size = int(np.ceil(np.sqrt(n_points)))
    step = max_coord // (grid_size - 1) if grid_size > 1 else 0

    points = []
    for i in range(grid_size):
        for j in range(grid_size):
            if len(points) < n_points:
                points.append((i * step, j * step))

    return points


def get_validation_spatial_points(n_points: int = 16, max_coord: int = 384) -> List[Tuple[int, int]]:
    """
    Generate fixed grid of spatial points for validation.

    For 16 points: 4x4 grid at (0, 128, 256, 384)
    """
    grid_size = int(np.sqrt(n_points))
    step = max_coord // (grid_size - 1) if grid_size > 1 else 0

    points = []
    for i in range(grid_size):
        for j in range(grid_size):
            points.append((i * step, j * step))

    return points[:n_points]


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
    ):
        self.config = config
        self.split = split
        self.train_ratio = train_ratio
        self.seed = seed
        self.clips_ratio = clips_ratio

        self.path = Path(config.path)
        self.needs_crop = config.needs_crop
        self.crop_size = 128

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

        self.max_start = self.n_timesteps - TEMPORAL_LENGTH

        logger.info(f"{self.config.name}: {self.n_samples} samples, T={self.n_timesteps}, "
                   f"spatial={self.spatial_size}, max_start={self.max_start}")

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

    def generate_clips(self, epoch: int = 0) -> List[Tuple[int, int, Optional[Tuple[int, int]]]]:
        """
        Generate clips for current epoch.

        Returns:
            List of (sample_idx, start_t, spatial_point) tuples.
            spatial_point is None for 128x128 datasets, (x, y) for 512x512.
        """
        rng = np.random.RandomState(self.seed + epoch)
        clips = []

        for sample_idx in self.sample_indices:
            # Determine time clips
            if self.split == 'train':
                time_clips = self._get_train_time_clips(rng)
            else:
                time_clips = self._get_val_time_clips()

            # Determine spatial points
            if self.needs_crop:
                if self.split == 'train':
                    spatial_points = generate_spatial_points(
                        self.config.train_spatial_points,
                        max_coord=self.spatial_size - self.crop_size,
                        rng=rng
                    )
                else:
                    spatial_points = get_validation_spatial_points(
                        self.config.val_spatial_points,
                        max_coord=self.spatial_size - self.crop_size
                    )
            else:
                spatial_points = [None]

            # Generate all combinations
            for start_t in time_clips:
                for sp in spatial_points:
                    clips.append((sample_idx, start_t, sp))

        return clips

    def _get_train_time_clips(self, rng: np.random.RandomState) -> List[int]:
        """Get time clip starting points for training (20% sampling)."""
        if self.max_start <= 0:
            return [0] if self.n_timesteps >= TEMPORAL_LENGTH else []

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
            return [0] if self.n_timesteps >= TEMPORAL_LENGTH else []

        if self.n_timesteps <= 30:
            # Short sequence (e.g., 2D_CFD): use all
            return list(range(self.max_start + 1))
        elif self.needs_crop:
            # NS_incom: interval of 10
            return list(range(0, self.max_start + 1, 10))
        else:
            # diffusion-reaction, SWE: interval of 5
            return list(range(0, self.max_start + 1, 5))

    def load_clip(
        self,
        sample_idx: int,
        start_t: int,
        spatial_point: Optional[Tuple[int, int]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Load a single clip from the dataset.

        Returns:
            data: [T, H, W, 18] unified format
            channel_mask: [18] valid channel indicator
        """
        end_t = start_t + TEMPORAL_LENGTH

        with h5py.File(self.path, 'r') as f:
            # Load vector if exists
            if self.has_vector:
                vector = f['vector'][sample_idx, start_t:end_t]  # [T, H, W, 3]
            else:
                # Create zero vector
                if 'scalar' in f:
                    T, H, W = f['scalar'].shape[1:4]
                    H, W = min(H, 128), min(W, 128)  # Handle crop size
                else:
                    T, H, W = TEMPORAL_LENGTH, 128, 128
                vector = np.zeros((TEMPORAL_LENGTH, H, W, NUM_VECTOR_CHANNELS), dtype=np.float32)

            # Load scalar
            if 'scalar' in f:
                scalar_compact = f['scalar'][sample_idx, start_t:end_t]  # [T, H, W, C_actual]
                scalar_indices = f['scalar_indices'][:]
            else:
                scalar_compact = None
                scalar_indices = np.array([], dtype=np.int32)

        # Apply spatial crop if needed
        if spatial_point is not None:
            x, y = spatial_point
            vector = vector[:, x:x+self.crop_size, y:y+self.crop_size, :]
            if scalar_compact is not None:
                scalar_compact = scalar_compact[:, x:x+self.crop_size, y:y+self.crop_size, :]

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
        if self.has_vector:
            channel_mask[:NUM_VECTOR_CHANNELS] = 1.0  # vector channels
        for idx in scalar_indices:
            channel_mask[NUM_VECTOR_CHANNELS + idx] = 1.0  # scalar channels

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
    ):
        self.split = split
        self.seed = seed
        self.epoch = 0

        # Initialize individual datasets
        self.datasets: Dict[str, SingleDataset] = {}
        for config in dataset_configs:
            self.datasets[config.name] = SingleDataset(
                config=config,
                split=split,
                train_ratio=train_ratio,
                seed=seed,
                clips_ratio=clips_ratio,
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
        self.clips = []  # List of (dataset_name, sample_idx, start_t, spatial_point)
        self.clips_by_equiv_sample: Dict[Tuple[str, int, Optional[Tuple[int, int]]], List[int]] = {}

        clip_idx = 0
        for name, ds in self.datasets.items():
            ds_clips = ds.generate_clips(self.epoch)

            for sample_idx, start_t, spatial_point in ds_clips:
                self.clips.append((name, sample_idx, start_t, spatial_point))

                # Group by equivalent sample (dataset, sample, spatial_point)
                equiv_key = (name, sample_idx, spatial_point)
                if equiv_key not in self.clips_by_equiv_sample:
                    self.clips_by_equiv_sample[equiv_key] = []
                self.clips_by_equiv_sample[equiv_key].append(clip_idx)

                clip_idx += 1

    def set_epoch(self, epoch: int):
        """Set epoch for reproducible shuffling."""
        self.epoch = epoch
        self._generate_all_clips()

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        dataset_name, sample_idx, start_t, spatial_point = self.clips[idx]
        ds = self.datasets[dataset_name]

        result = ds.load_clip(sample_idx, start_t, spatial_point)

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
        equiv_keys = list(self.dataset.clips_by_equiv_sample.keys())
        if self.shuffle:
            rng.shuffle(equiv_keys)

        batches = []
        small_clips_pool = []  # Pool for merging small equiv samples

        for equiv_key in equiv_keys:
            clip_indices = list(self.dataset.clips_by_equiv_sample[equiv_key])

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
):
    """
    Create train and validation dataloaders for pretraining.

    Args:
        data_dir: Directory containing pretrained/*.hdf5 files
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        seed: Random seed

    Returns:
        train_loader, val_loader, train_sampler, val_sampler
    """
    data_dir = Path(data_dir)

    # Define dataset configurations
    configs = [
        DatasetConfig(
            name='diffusion_reaction',
            path=str(data_dir / 'pretrained' / '2D_diff-react_NA_NA.hdf5'),
            spatial_size=128,
            needs_crop=False,
            train_spatial_points=1,
            val_spatial_points=1,
        ),
        DatasetConfig(
            name='2d_cfd',
            path=str(data_dir / 'pretrained' / '2D_CFD_128_merged.hdf5'),
            spatial_size=128,
            needs_crop=False,
            train_spatial_points=1,
            val_spatial_points=1,
        ),
        DatasetConfig(
            name='swe',
            path=str(data_dir / 'pretrained' / '2D_rdb_NA_NA.h5'),
            spatial_size=128,
            needs_crop=False,
            train_spatial_points=1,
            val_spatial_points=1,
        ),
        DatasetConfig(
            name='ns_incom',
            path=str(data_dir / 'pretrained' / 'ns_incom_inhom_2d_512_merged.hdf5'),
            spatial_size=512,
            needs_crop=True,
            train_spatial_points=11,
            val_spatial_points=16,
        ),
    ]

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
        clips_ratio=0.2,
    )

    val_dataset = PretrainDataset(
        dataset_configs=existing_configs,
        split='val',
        train_ratio=0.9,
        seed=seed,
        clips_ratio=0.2,  # Not used for validation
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

    # Test with mock data directory
    data_dir = "./data"

    print("=" * 60)
    print("Testing PretrainDataset")
    print("=" * 60)

    # Test spatial point generation
    print("\nTesting spatial point generation:")
    points = generate_spatial_points(11, max_coord=384)
    print(f"Generated {len(points)} points: {points}")

    # Verify distances
    for i, (x1, y1) in enumerate(points):
        for j, (x2, y2) in enumerate(points):
            if i < j:
                dist = max(abs(x1 - x2), abs(y1 - y2))
                if dist < 128:
                    print(f"WARNING: Points {i} and {j} too close: {dist}")

    print("\nValidation spatial points (4x4 grid):")
    val_points = get_validation_spatial_points(16, max_coord=384)
    print(val_points)

    # Test with actual data if available
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
