"""
Finetune Dataset for PDE Foundation Model V2.

Supports new HDF5 format:
- vector: [N, T, H, W, 3] with Vz=0 for 2D
- scalar: [N, T, H, W, C_s] (optional, can be empty)
- scalar_indices: array of scalar channel indices

Output format matches pretrain:
- data: [T, H, W, 18] (3 vector + 15 scalar)
- channel_mask: [18]
- vector_dim: 0/2/3

Sampling Strategy:
- Each sample contributes K clips per epoch
- Same-sample batching for consistent boundary conditions (PDE loss)
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants matching pretrain
NUM_VECTOR_CHANNELS = 3
NUM_SCALAR_CHANNELS = 15
TOTAL_CHANNELS = NUM_VECTOR_CHANNELS + NUM_SCALAR_CHANNELS  # 18


class FinetuneDataset(Dataset):
    """
    Dataset for finetuning on specific PDE problems.

    Supports both old (per-sample groups) and new (global arrays) HDF5 formats.
    """

    def __init__(
        self,
        data_path: str,
        temporal_length: int = 9,
        split: str = 'train',
        train_ratio: float = 0.9,
        seed: int = 42,
        clips_per_sample: Optional[int] = 100,
        vector_dim: int = 2,  # 2 for 2D velocity, 3 for 3D
        val_time_interval: int = 8,  # Interval for validation clips (0, 8, 16, ...)
    ):
        """
        Args:
            data_path: Path to HDF5 file
            temporal_length: Number of timesteps per clip
            split: 'train' or 'val'
            train_ratio: Fraction for training
            seed: Random seed
            clips_per_sample: Clips per sample per epoch (None = all for train, interval for val)
            vector_dim: 2 for 2D velocity (Vz=0), 3 for 3D
            val_time_interval: Time interval for validation clip sampling (e.g., 8 = 0,8,16,...)
        """
        super().__init__()
        self.data_path = Path(data_path)
        self.temporal_length = temporal_length
        self.split = split
        self.train_ratio = train_ratio
        self.seed = seed
        self.clips_per_sample = clips_per_sample
        self.vector_dim = vector_dim
        self.val_time_interval = val_time_interval

        # Detect format and load metadata
        self._detect_format()
        self._split_dataset()

        # Generate initial clips
        self.epoch = 0
        self._generate_clips()

        if split == 'val':
            clips_info = f"interval={val_time_interval} (~{len(self.clips)//max(1,len(self.sample_indices))}/sample)"
        else:
            clips_info = "all" if clips_per_sample is None else f"{clips_per_sample}/sample"
        logger.info(f"FinetuneDataset ({split}): {len(self.sample_indices)} samples, "
                   f"{clips_info}, {len(self.clips)} total clips")

    def _detect_format(self):
        """Detect HDF5 format (old per-sample or new global arrays)."""
        with h5py.File(self.data_path, 'r') as f:
            keys = list(f.keys())

            # New format: has 'vector' as top-level dataset
            if 'vector' in keys:
                self.format = 'new'
                self.n_samples = f['vector'].shape[0]
                self.n_timesteps = f['vector'].shape[1]
                self.spatial_shape = f['vector'].shape[2:4]  # (H, W)

                # Check for scalar
                if 'scalar' in f and f['scalar'].shape[-1] > 0:
                    self.scalar_channels = f['scalar'].shape[-1]
                    self.scalar_indices = f['scalar_indices'][:] if 'scalar_indices' in f else []
                else:
                    self.scalar_channels = 0
                    self.scalar_indices = []

                logger.info(f"Detected NEW format: {self.n_samples} samples, "
                           f"{self.n_timesteps} timesteps, shape {self.spatial_shape}")

            # Old format: numbered groups like '0', '1', ...
            else:
                self.format = 'old'
                self.sample_keys = sorted([k for k in keys if k.isdigit()], key=int)
                self.n_samples = len(self.sample_keys)

                # Get metadata from first sample
                first_key = self.sample_keys[0]
                if 'vector' in f[first_key] and 'data' in f[first_key]['vector']:
                    shape = f[first_key]['vector']['data'].shape
                else:
                    shape = f[first_key]['data'].shape
                self.n_timesteps = shape[0]
                self.spatial_shape = shape[1:3]
                self.scalar_channels = 0
                self.scalar_indices = []

                logger.info(f"Detected OLD format: {self.n_samples} samples")

    def _split_dataset(self):
        """Split samples into train/val."""
        np.random.seed(self.seed)
        indices = np.random.permutation(self.n_samples)
        split_idx = int(self.n_samples * self.train_ratio)

        if self.split == 'train':
            self.sample_indices = indices[:split_idx].tolist()
        elif self.split == 'val':
            self.sample_indices = indices[split_idx:].tolist()
        else:
            raise ValueError(f"Unknown split: {self.split}")

        self.max_start = self.n_timesteps - self.temporal_length

    def _generate_clips(self):
        """Generate clips for current epoch."""
        rng = np.random.RandomState(self.seed + self.epoch)

        self.clips = []
        self.clips_by_sample: Dict[int, List[int]] = {}

        clip_idx = 0
        for i, sample_idx in enumerate(self.sample_indices):
            num_available = self.max_start + 1

            if self.split == 'val':
                # Validation: use interval sampling (0, 8, 16, ...)
                starts = list(range(0, num_available, self.val_time_interval))
                if i == 0:  # Log once
                    logger.info(f"[Val] Interval sampling: max_start={self.max_start}, "
                               f"interval={self.val_time_interval}, clips/sample={len(starts)}")
            elif self.clips_per_sample is None:
                # Train with all clips
                starts = list(range(num_available))
            else:
                # Train with random sampling
                if self.clips_per_sample <= num_available:
                    starts = rng.choice(num_available, self.clips_per_sample, replace=False)
                else:
                    starts = rng.choice(num_available, self.clips_per_sample, replace=True)

            self.clips_by_sample[i] = []
            for start_t in starts:
                self.clips.append((i, sample_idx, int(start_t)))
                self.clips_by_sample[i].append(clip_idx)
                clip_idx += 1

    def set_epoch(self, epoch: int):
        """Set epoch for reproducible shuffling."""
        self.epoch = epoch
        self._generate_clips()

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        local_idx, sample_idx, start_t = self.clips[idx]
        end_t = start_t + self.temporal_length

        with h5py.File(self.data_path, 'r') as f:
            if self.format == 'new':
                return self._load_new_format(f, sample_idx, start_t, end_t)
            else:
                return self._load_old_format(f, sample_idx, start_t, end_t)

    def _load_new_format(
        self, f: h5py.File, sample_idx: int, start_t: int, end_t: int
    ) -> Dict[str, torch.Tensor]:
        """Load from new global array format."""
        # Load vector: [T, H, W, 3]
        vector = np.array(f['vector'][sample_idx, start_t:end_t], dtype=np.float32)

        # Load scalar if exists
        if self.scalar_channels > 0:
            scalar = np.array(f['scalar'][sample_idx, start_t:end_t], dtype=np.float32)
        else:
            scalar = None

        # Build 18-channel output
        T, H, W, C_vec = vector.shape
        data = np.zeros((T, H, W, TOTAL_CHANNELS), dtype=np.float32)

        # Vector channels [0:3] - pad to 3 channels if 2D data (Vz=0)
        data[..., :C_vec] = vector  # Fill actual channels, rest stays 0

        # Scalar channels [3:18] - map using scalar_indices
        if scalar is not None and len(self.scalar_indices) > 0:
            for i, idx in enumerate(self.scalar_indices):
                if idx < NUM_SCALAR_CHANNELS:
                    data[..., NUM_VECTOR_CHANNELS + idx] = scalar[..., i]

        # Channel mask
        channel_mask = np.zeros(TOTAL_CHANNELS, dtype=np.float32)
        channel_mask[:self.vector_dim] = 1.0  # Vx, Vy (and Vz if 3D)
        for idx in self.scalar_indices:
            if idx < NUM_SCALAR_CHANNELS:
                channel_mask[NUM_VECTOR_CHANNELS + idx] = 1.0

        # Load nu if available
        nu = f['nu'][sample_idx] if 'nu' in f else 0.0

        result = {
            'data': torch.from_numpy(data),
            'channel_mask': torch.from_numpy(channel_mask),
            'nu': torch.tensor(nu, dtype=torch.float32),
            'vector_dim': torch.tensor(self.vector_dim, dtype=torch.long),
        }

        # Load boundary data if available (for PDE loss)
        if 'boundary_left' in f:
            result['boundary_left'] = torch.from_numpy(
                np.array(f['boundary_left'][sample_idx, start_t:end_t], dtype=np.float32)
            )
            result['boundary_right'] = torch.from_numpy(
                np.array(f['boundary_right'][sample_idx, start_t:end_t], dtype=np.float32)
            )
            result['boundary_bottom'] = torch.from_numpy(
                np.array(f['boundary_bottom'][sample_idx, start_t:end_t], dtype=np.float32)
            )
            result['boundary_top'] = torch.from_numpy(
                np.array(f['boundary_top'][sample_idx, start_t:end_t], dtype=np.float32)
            )

        return result

    def _load_old_format(
        self, f: h5py.File, sample_idx: int, start_t: int, end_t: int
    ) -> Dict[str, torch.Tensor]:
        """Load from old per-sample group format."""
        sample_key = self.sample_keys[sample_idx]
        grp = f[sample_key]

        # Load vector data
        if 'vector' in grp and 'data' in grp['vector']:
            vector_2ch = np.array(grp['vector']['data'][start_t:end_t], dtype=np.float32)
            has_boundary = 'boundary' in grp['vector']
        else:
            vector_2ch = np.array(grp['data'][start_t:end_t], dtype=np.float32)
            has_boundary = False

        T, H, W, C_in = vector_2ch.shape

        # Build 18-channel output
        data = np.zeros((T, H, W, TOTAL_CHANNELS), dtype=np.float32)
        data[..., :C_in] = vector_2ch  # Usually 2 channels for 2D

        # Channel mask
        channel_mask = np.zeros(TOTAL_CHANNELS, dtype=np.float32)
        channel_mask[:self.vector_dim] = 1.0

        # Load nu
        nu = float(grp['nu'][()]) if 'nu' in grp else 0.0

        result = {
            'data': torch.from_numpy(data),
            'channel_mask': torch.from_numpy(channel_mask),
            'nu': torch.tensor(nu, dtype=torch.float32),
            'vector_dim': torch.tensor(self.vector_dim, dtype=torch.long),
        }

        # Load boundaries if available (for PDE loss)
        if has_boundary:
            bnd = grp['vector']['boundary']
            result['boundary_left'] = torch.from_numpy(
                np.array(bnd['left'][start_t:end_t], dtype=np.float32)
            )
            result['boundary_right'] = torch.from_numpy(
                np.array(bnd['right'][start_t:end_t], dtype=np.float32)
            )
            result['boundary_bottom'] = torch.from_numpy(
                np.array(bnd['bottom'][start_t:end_t], dtype=np.float32)
            )
            result['boundary_top'] = torch.from_numpy(
                np.array(bnd['top'][start_t:end_t], dtype=np.float32)
            )

        return result


class FinetuneSampler(Sampler):
    """
    Distributed-aware sampler ensuring same-sample batching.

    Critical for PDE loss where boundary conditions must be consistent
    within each batch.
    """

    def __init__(
        self,
        dataset: FinetuneDataset,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Distributed settings
        if num_replicas is None:
            num_replicas = int(os.environ.get('WORLD_SIZE', 1))
        if rank is None:
            rank = int(os.environ.get('RANK', 0))

        self.num_replicas = num_replicas
        self.rank = rank

        self._compute_batches()

    def _compute_batches(self):
        """Compute batches with same-sample constraint."""
        rng = np.random.RandomState(self.seed + self.epoch)

        # Shuffle sample order
        sample_order = list(range(len(self.dataset.sample_indices)))
        if self.shuffle:
            rng.shuffle(sample_order)

        batches = []
        for local_idx in sample_order:
            clip_indices = list(self.dataset.clips_by_sample[local_idx])
            if len(clip_indices) == 0:
                continue

            if self.shuffle:
                rng.shuffle(clip_indices)

            # Create batches, padding incomplete ones
            for i in range(0, len(clip_indices), self.batch_size):
                batch = clip_indices[i:i + self.batch_size]

                if len(batch) < self.batch_size:
                    need = self.batch_size - len(batch)
                    pool = clip_indices[:i] if i > 0 else batch
                    if len(pool) >= need:
                        pad = rng.choice(pool, need, replace=False).tolist()
                    else:
                        pad = rng.choice(pool, need, replace=True).tolist()
                    batch = batch + pad

                batches.append(batch)

        self._all_batches = batches

        # Distribute across ranks
        total = len(batches)
        self.num_batches_per_rank = total // self.num_replicas
        usable = self.num_batches_per_rank * self.num_replicas
        self._all_batches = self._all_batches[:usable]

        if self.rank == 0:
            logger.info(f"FinetuneSampler ({self.dataset.split}): {len(self._all_batches)} total batches, "
                       f"{self.num_batches_per_rank} per rank, batch_size={self.batch_size}")

    def set_epoch(self, epoch: int):
        """Set epoch and regenerate clips/batches."""
        self.epoch = epoch
        self.dataset.set_epoch(epoch)
        self._compute_batches()

    def __iter__(self):
        start = self.rank * self.num_batches_per_rank
        end = start + self.num_batches_per_rank
        for batch in self._all_batches[start:end]:
            yield batch

    def __len__(self) -> int:
        return self.num_batches_per_rank


def finetune_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for finetune dataset."""
    result = {
        'data': torch.stack([item['data'] for item in batch], dim=0),
        'channel_mask': torch.stack([item['channel_mask'] for item in batch], dim=0),
        'nu': torch.stack([item['nu'] for item in batch], dim=0),
        'vector_dim': batch[0]['vector_dim'],
    }

    # Optional boundary data
    if 'boundary_left' in batch[0]:
        result['boundary_left'] = torch.stack([item['boundary_left'] for item in batch], dim=0)
        result['boundary_right'] = torch.stack([item['boundary_right'] for item in batch], dim=0)
        result['boundary_bottom'] = torch.stack([item['boundary_bottom'] for item in batch], dim=0)
        result['boundary_top'] = torch.stack([item['boundary_top'] for item in batch], dim=0)

    return result


def create_finetune_dataloaders(
    data_path: str,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
    temporal_length: int = 9,
    train_ratio: float = 0.9,
    clips_per_sample: Optional[int] = 100,
    vector_dim: int = 2,
    val_time_interval: int = 8,
):
    """Create train and validation dataloaders."""
    train_dataset = FinetuneDataset(
        data_path=data_path,
        temporal_length=temporal_length,
        split='train',
        train_ratio=train_ratio,
        seed=seed,
        clips_per_sample=clips_per_sample,
        vector_dim=vector_dim,
        val_time_interval=val_time_interval,
    )

    val_dataset = FinetuneDataset(
        data_path=data_path,
        temporal_length=temporal_length,
        split='val',
        train_ratio=train_ratio,
        seed=seed,
        clips_per_sample=None,  # Not used for val (uses interval sampling)
        vector_dim=vector_dim,
        val_time_interval=val_time_interval,
    )

    train_sampler = FinetuneSampler(train_dataset, batch_size, shuffle=True, seed=seed)
    val_sampler = FinetuneSampler(val_dataset, batch_size, shuffle=False, seed=seed)

    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=finetune_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=finetune_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, train_sampler, val_sampler


if __name__ == "__main__":
    """Test dataset."""
    import sys

    # Test with new format file
    new_format_path = "/scratch-share/SONG0304/finetune/burgers2d_nu0.1_0.15_res128_t1000_n100.h5"

    if not Path(new_format_path).exists():
        print(f"Test file not found: {new_format_path}")
        print("Please run generate_burgers2d_dataset.py first")
        sys.exit(0)

    print("=" * 60)
    print("Testing FinetuneDataset with NEW format")
    print("=" * 60)

    dataset = FinetuneDataset(
        new_format_path,
        temporal_length=9,
        split='train',
        clips_per_sample=50,
        vector_dim=2,
    )

    print(f"\nDataset: {len(dataset)} clips")

    # Test single sample
    sample = dataset[0]
    print(f"\nSample shapes:")
    print(f"  data: {sample['data'].shape}")
    print(f"  channel_mask: {sample['channel_mask']}")
    print(f"  nu: {sample['nu']}")
    print(f"  vector_dim: {sample['vector_dim']}")

    # Test sampler
    print("\n" + "=" * 60)
    print("Testing FinetuneSampler")
    print("=" * 60)

    sampler = FinetuneSampler(dataset, batch_size=4, shuffle=True, seed=42)
    print(f"Batches per rank: {len(sampler)}")

    # Verify same-sample batching
    all_same = True
    for batch_idx, batch in enumerate(sampler):
        samples_in_batch = set()
        for clip_idx in batch:
            local_idx, _, _ = dataset.clips[clip_idx]
            samples_in_batch.add(local_idx)
        if len(samples_in_batch) != 1:
            print(f"ERROR: Batch {batch_idx} has clips from multiple samples")
            all_same = False
        if batch_idx < 3:
            print(f"  Batch {batch_idx}: sample={list(samples_in_batch)[0]}, size={len(batch)}")

    if all_same:
        print(f"\n✓ All batches contain clips from same sample!")

    print("\nDataset test passed!")
