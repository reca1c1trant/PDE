"""
Burgers2D Dataset for LoRA Finetuning.

Loads 2D Burgers equation data and prepares it for PDE loss training.
- Pads 2 channels (u, v) to 6 channels to match pretrained model
- Returns boundary conditions for PDE loss computation
- Returns viscosity coefficient nu
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BurgersDataset(Dataset):
    """
    Dataset for 2D Burgers equation with boundary conditions.

    HDF5 Structure (from generate_burgers2d_dataset.py):
        sample_N/
        ├── nu: float32 (viscosity)
        ├── scalar: Empty
        └── vector/
            ├── data: [T=1000, H=128, W=128, C=2]
            └── boundary/
                ├── left:   [T=1000, H=128, 1, C=2]
                ├── right:  [T=1000, H=128, 1, C=2]
                ├── bottom: [T=1000, 1, W=128, C=2]
                └── top:    [T=1000, 1, W=128, C=2]

    Output:
        - data: [T=17, H=128, W=128, C=6] padded to 6 channels
        - channel_mask: [6] = [1, 1, 0, 0, 0, 0]
        - boundaries: dict with left/right/bottom/top [T=17, ...]
        - nu: float viscosity coefficient
    """

    def __init__(
        self,
        data_path: str,
        temporal_length: int = 16,
        split: str = 'train',
        train_ratio: float = 0.9,
        seed: int = 42
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.temporal_length = temporal_length + 1  # 17 frames for PDE loss
        self.split = split
        self.train_ratio = train_ratio
        self.seed = seed

        # Build sample index
        self.samples: List[Tuple[str, float, int]] = []  # (sample_key, nu, total_timesteps)
        self._build_index()
        self._split_dataset()

        logger.info(f"BurgersDataset: {len(self.samples)} samples for {split}")

    def _build_index(self):
        """Build index of all samples."""
        with h5py.File(self.data_path, 'r') as f:
            for sample_key in f.keys():
                nu = float(f[sample_key]['nu'][()])
                total_timesteps = f[sample_key]['vector']['data'].shape[0]

                if total_timesteps >= self.temporal_length:
                    self.samples.append((sample_key, nu, total_timesteps))
                else:
                    logger.warning(f"Skipping {sample_key}: only {total_timesteps} timesteps")

        logger.info(f"Found {len(self.samples)} valid samples")

    def _split_dataset(self):
        """Split into train/val sets."""
        np.random.seed(self.seed)
        indices = np.random.permutation(len(self.samples))
        split_idx = int(len(self.samples) * self.train_ratio)

        if self.split == 'train':
            selected = indices[:split_idx]
        elif self.split == 'val':
            selected = indices[split_idx:]
        else:
            raise ValueError(f"Unknown split: {self.split}")

        self.samples = [self.samples[i] for i in selected]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_key, nu, total_timesteps = self.samples[idx]

        # Random temporal slice
        max_start = total_timesteps - self.temporal_length
        start_t = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        end_t = start_t + self.temporal_length

        with h5py.File(self.data_path, 'r') as f:
            grp = f[sample_key]

            # Load interior data [T, H, W, 2]
            data_2ch = np.array(grp['vector']['data'][start_t:end_t], dtype=np.float32)

            # Load boundaries
            bnd = grp['vector']['boundary']
            boundary_left = np.array(bnd['left'][start_t:end_t], dtype=np.float32)
            boundary_right = np.array(bnd['right'][start_t:end_t], dtype=np.float32)
            boundary_bottom = np.array(bnd['bottom'][start_t:end_t], dtype=np.float32)
            boundary_top = np.array(bnd['top'][start_t:end_t], dtype=np.float32)

        # Pad to 6 channels: [u, v, 0, 0, 0, 0]
        T, H, W, _ = data_2ch.shape
        data_6ch = np.zeros((T, H, W, 6), dtype=np.float32)
        data_6ch[..., :2] = data_2ch  # u → ch0, v → ch1

        # Channel mask
        channel_mask = np.array([1, 1, 0, 0, 0, 0], dtype=np.float32)

        return {
            'data': torch.from_numpy(data_6ch),
            'channel_mask': torch.from_numpy(channel_mask),
            'boundary_left': torch.from_numpy(boundary_left),
            'boundary_right': torch.from_numpy(boundary_right),
            'boundary_bottom': torch.from_numpy(boundary_bottom),
            'boundary_top': torch.from_numpy(boundary_top),
            'nu': torch.tensor(nu, dtype=torch.float32),
        }


class BurgersSampler(Sampler):
    """
    Simple batch sampler for Burgers dataset.
    All samples have the same dimension (2D 128x128), so no grouping needed.
    """

    def __init__(
        self,
        dataset: BurgersDataset,
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
        import os
        if num_replicas is None:
            num_replicas = int(os.environ.get('WORLD_SIZE', 1))
        if rank is None:
            rank = int(os.environ.get('RANK', 0))

        self.num_replicas = num_replicas
        self.rank = rank

        self._compute_batches()

    def _compute_batches(self):
        """Compute batches for this epoch."""
        rng = np.random.RandomState(self.seed + self.epoch)

        indices = list(range(len(self.dataset)))
        if self.shuffle:
            rng.shuffle(indices)

        # Create batches
        batches = []
        for i in range(0, len(indices), self.batch_size):
            batch = indices[i:i + self.batch_size]
            if len(batch) == self.batch_size:
                batches.append(batch)

        if self.shuffle:
            rng.shuffle(batches)

        self._all_batches = batches
        self.num_batches_per_rank = len(batches) // self.num_replicas

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self._compute_batches()

    def __iter__(self):
        start_idx = self.rank * self.num_batches_per_rank
        end_idx = start_idx + self.num_batches_per_rank

        for batch in self._all_batches[start_idx:end_idx]:
            yield batch

    def __len__(self) -> int:
        return self.num_batches_per_rank


def burgers_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for Burgers dataset.
    """
    return {
        'data': torch.stack([item['data'] for item in batch], dim=0),
        'channel_mask': torch.stack([item['channel_mask'] for item in batch], dim=0),
        'boundary_left': torch.stack([item['boundary_left'] for item in batch], dim=0),
        'boundary_right': torch.stack([item['boundary_right'] for item in batch], dim=0),
        'boundary_bottom': torch.stack([item['boundary_bottom'] for item in batch], dim=0),
        'boundary_top': torch.stack([item['boundary_top'] for item in batch], dim=0),
        'nu': torch.stack([item['nu'] for item in batch], dim=0),
    }


if __name__ == "__main__":
    """Test dataset loading."""
    import sys

    # Test with a dummy path - update with actual path
    data_path = "./burgers2d_nu0.1_0.15_res128_t1000_n100.h5"

    if not Path(data_path).exists():
        print(f"Test file not found: {data_path}")
        print("Please generate the dataset first using generate_burgers2d_dataset.py")
        sys.exit(0)

    dataset = BurgersDataset(data_path, temporal_length=16, split='train')

    print(f"\nDataset size: {len(dataset)}")

    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Data shape: {sample['data'].shape}")  # [17, 128, 128, 6]
    print(f"Channel mask: {sample['channel_mask']}")
    print(f"Boundary left shape: {sample['boundary_left'].shape}")  # [17, 128, 1, 2]
    print(f"Boundary bottom shape: {sample['boundary_bottom'].shape}")  # [17, 1, 128, 2]
    print(f"Nu: {sample['nu']}")

    # Verify u + v = 1.5 constraint
    u = sample['data'][..., 0]
    v = sample['data'][..., 1]
    constraint_error = torch.abs(u + v - 1.5).max()
    print(f"\nConstraint error (u+v=1.5): {constraint_error:.2e}")

    print("\nDataset test passed!")
