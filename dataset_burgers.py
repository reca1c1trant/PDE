"""
Burgers2D Dataset for LoRA Finetuning.

Sampling Strategy (方案C):
- Each sample contributes K clips per epoch
- Ensures balanced usage across all samples (different nu values)
- Random start times provide temporal diversity

Example with K=100:
- 90 train samples × 100 clips = 9000 clips/epoch
- batch_size=4 → 2250 steps/epoch
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
    Dataset for 2D Burgers equation with balanced sampling.

    Each epoch, every sample contributes exactly `clips_per_sample` clips,
    ensuring balanced training across different viscosity values.

    HDF5 Structure:
        sample_N/
        ├── nu: float32 (viscosity)
        ├── scalar: Empty
        └── vector/
            ├── data: [T=1000, H=128, W=128, C=2]
            └── boundary/
                ├── left/right/bottom/top

    Output per clip:
        - data: [T=17, H=128, W=128, C=6] padded to 6 channels
        - channel_mask: [6] = [1, 1, 0, 0, 0, 0]
        - boundaries: [T=17, ...]
        - nu: float viscosity
    """

    def __init__(
        self,
        data_path: str,
        temporal_length: int = 16,
        split: str = 'train',
        train_ratio: float = 0.9,
        seed: int = 42,
        clips_per_sample: Optional[int] = 100,  # K clips per sample per epoch, None = all
    ):
        """
        Args:
            clips_per_sample: Number of clips per sample per epoch.
                              If None, use all available clips (for validation).
        """
        super().__init__()
        self.data_path = Path(data_path)
        self.temporal_length = temporal_length + 1  # 17 frames
        self.split = split
        self.train_ratio = train_ratio
        self.seed = seed
        self.clips_per_sample = clips_per_sample  # None means use all

        # Build sample index: (sample_key, nu, total_timesteps)
        self.samples: List[Tuple[str, float, int]] = []
        self._build_index()
        self._split_dataset()

        # Max possible start index
        self.max_start = self.samples[0][2] - self.temporal_length if self.samples else 0

        # Generate initial clips
        self.epoch = 0
        self._generate_clips()

        clips_info = "all" if clips_per_sample is None else str(clips_per_sample)
        logger.info(f"BurgersDataset ({split}): {len(self.samples)} samples, "
                   f"{clips_info} clips/sample, {len(self.clips)} total clips")

    def _build_index(self):
        """Build index of all samples."""
        with h5py.File(self.data_path, 'r') as f:
            for sample_key in sorted(f.keys(), key=lambda x: int(x)):
                nu = float(f[sample_key]['nu'][()])
                total_timesteps = f[sample_key]['vector']['data'].shape[0]

                if total_timesteps >= self.temporal_length:
                    self.samples.append((sample_key, nu, total_timesteps))

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

    def _generate_clips(self):
        """
        Generate clips for current epoch.

        If clips_per_sample is set: each sample contributes exactly K clips (random sampling).
        If clips_per_sample is None: each sample contributes ALL available clips.

        IMPORTANT: Clips are stored grouped by sample (not shuffled globally).
        The BurgersSampler handles shuffling to ensure each batch contains
        clips from the same sample only.
        """
        rng = np.random.RandomState(self.seed + self.epoch)

        self.clips = []
        self.clips_by_sample: Dict[int, List[int]] = {}  # sample_idx -> list of clip indices

        clip_idx = 0
        for sample_idx, (sample_key, nu, total_t) in enumerate(self.samples):
            max_start = total_t - self.temporal_length
            num_available = max_start + 1

            if self.clips_per_sample is None:
                # Use ALL available clips (for validation)
                starts = list(range(num_available))
            else:
                # Sample K start times (with replacement if K > num_available)
                if self.clips_per_sample <= num_available:
                    starts = rng.choice(num_available, self.clips_per_sample, replace=False)
                else:
                    starts = rng.choice(num_available, self.clips_per_sample, replace=True)

            self.clips_by_sample[sample_idx] = []
            for start_t in starts:
                self.clips.append((sample_idx, int(start_t)))
                self.clips_by_sample[sample_idx].append(clip_idx)
                clip_idx += 1

        # NOTE: Do NOT shuffle self.clips here. BurgersSampler handles
        # shuffling to ensure same-sample batching.

    def set_epoch(self, epoch: int):
        """Set epoch for reproducible shuffling. Call before each epoch."""
        self.epoch = epoch
        self._generate_clips()

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_idx, start_t = self.clips[idx]
        sample_key, nu, _ = self.samples[sample_idx]

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
        data_6ch[..., :2] = data_2ch

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
    Distributed-aware sampler for BurgersDataset.

    IMPORTANT: Ensures each batch contains clips from the SAME sample only.
    This is critical for PDE loss computation where boundary conditions
    must be consistent within a batch.

    Sampling logic:
    1. Shuffle sample order (which sample comes first)
    2. For each sample, shuffle its K clips
    3. Divide each sample's clips into batches of size batch_size
    4. Last incomplete batch: pad from previous clips (no discarding)
    5. Result: each batch contains clips from one sample only
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

        # No longer require clips_per_sample to be divisible by batch_size
        # Incomplete batches will be padded from previous clips

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
        """
        Compute batches ensuring each batch contains clips from same sample.

        Process:
        1. Shuffle sample order
        2. For each sample: shuffle its clips, then divide into batches
        3. Last incomplete batch: pad from previous clips of same sample
        4. Each batch = batch_size clips from ONE sample
        """
        rng = np.random.RandomState(self.seed + self.epoch)

        # Get sample indices and shuffle their order
        sample_indices = list(range(len(self.dataset.samples)))
        if self.shuffle:
            rng.shuffle(sample_indices)

        batches = []
        for sample_idx in sample_indices:
            # Get this sample's clip indices
            clip_indices = list(self.dataset.clips_by_sample[sample_idx])
            if len(clip_indices) == 0:
                continue

            # Shuffle clips within this sample
            if self.shuffle:
                rng.shuffle(clip_indices)

            # Divide into batches with padding for incomplete ones
            num_clips = len(clip_indices)
            for i in range(0, num_clips, self.batch_size):
                batch = clip_indices[i:i + self.batch_size]

                # Pad last incomplete batch from previous clips
                if len(batch) < self.batch_size:
                    need = self.batch_size - len(batch)
                    # Sample from previous clips (indices 0 to i)
                    pool = clip_indices[:i] if i > 0 else batch
                    if len(pool) >= need:
                        pad_clips = rng.choice(pool, need, replace=False).tolist()
                    else:
                        # Not enough, sample with replacement
                        pad_clips = rng.choice(pool, need, replace=True).tolist()
                    batch = batch + pad_clips

                batches.append(batch)

        self._all_batches = batches

        # Distribute across ranks (drop remainder for equal distribution)
        total_batches = len(batches)
        self.num_batches_per_rank = total_batches // self.num_replicas
        usable = self.num_batches_per_rank * self.num_replicas
        self._all_batches = self._all_batches[:usable]

        if self.rank == 0:
            logger.info(f"BurgersSampler: {len(self._all_batches)} batches total, "
                       f"{self.num_batches_per_rank} per rank, "
                       f"each batch from same sample (with padding)")

    def set_epoch(self, epoch: int):
        """Set epoch for shuffling. Also updates dataset's epoch."""
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


def burgers_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for Burgers dataset."""
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
    """Test dataset and sampling."""
    import sys

    data_path = "./burgers2d_nu0.1_0.15_res128_t1000_n100.h5"

    if not Path(data_path).exists():
        print(f"Test file not found: {data_path}")
        sys.exit(0)

    # Test dataset with clips_per_sample NOT divisible by batch_size (to test padding)
    batch_size = 8
    clips_per_sample = 100  # 100 % 8 = 4, so last batch needs padding

    dataset = BurgersDataset(
        data_path,
        temporal_length=16,
        split='train',
        clips_per_sample=clips_per_sample
    )

    print(f"\n{'='*60}")
    print(f"Dataset Statistics")
    print(f"{'='*60}")
    print(f"Number of samples: {len(dataset.samples)}")
    print(f"Clips per sample: {dataset.clips_per_sample}")
    print(f"Total clips: {len(dataset)}")
    print(f"Expected: {len(dataset.samples)} × {dataset.clips_per_sample} = {len(dataset.samples) * dataset.clips_per_sample}")
    print(f"clips_per_sample % batch_size = {clips_per_sample % batch_size} (will need padding)")

    # Test one clip
    sample = dataset[0]
    print(f"\nSample shapes:")
    print(f"  data: {sample['data'].shape}")
    print(f"  boundary_left: {sample['boundary_left'].shape}")
    print(f"  nu: {sample['nu']}")

    # Verify sampling balance
    print(f"\n{'='*60}")
    print(f"Sampling Balance Check")
    print(f"{'='*60}")
    sample_counts = {}
    for sample_idx, start_t in dataset.clips:
        sample_counts[sample_idx] = sample_counts.get(sample_idx, 0) + 1

    counts = list(sample_counts.values())
    print(f"Clips per sample - min: {min(counts)}, max: {max(counts)}, expected: {dataset.clips_per_sample}")

    # Test sampler and same-sample batching (with padding)
    print(f"\n{'='*60}")
    print(f"Same-Sample Batching Test (with Padding)")
    print(f"{'='*60}")
    sampler = BurgersSampler(dataset, batch_size=batch_size, shuffle=True, seed=42)

    # Verify each batch contains clips from same sample and has correct size
    all_same_sample = True
    all_correct_size = True
    for batch_idx, batch in enumerate(sampler):
        sample_indices_in_batch = set()
        for clip_idx in batch:
            sample_idx, _ = dataset.clips[clip_idx]
            sample_indices_in_batch.add(sample_idx)

        if len(sample_indices_in_batch) != 1:
            print(f"ERROR: Batch {batch_idx} has clips from multiple samples: {sample_indices_in_batch}")
            all_same_sample = False

        if len(batch) != batch_size:
            print(f"ERROR: Batch {batch_idx} has size {len(batch)} != {batch_size}")
            all_correct_size = False

        if batch_idx < 3:  # Show first 3 batches
            sample_idx = list(sample_indices_in_batch)[0]
            start_times = [dataset.clips[clip_idx][1] for clip_idx in batch]
            print(f"Batch {batch_idx}: sample={sample_idx}, size={len(batch)}, start_times={start_times}")

    if all_same_sample:
        print(f"\n✓ All {len(sampler)} batches contain clips from same sample only!")
    else:
        print(f"\n✗ Some batches have mixed samples!")

    if all_correct_size:
        print(f"✓ All batches have correct size {batch_size} (padding works!)")
    else:
        print(f"✗ Some batches have incorrect size!")

    # Test epoch change
    print(f"\n{'='*60}")
    print(f"Epoch Change Test")
    print(f"{'='*60}")
    sampler.set_epoch(1)
    first_batch_epoch1 = list(sampler)[0]
    sampler.set_epoch(0)
    first_batch_epoch0 = list(sampler)[0]
    print(f"Epoch 0 first batch: {first_batch_epoch0}")
    print(f"Epoch 1 first batch: {first_batch_epoch1}")
    print(f"Different after epoch change: {first_batch_epoch0 != first_batch_epoch1}")

    print("\nDataset test passed!")
