"""
PDE Dataset Implementation with Dimension Grouping and Scalar/Vector Separation
Supports 1D, 2D, and 3D PDE data with efficient lazy loading.

Sampling Strategy:
- Each sample contributes K clips per epoch (clips_per_sample)
- Each batch contains clips from the SAME sample only
- Ensures balanced usage across all samples
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


class PDEDataset(Dataset):
    """
    Dataset for PDE data with mixed dimensions (1D/2D/3D) and scalar/vector separation.

    Data format:
        - scalar: [T, *spatial, C_s] where C_s ∈ [0, 3] (pressure, density, temperature, etc.)
        - vector: [T, *spatial, C_v] where C_v ∈ [1, 3] (velocity components)

    Spatial dimensions:
        - 1D: H=1024
        - 2D: H=W=128
        - 3D: H=W=D=128

    Output:
        - data: [T, *spatial, 6] where channels are [vx, vy, vz, p, ρ, T]
                First 3 channels: vector (padded to 3)
                Last 3 channels: scalar (padded to 3)
        - channel_mask: [6] indicating which channels are real (1) vs padded (0)

    Sampling Strategy:
        - Each sample contributes exactly `clips_per_sample` clips per epoch
        - Clips are stored grouped by sample for same-sample batching
        - Each epoch regenerates clips with new random start times

    Args:
        data_dir (str): Directory containing .h5 files
        temporal_length (int): Number of temporal steps to sample (default: 16)
        split (str): 'train' or 'val'
        train_ratio (float): Ratio of training data (default: 0.9)
        seed (int): Random seed for train/val split
        clips_per_sample (int): Number of clips per sample per epoch (default: 88)
    """

    SPATIAL_DIMS = {
        '1D': (1024,),
        '2D': (128, 128),
        '3D': (128, 128, 128)
    }

    def __init__(
        self,
        data_dir: str,
        temporal_length: int = 16,
        split: str = 'train',
        train_ratio: float = 0.9,
        seed: int = 42,
        clips_per_sample: Optional[int] = 88,
    ):
        """
        Args:
            clips_per_sample: Number of clips per sample per epoch.
                              If None, use all available clips (for validation).
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.temporal_length = temporal_length + 1  # 17 frames for causal AR
        self.split = split
        self.train_ratio = train_ratio
        self.seed = seed
        self.clips_per_sample = clips_per_sample  # None means use all

        # Build index: stores (file_path, sample_key, dim_type, total_timesteps)
        self.samples: List[Tuple[Path, str, str, int]] = []
        self._build_index()

        # Split into train/val
        self._split_dataset()

        # Generate initial clips
        self.epoch = 0
        self._generate_clips()

        clips_info = "all" if clips_per_sample is None else str(clips_per_sample)
        logger.info(f"PDEDataset ({split}): {len(self.samples)} samples, "
                   f"{clips_info} clips/sample, {len(self.clips)} total clips")
        self._log_statistics()

    def _build_index(self):
        """Build index of all samples across all h5 files."""
        # Support both single file and directory
        if self.data_dir.is_file():
            h5_files = [self.data_dir]
        else:
            h5_files = sorted(self.data_dir.glob("*.hdf5")) + sorted(self.data_dir.glob("*.h5"))

        if not h5_files:
            raise ValueError(f"No .hdf5/.h5 files found in {self.data_dir}")

        logger.info(f"Found {len(h5_files)} hdf5 files")

        for file_path in h5_files:
            with h5py.File(file_path, 'r') as f:
                for sample_key in f.keys():
                    # Use vector to infer dimension (vector is guaranteed non-empty)
                    vector_shape = f[sample_key]['vector'].shape
                    scalar_shape = f[sample_key]['scalar'].shape

                    # Verify spatial dimensions match (if scalar not empty)
                    if scalar_shape[-1] > 0:
                        assert scalar_shape[:-1] == vector_shape[:-1], \
                            f"Spatial dims mismatch in {file_path}:{sample_key} - " \
                            f"scalar {scalar_shape[:-1]} vs vector {vector_shape[:-1]}"

                    # Infer dimension type from vector
                    dim_type = self._infer_dim_type(vector_shape)
                    total_timesteps = vector_shape[0]

                    # Only add if enough temporal steps
                    if total_timesteps >= self.temporal_length:
                        self.samples.append((
                            file_path,
                            sample_key,
                            dim_type,
                            total_timesteps
                        ))
                    else:
                        logger.warning(
                            f"Skipping {file_path}:{sample_key} - "
                            f"only {total_timesteps} timesteps (need {self.temporal_length})"
                        )

    def _infer_dim_type(self, shape: Tuple[int, ...]) -> str:
        """
        Infer dimension type from data shape.
        Shape format: [T, *spatial, C]
        """
        ndim = len(shape) - 2  # Exclude T and C

        if ndim == 1:
            assert shape[1] == self.SPATIAL_DIMS['1D'][0], \
                f"1D data should have H={self.SPATIAL_DIMS['1D'][0]}, got {shape[1]}"
            return '1D'
        elif ndim == 2:
            assert shape[1:3] == self.SPATIAL_DIMS['2D'], \
                f"2D data should have H=W={self.SPATIAL_DIMS['2D'][0]}, got {shape[1:3]}"
            return '2D'
        elif ndim == 3:
            assert shape[1:4] == self.SPATIAL_DIMS['3D'], \
                f"3D data should have H=W=D={self.SPATIAL_DIMS['3D'][0]}, got {shape[1:4]}"
            return '3D'
        else:
            raise ValueError(f"Unsupported data shape: {shape}")

    def _split_dataset(self):
        """Split samples into train and validation sets."""
        np.random.seed(self.seed)
        indices = np.random.permutation(len(self.samples))
        split_idx = int(len(self.samples) * self.train_ratio)

        if self.split == 'train':
            selected_indices = indices[:split_idx]
        elif self.split == 'val':
            selected_indices = indices[split_idx:]
        else:
            raise ValueError(f"Unknown split: {self.split}")

        self.samples = [self.samples[i] for i in selected_indices]

    def _generate_clips(self):
        """
        Generate clips for current epoch.

        If clips_per_sample is set: each sample contributes exactly K clips (random sampling).
        If clips_per_sample is None: each sample contributes ALL available clips.

        Clips are stored grouped by sample for same-sample batching.
        """
        rng = np.random.RandomState(self.seed + self.epoch)

        self.clips = []  # List of (sample_idx, start_t)
        self.clips_by_sample: Dict[int, List[int]] = {}  # sample_idx -> list of clip indices

        clip_idx = 0
        for sample_idx, (_, _, _, total_t) in enumerate(self.samples):
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

        # NOTE: Do NOT shuffle here. DimensionGroupedSampler handles shuffling.

    def set_epoch(self, epoch: int):
        """Set epoch for reproducible shuffling. Call before each epoch."""
        self.epoch = epoch
        self._generate_clips()

    def _log_statistics(self):
        """Log dataset statistics."""
        dim_counts = {'1D': 0, '2D': 0, '3D': 0}
        for _, _, dim_type, _ in self.samples:
            dim_counts[dim_type] += 1

        logger.info(f"Dataset statistics ({self.split}):")
        for dim_type, count in dim_counts.items():
            if count > 0:
                logger.info(f"  {dim_type}: {count} samples")

    def _pad_to_3_channels(self, data: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Pad data to 3 channels along last dimension.
        """
        C = data.shape[-1]

        if C == 0:
            padded = np.zeros((*data.shape[:-1], 3), dtype=np.float32)
            n_real = 0
        elif C < 3:
            pad_width = [(0, 0)] * (data.ndim - 1) + [(0, 3 - C)]
            padded = np.pad(data, pad_width, mode='constant', constant_values=0)
            n_real = C
        else:
            padded = data.astype(np.float32)
            n_real = 3

        return padded, n_real

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary containing:
            - data: [T, *spatial, 6] with channels [vx, vy, vz, p, ρ, T]
            - channel_mask: [6] indicating real (1) vs padded (0) channels

        Uses pre-generated clip (sample_idx, start_t) from _generate_clips().
        """
        sample_idx, start_t = self.clips[idx]
        file_path, sample_key, dim_type, total_timesteps = self.samples[sample_idx]

        end_t = start_t + self.temporal_length

        # Load data lazily
        with h5py.File(file_path, 'r') as f:
            scalar = np.array(f[sample_key]['scalar'][start_t:end_t], dtype=np.float32)
            vector = np.array(f[sample_key]['vector'][start_t:end_t], dtype=np.float32)

        # Pad to 3 channels each
        scalar_padded, n_scalar = self._pad_to_3_channels(scalar)
        vector_padded, n_vector = self._pad_to_3_channels(vector)

        # Concatenate: [vector (3), scalar (3)]
        data = np.concatenate([vector_padded, scalar_padded], axis=-1)

        # Convert to torch tensor
        data_tensor = torch.from_numpy(data)

        # Create channel mask
        channel_mask = torch.zeros(6, dtype=torch.float32)
        channel_mask[:n_vector] = 1.0
        channel_mask[3:3+n_scalar] = 1.0

        return {
            'data': data_tensor,
            'channel_mask': channel_mask
        }

    def get_dim_type(self, idx: int) -> str:
        """Get dimension type for a clip."""
        sample_idx, _ = self.clips[idx]
        return self.samples[sample_idx][2]

    def get_sample_idx(self, clip_idx: int) -> int:
        """Get sample index for a clip."""
        return self.clips[clip_idx][0]


class DimensionGroupedSampler(Sampler):
    """
    Distributed-compatible sampler that groups samples by dimension type.

    IMPORTANT: Ensures each batch contains clips from the SAME sample only.
    This is critical for consistent training where all items in a batch
    come from the same physical simulation.

    Sampling logic:
    1. Group samples by dimension (1D/2D/3D)
    2. For each dimension group, shuffle sample order
    3. For each sample, shuffle its K clips, then divide into batches
    4. Result: each batch contains clips from one sample only

    Args:
        dataset (PDEDataset): The dataset to sample from
        batch_size (int): Batch size per GPU
        shuffle (bool): Whether to shuffle
        num_replicas (int): Number of distributed processes
        rank (int): Current process rank
        seed (int): Random seed for shuffling
    """

    def __init__(
        self,
        dataset: PDEDataset,
        batch_size: int,
        shuffle: bool = True,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Validate: clips_per_sample must be divisible by batch_size (only when fixed K)
        # When clips_per_sample is None (use all), skip this check - incomplete batches are dropped
        if dataset.clips_per_sample is not None and dataset.clips_per_sample % batch_size != 0:
            raise ValueError(
                f"clips_per_sample ({dataset.clips_per_sample}) must be "
                f"divisible by batch_size ({batch_size})"
            )

        # Auto-detect distributed settings
        if num_replicas is None:
            import os
            num_replicas = int(os.environ.get('WORLD_SIZE', 1))
        if rank is None:
            import os
            rank = int(os.environ.get('RANK', 0))

        self.num_replicas = num_replicas
        self.rank = rank

        # Group sample indices by dimension type
        self.dim_groups: Dict[str, List[int]] = {'1D': [], '2D': [], '3D': []}
        for sample_idx, (_, _, dim_type, _) in enumerate(dataset.samples):
            self.dim_groups[dim_type].append(sample_idx)

        # Pre-compute batches
        self._compute_batches()

        if rank == 0:
            clips_info = "all" if dataset.clips_per_sample is None else str(dataset.clips_per_sample)
            logger.info("DimensionGroupedSampler initialized:")
            for dim_type, indices in self.dim_groups.items():
                if len(indices) > 0:
                    logger.info(f"  {dim_type}: {len(indices)} samples")
            logger.info(f"  Clips per sample: {clips_info}")
            logger.info(f"  Batch size: {batch_size}")
            logger.info(f"  Total batches: {len(self._all_batches)}, per rank: {self.num_batches_per_rank}")
            logger.info(f"  Each batch from same sample: YES")

    def _compute_batches(self):
        """
        Compute batches ensuring each batch contains clips from same sample.

        Process:
        1. For each dimension group, shuffle sample order
        2. For each sample: shuffle its clips, then divide into batches
        3. Each batch = batch_size clips from ONE sample
        """
        rng = np.random.RandomState(self.seed + self.epoch)

        batches = []
        for dim_type, sample_indices in self.dim_groups.items():
            if len(sample_indices) == 0:
                continue

            # Shuffle sample order within this dimension
            if self.shuffle:
                sample_indices = rng.permutation(sample_indices).tolist()
            else:
                sample_indices = list(sample_indices)

            for sample_idx in sample_indices:
                # Get this sample's clip indices
                clip_indices = list(self.dataset.clips_by_sample[sample_idx])

                # Shuffle clips within this sample
                if self.shuffle:
                    rng.shuffle(clip_indices)

                # Divide into batches (all clips in a batch are from same sample)
                for i in range(0, len(clip_indices), self.batch_size):
                    batch = clip_indices[i:i + self.batch_size]
                    if len(batch) == self.batch_size:
                        batches.append(batch)

        # Shuffle all batches across dimensions
        if self.shuffle:
            rng.shuffle(batches)

        self._all_batches = batches
        self.num_batches_per_rank = len(batches) // self.num_replicas

        if self.rank == 0:
            logger.info(f"DimensionGroupedSampler: {len(batches)} batches total, "
                       f"{self.num_batches_per_rank} per rank")

    def set_epoch(self, epoch: int):
        """Set epoch for shuffling. Also updates dataset's epoch."""
        self.epoch = epoch
        self.dataset.set_epoch(epoch)
        self._compute_batches()

    def __iter__(self):
        """Generate batches for this rank only."""
        start_idx = self.rank * self.num_batches_per_rank
        end_idx = start_idx + self.num_batches_per_rank

        for batch in self._all_batches[start_idx:end_idx]:
            yield batch

    def __len__(self) -> int:
        """Return number of batches for this rank."""
        return self.num_batches_per_rank


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function that stacks batch of dicts.
    """
    data = torch.stack([item['data'] for item in batch], dim=0)
    channel_mask = torch.stack([item['channel_mask'] for item in batch], dim=0)

    return {
        'data': data,
        'channel_mask': channel_mask
    }


if __name__ == "__main__":
    """Test dataset and sampling."""
    data_dir = "/home/msai/song0304/code/PDE/data"

    # Test dataset
    batch_size = 8
    clips_per_sample = 88  # Must be divisible by batch_size

    train_dataset = PDEDataset(
        data_dir=data_dir,
        temporal_length=16,
        split='train',
        train_ratio=0.9,
        seed=42,
        clips_per_sample=clips_per_sample
    )

    print(f"\n{'='*60}")
    print(f"Dataset Statistics")
    print(f"{'='*60}")
    print(f"Number of samples: {len(train_dataset.samples)}")
    print(f"Clips per sample: {train_dataset.clips_per_sample}")
    print(f"Total clips: {len(train_dataset)}")

    # Test sampler
    print(f"\n{'='*60}")
    print(f"Same-Sample Batching Test")
    print(f"{'='*60}")
    sampler = DimensionGroupedSampler(train_dataset, batch_size=batch_size, shuffle=True, seed=42)

    # Verify each batch contains clips from same sample
    all_same_sample = True
    for batch_idx, batch in enumerate(sampler):
        sample_indices_in_batch = set()
        for clip_idx in batch:
            sample_idx = train_dataset.get_sample_idx(clip_idx)
            sample_indices_in_batch.add(sample_idx)

        if len(sample_indices_in_batch) != 1:
            print(f"ERROR: Batch {batch_idx} has clips from multiple samples: {sample_indices_in_batch}")
            all_same_sample = False

        if batch_idx < 3:
            sample_idx = list(sample_indices_in_batch)[0]
            start_times = [train_dataset.clips[clip_idx][1] for clip_idx in batch]
            print(f"Batch {batch_idx}: sample={sample_idx}, start_times={start_times}")

    if all_same_sample:
        print(f"\n✓ All {len(sampler)} batches contain clips from same sample only!")
    else:
        print(f"\n✗ Some batches have mixed samples!")

    print("\nDataset test passed!")
