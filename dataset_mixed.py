"""
Mixed PDE Dataset with Adaptive Temporal Length

Features:
- Adaptive temporal_length: T < threshold → 8, else → 16
- Mixed datasets with ratio control (e.g., 100% new + 20% old)
- No clip discarding: last incomplete batch is padded from previous clips
- Same-sample batching preserved

Usage:
    dataset = MixedPDEDataset(
        data_sources=[
            {'path': './new_data.hdf5', 'ratio': 1.0},
            {'path': './old_data/', 'ratio': 0.2},
        ],
        temporal_threshold=24,
        batch_size=8,
    )
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MixedPDEDataset(Dataset):
    """
    Mixed dataset with adaptive temporal length.

    Key features:
    - Supports multiple data sources (files or directories)
    - Adaptive temporal_length based on sample's T
    - All clips are used (no discarding)

    Args:
        data_sources: List of {'path': str, 'ratio': float}
            - path: hdf5 file or directory containing hdf5 files
            - ratio: sampling ratio (1.0 = use all samples, 0.2 = use 20%)
        temporal_threshold: T < threshold uses temporal_length=8, else 16
        split: 'train' or 'val'
        train_ratio: ratio for train/val split
        seed: random seed
    """

    SPATIAL_DIMS = {
        '1D': (1024,),
        '2D': (128, 128),
        '3D': (128, 128, 128)
    }

    def __init__(
        self,
        data_sources: List[Dict[str, Union[str, float]]],
        temporal_threshold: int = 24,
        split: str = 'train',
        train_ratio: float = 0.9,
        seed: int = 42,
    ):
        super().__init__()
        self.data_sources = data_sources
        self.temporal_threshold = temporal_threshold
        self.split = split
        self.train_ratio = train_ratio
        self.seed = seed

        # Build index from all sources
        self.samples: List[Dict] = []
        self._build_index()

        # Split into train/val
        self._split_dataset()

        # Generate clips (done per epoch)
        self.epoch = 0
        self._generate_clips()

        logger.info(f"MixedPDEDataset ({split}): {len(self.samples)} samples, {len(self.clips)} clips")
        self._log_statistics()

    def _build_index(self):
        """Build index from all data sources."""
        for source in self.data_sources:
            path = Path(source['path'])
            ratio = source.get('ratio', 1.0)

            if path.is_file() and path.suffix in ['.h5', '.hdf5']:
                h5_files = [path]
            elif path.is_dir():
                h5_files = sorted(path.glob("*.hdf5")) + sorted(path.glob("*.h5"))
            else:
                raise ValueError(f"Invalid path: {path}")

            if not h5_files:
                raise ValueError(f"No hdf5 files found at {path}")

            source_samples = []
            for file_path in h5_files:
                with h5py.File(file_path, 'r') as f:
                    for sample_key in f.keys():
                        vector_shape = f[sample_key]['vector'].shape
                        scalar_shape = f[sample_key]['scalar'].shape

                        # Verify spatial dims match
                        if scalar_shape[-1] > 0:
                            assert scalar_shape[:-1] == vector_shape[:-1]

                        dim_type = self._infer_dim_type(vector_shape)
                        total_timesteps = vector_shape[0]

                        # Determine adaptive temporal_length
                        if total_timesteps < self.temporal_threshold:
                            temporal_length = 8
                        else:
                            temporal_length = 16

                        required = temporal_length + 1  # causal AR needs +1
                        if total_timesteps >= required:
                            source_samples.append({
                                'file_path': file_path,
                                'sample_key': sample_key,
                                'dim_type': dim_type,
                                'total_timesteps': total_timesteps,
                                'temporal_length': temporal_length,
                                'source_path': str(path),
                                'ratio': ratio,
                            })

            logger.info(f"Source {path}: {len(source_samples)} samples, ratio={ratio}")
            self.samples.extend(source_samples)

    def _infer_dim_type(self, shape: Tuple[int, ...]) -> str:
        """Infer dimension type from data shape."""
        ndim = len(shape) - 2  # Exclude T and C
        if ndim == 1:
            return '1D'
        elif ndim == 2:
            return '2D'
        elif ndim == 3:
            return '3D'
        else:
            raise ValueError(f"Unsupported shape: {shape}")

    def _split_dataset(self):
        """Split samples into train/val. Ratio is applied per-epoch in _generate_clips."""
        rng = np.random.RandomState(self.seed)

        # Group by source
        by_source: Dict[str, List[int]] = {}
        for idx, sample in enumerate(self.samples):
            src = sample['source_path']
            if src not in by_source:
                by_source[src] = []
            by_source[src].append(idx)

        selected_indices = []
        for src, indices in by_source.items():
            # Shuffle within source
            indices = rng.permutation(indices).tolist()

            # Train/val split only (ratio applied per-epoch)
            split_idx = int(len(indices) * self.train_ratio)
            if self.split == 'train':
                src_indices = indices[:split_idx]
            else:
                src_indices = indices[split_idx:]

            selected_indices.extend(src_indices)

        self.samples = [self.samples[i] for i in selected_indices]

    def _generate_clips(self):
        """
        Generate all clips for current epoch.

        For training:
        - ratio=1.0 sources: use all samples
        - ratio<1.0 sources: randomly select ratio% samples (different each epoch)

        For validation: always use all samples (ratio ignored).
        """
        rng = np.random.RandomState(self.seed + self.epoch)

        self.clips: List[Dict] = []
        self.clips_by_sample: Dict[int, List[int]] = {}

        # Group samples by source for ratio-based selection
        by_source: Dict[str, List[int]] = {}
        for sample_idx, sample in enumerate(self.samples):
            src = sample['source_path']
            if src not in by_source:
                by_source[src] = []
            by_source[src].append(sample_idx)

        # Select samples based on ratio (train only)
        selected_samples: List[int] = []
        for src, sample_indices in by_source.items():
            ratio = self.samples[sample_indices[0]]['ratio'] if sample_indices else 1.0

            if self.split == 'train' and ratio < 1.0:
                # Randomly select ratio% of samples (different each epoch)
                n_select = max(1, int(len(sample_indices) * ratio))
                chosen = rng.choice(sample_indices, n_select, replace=False).tolist()
                selected_samples.extend(chosen)
            else:
                # Use all samples
                selected_samples.extend(sample_indices)

        # Generate clips for selected samples
        clip_idx = 0
        for sample_idx in selected_samples:
            sample = self.samples[sample_idx]
            temporal_length = sample['temporal_length']
            total_t = sample['total_timesteps']
            required = temporal_length + 1
            num_available = total_t - required + 1

            # Use ALL available clips (no discarding)
            starts = list(range(num_available))
            rng.shuffle(starts)  # Shuffle order within sample

            self.clips_by_sample[sample_idx] = []
            for start_t in starts:
                self.clips.append({
                    'sample_idx': sample_idx,
                    'start_t': start_t,
                    'temporal_length': temporal_length,
                })
                self.clips_by_sample[sample_idx].append(clip_idx)
                clip_idx += 1

    def set_epoch(self, epoch: int):
        """Set epoch for reproducible shuffling."""
        self.epoch = epoch
        self._generate_clips()

    def _log_statistics(self):
        """Log dataset statistics."""
        # By dimension
        dim_counts = {'1D': 0, '2D': 0, '3D': 0}
        # By temporal_length
        tl_counts = {8: 0, 16: 0}

        for sample in self.samples:
            dim_counts[sample['dim_type']] += 1
            tl_counts[sample['temporal_length']] += 1

        logger.info(f"Statistics ({self.split}):")
        for dim_type, count in dim_counts.items():
            if count > 0:
                logger.info(f"  {dim_type}: {count} samples")
        for tl, count in tl_counts.items():
            if count > 0:
                logger.info(f"  temporal_length={tl}: {count} samples")

    def _pad_to_3_channels(self, data: np.ndarray) -> Tuple[np.ndarray, int]:
        """Pad data to 3 channels."""
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
        """Get a clip."""
        clip = self.clips[idx]
        sample = self.samples[clip['sample_idx']]

        start_t = clip['start_t']
        temporal_length = clip['temporal_length']
        end_t = start_t + temporal_length + 1  # +1 for causal AR

        with h5py.File(sample['file_path'], 'r') as f:
            scalar = np.array(f[sample['sample_key']]['scalar'][start_t:end_t], dtype=np.float32)
            vector = np.array(f[sample['sample_key']]['vector'][start_t:end_t], dtype=np.float32)

        scalar_padded, n_scalar = self._pad_to_3_channels(scalar)
        vector_padded, n_vector = self._pad_to_3_channels(vector)

        data = np.concatenate([vector_padded, scalar_padded], axis=-1)
        data_tensor = torch.from_numpy(data)

        channel_mask = torch.zeros(6, dtype=torch.float32)
        channel_mask[:n_vector] = 1.0
        channel_mask[3:3+n_scalar] = 1.0

        return {
            'data': data_tensor,
            'channel_mask': channel_mask,
            'temporal_length': temporal_length,
        }

    def get_sample_idx(self, clip_idx: int) -> int:
        """Get sample index for a clip."""
        return self.clips[clip_idx]['sample_idx']

    def get_temporal_length(self, clip_idx: int) -> int:
        """Get temporal_length for a clip."""
        return self.clips[clip_idx]['temporal_length']


class AdaptiveSampler(Sampler):
    """
    Sampler that:
    1. Groups by temporal_length (8 or 16)
    2. Uses ALL clips (no discarding)
    3. Pads last incomplete batch from previous clips

    Args:
        dataset: MixedPDEDataset
        batch_size: batch size
        shuffle: whether to shuffle
        num_replicas: for distributed training
        rank: current rank
        seed: random seed
    """

    def __init__(
        self,
        dataset: MixedPDEDataset,
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

        # Auto-detect distributed settings
        if num_replicas is None:
            import os
            num_replicas = int(os.environ.get('WORLD_SIZE', 1))
        if rank is None:
            import os
            rank = int(os.environ.get('RANK', 0))

        self.num_replicas = num_replicas
        self.rank = rank

        # Group samples by (dim_type, temporal_length)
        self.sample_groups: Dict[Tuple[str, int], List[int]] = {}
        for sample_idx, sample in enumerate(dataset.samples):
            key = (sample['dim_type'], sample['temporal_length'])
            if key not in self.sample_groups:
                self.sample_groups[key] = []
            self.sample_groups[key].append(sample_idx)

        self._compute_batches()

        if rank == 0:
            logger.info("AdaptiveSampler initialized:")
            for key, indices in self.sample_groups.items():
                logger.info(f"  {key}: {len(indices)} samples")
            logger.info(f"  Total batches: {len(self._all_batches)}")
            logger.info(f"  Batches per rank: {self.num_batches_per_rank}")

    def _compute_batches(self):
        """
        Compute batches with padding for incomplete ones.

        For each sample:
        - Divide clips into batches of size batch_size
        - Last batch: if incomplete, pad from previous clips
        """
        rng = np.random.RandomState(self.seed + self.epoch)
        batches = []

        for (dim_type, temporal_length), sample_indices in self.sample_groups.items():
            if len(sample_indices) == 0:
                continue

            # Filter to only samples that have clips this epoch (ratio-based selection)
            active_samples = [idx for idx in sample_indices if idx in self.dataset.clips_by_sample]
            if len(active_samples) == 0:
                continue

            # Shuffle sample order
            if self.shuffle:
                active_samples = rng.permutation(active_samples).tolist()
            else:
                active_samples = list(active_samples)

            for sample_idx in active_samples:
                clip_indices = list(self.dataset.clips_by_sample[sample_idx])
                if len(clip_indices) == 0:
                    continue

                # Shuffle clips within sample
                if self.shuffle:
                    rng.shuffle(clip_indices)

                # Divide into batches
                num_clips = len(clip_indices)
                for i in range(0, num_clips, self.batch_size):
                    batch = clip_indices[i:i + self.batch_size]

                    # Pad last incomplete batch
                    if len(batch) < self.batch_size:
                        # Need to pad with (batch_size - len(batch)) clips
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

        # Shuffle all batches
        if self.shuffle:
            rng.shuffle(batches)

        self._all_batches = batches

        # Distribute across ranks
        total_batches = len(batches)
        self.num_batches_per_rank = total_batches // self.num_replicas

        # Handle remainder: drop extra batches to ensure equal distribution
        # (Could also pad, but dropping is simpler)
        usable = self.num_batches_per_rank * self.num_replicas
        self._all_batches = self._all_batches[:usable]

    def set_epoch(self, epoch: int):
        """Set epoch and recompute batches."""
        self.epoch = epoch
        self.dataset.set_epoch(epoch)
        self._compute_batches()

    def __iter__(self):
        """Yield batches for this rank."""
        start = self.rank * self.num_batches_per_rank
        end = start + self.num_batches_per_rank
        for batch in self._all_batches[start:end]:
            yield batch

    def __len__(self) -> int:
        return self.num_batches_per_rank


def mixed_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function that handles variable temporal_length.
    All items in a batch should have the same temporal_length (ensured by sampler).
    """
    data = torch.stack([item['data'] for item in batch], dim=0)
    channel_mask = torch.stack([item['channel_mask'] for item in batch], dim=0)

    # All should be same temporal_length (verified)
    temporal_lengths = [item['temporal_length'] for item in batch]
    assert len(set(temporal_lengths)) == 1, f"Mixed temporal_lengths in batch: {set(temporal_lengths)}"

    return {
        'data': data,
        'channel_mask': channel_mask,
        'temporal_length': temporal_lengths[0],
    }


if __name__ == "__main__":
    """Test the mixed dataset."""
    print("=" * 60)
    print("Testing MixedPDEDataset")
    print("=" * 60)

    # Example configuration
    data_sources = [
        {'path': './2D_CFD_2000_final.hdf5', 'ratio': 1.0},  # New dataset: 100%
        {'path': './data/', 'ratio': 0.2},  # Old dataset: 20%
    ]

    # Check if paths exist before testing
    from pathlib import Path
    valid_sources = [s for s in data_sources if Path(s['path']).exists()]

    if not valid_sources:
        print("No valid data sources found. Skipping test.")
    else:
        dataset = MixedPDEDataset(
            data_sources=valid_sources,
            temporal_threshold=24,
            split='train',
            train_ratio=0.9,
            seed=42,
        )

        print(f"\nDataset size: {len(dataset)} clips")

        # Test sampler
        batch_size = 8
        sampler = AdaptiveSampler(dataset, batch_size=batch_size, shuffle=True, seed=42)

        print(f"\nSampler: {len(sampler)} batches")

        # Verify batches
        for batch_idx, batch in enumerate(sampler):
            if batch_idx >= 3:
                break

            # Check all clips from same sample and same temporal_length
            sample_indices = set()
            temporal_lengths = set()
            for clip_idx in batch:
                sample_indices.add(dataset.get_sample_idx(clip_idx))
                temporal_lengths.add(dataset.get_temporal_length(clip_idx))

            print(f"Batch {batch_idx}: samples={sample_indices}, T={temporal_lengths}, size={len(batch)}")

            assert len(temporal_lengths) == 1, "Mixed temporal_lengths!"
            assert len(batch) == batch_size, f"Batch size {len(batch)} != {batch_size}"

        print("\nTest passed!")
