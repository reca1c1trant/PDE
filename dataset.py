"""
PDE Dataset Implementation with Dimension Grouping
Supports 1D, 2D, and 3D PDE data with efficient lazy loading.
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
    Dataset for PDE data with mixed dimensions (1D/2D/3D).
    
    Data format:
        - 1D: [T, H, C] where H=1024
        - 2D: [T, H, W, C] where H=W=128
        - 3D: [T, H, W, D, C] where H=W=D=128
    
    Args:
        data_dir (str): Directory containing .h5 files
        temporal_length (int): Number of temporal steps to sample (default: 16)
        split (str): 'train' or 'val'
        train_ratio (float): Ratio of training data (default: 0.9)
        seed (int): Random seed for train/val split
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
        seed: int = 42
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.temporal_length = temporal_length
        self.split = split
        self.train_ratio = train_ratio
        self.seed = seed
        
        # Build index: stores (file_path, sample_key, dim_type, total_timesteps)
        self.samples: List[Tuple[Path, str, str, int]] = []
        self._build_index()
        
        # Split into train/val
        self._split_dataset()
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} set")
        self._log_statistics()
    
    def _build_index(self):
        """Build index of all samples across all h5 files."""
        h5_files = sorted(self.data_dir.glob("*.h5"))
        
        if not h5_files:
            raise ValueError(f"No .h5 files found in {self.data_dir}")
        
        logger.info(f"Found {len(h5_files)} .h5 files")
        
        for file_path in h5_files:
            with h5py.File(file_path, 'r') as f:
                for sample_key in f.keys():
                    data_shape = f[sample_key]['data'].shape
                    
                    # Determine dimension type
                    dim_type = self._infer_dim_type(data_shape)
                    total_timesteps = data_shape[0]
                    
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
        """Infer dimension type from data shape."""
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
    
    def _log_statistics(self):
        """Log dataset statistics."""
        dim_counts = {'1D': 0, '2D': 0, '3D': 0}
        for _, _, dim_type, _ in self.samples:
            dim_counts[dim_type] += 1
        
        logger.info(f"Dataset statistics ({self.split}):")
        for dim_type, count in dim_counts.items():
            logger.info(f"  {dim_type}: {count} samples")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns a tensor of shape [temporal_length, *spatial_dims, channels].
        Randomly samples consecutive temporal steps.
        """
        file_path, sample_key, dim_type, total_timesteps = self.samples[idx]
        
        # Randomly select starting temporal index
        max_start = total_timesteps - self.temporal_length
        start_t = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        end_t = start_t + self.temporal_length
        
        # Load data lazily
        with h5py.File(file_path, 'r') as f:
            data = f[sample_key]['data'][start_t:end_t]
            data = np.array(data, dtype=np.float32)
        
        # Convert to torch tensor
        tensor = torch.from_numpy(data)
        
        return tensor
    
    def get_dim_type(self, idx: int) -> str:
        """Get dimension type for a sample."""
        return self.samples[idx][2]


class DimensionGroupedSampler(Sampler):
    """
    Sampler that groups samples by dimension type.
    Ensures each batch contains only samples of the same dimension.
    
    Args:
        dataset (PDEDataset): The dataset to sample from
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle within each dimension group
    """
    
    def __init__(
        self,
        dataset: PDEDataset,
        batch_size: int,
        shuffle: bool = True
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Group indices by dimension type
        self.dim_groups: Dict[str, List[int]] = {'1D': [], '2D': [], '3D': []}
        for idx in range(len(dataset)):
            dim_type = dataset.get_dim_type(idx)
            self.dim_groups[dim_type].append(idx)
        
        logger.info("Dimension groups for sampling:")
        for dim_type, indices in self.dim_groups.items():
            logger.info(f"  {dim_type}: {len(indices)} samples")
    
    def __iter__(self):
        """Generate batches grouped by dimension."""
        batches = []
        
        for dim_type, indices in self.dim_groups.items():
            if len(indices) == 0:
                continue
            
            # Shuffle within dimension group
            if self.shuffle:
                indices = np.random.permutation(indices).tolist()
            
            # Create batches for this dimension
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                batches.append(batch)
        
        # Shuffle batch order
        if self.shuffle:
            np.random.shuffle(batches)
        
        # Flatten and yield
        for batch in batches:
            yield batch
    
    def __len__(self) -> int:
        return len(self.dataset)


def collate_fn(batch: List[torch.Tensor]) -> torch.Tensor:
    """
    Simple collate function that stacks tensors.
    Since DimensionGroupedSampler ensures same dimensions in batch,
    we can safely stack.
    
    Args:
        batch: List of tensors with shape [T, *spatial_dims, C]
    
    Returns:
        Stacked tensor of shape [B, T, *spatial_dims, C]
    """
    return torch.stack(batch, dim=0)


if __name__ == "__main__":
    """
    Debug script to test dataset and dataloader.
    """
    # Example usage
    data_dir = "/home/msai/song0304/code/PDE/data"  # Change this
    
    # Create datasets
    train_dataset = PDEDataset(
        data_dir=data_dir,
        temporal_length=16,
        split='train',
        train_ratio=0.9,
        seed=42
    )
    
    val_dataset = PDEDataset(
        data_dir=data_dir,
        temporal_length=16,
        split='val',
        train_ratio=0.9,
        seed=42
    )
    
    # Create samplers
    train_sampler = DimensionGroupedSampler(
        dataset=train_dataset,
        batch_size=4,
        shuffle=True
    )
    
    val_sampler = DimensionGroupedSampler(
        dataset=val_dataset,
        batch_size=4,
        shuffle=False
    )
    
    # Create dataloaders
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # Test loading
    logger.info("\n" + "="*50)
    logger.info("Testing data loading...")
    logger.info("="*50)
    
    for i, batch in enumerate(train_loader):
        logger.info(f"\nBatch {i+1}:")
        logger.info(f"  Shape: {batch.shape}")
        logger.info(f"  Dtype: {batch.dtype}")
        logger.info(f"  Device: {batch.device}")
        logger.info(f"  Min: {batch.min():.4f}, Max: {batch.max():.4f}")
        
        if i >= 2:  # Only show first 3 batches
            break
    
    logger.info("\n" + "="*50)
    logger.info("Data loading test completed successfully!")
    logger.info("="*50)