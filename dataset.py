"""
PDE Dataset Implementation with Dimension Grouping and Scalar/Vector Separation
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
        h5_files = sorted(self.data_dir.glob("*.hdf5"))
        

        
        if not h5_files:
            raise ValueError(f"No .hdf5 files found in {self.data_dir}")
        
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
    
    def _log_statistics(self):
        """Log dataset statistics."""
        dim_counts = {'1D': 0, '2D': 0, '3D': 0}
        for _, _, dim_type, _ in self.samples:
            dim_counts[dim_type] += 1
        
        logger.info(f"Dataset statistics ({self.split}):")
        for dim_type, count in dim_counts.items():
            logger.info(f"  {dim_type}: {count} samples")
    
    def _pad_to_3_channels(self, data: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Pad data to 3 channels along last dimension.
        
        Args:
            data: Array with shape [..., C] where C ∈ [0, 3]
        
        Returns:
            padded: Array with shape [..., 3]
            n_real: Number of real channels (before padding)
        
        Examples:
            - [..., 2] → [..., 3] by padding [0] at the end
            - [..., 0] → [..., 3] by creating full zeros
            - [..., 3] → [..., 3] unchanged
        """
        C = data.shape[-1]
        
        if C == 0:  # Empty array
            # Create full zeros with same shape but C=3
            padded = np.zeros((*data.shape[:-1], 3), dtype=np.float32)
            n_real = 0
        elif C < 3:
            # Pad at the end of last dimension
            pad_width = [(0, 0)] * (data.ndim - 1) + [(0, 3 - C)]
            padded = np.pad(data, pad_width, mode='constant', constant_values=0)
            n_real = C
        else:  # C == 3
            padded = data.astype(np.float32)
            n_real = 3
        
        return padded, n_real
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary containing:
            - data: [T, *spatial, 6] with channels [vx, vy, vz, p, ρ, T]
            - channel_mask: [6] indicating real (1) vs padded (0) channels
        
        Randomly samples consecutive temporal steps.
        """
        file_path, sample_key, dim_type, total_timesteps = self.samples[idx]
        
        # Randomly select starting temporal index
        max_start = total_timesteps - self.temporal_length
        start_t = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        end_t = start_t + self.temporal_length
        
        # Load data lazily
        with h5py.File(file_path, 'r') as f:
            scalar = np.array(f[sample_key]['scalar'][start_t:end_t], dtype=np.float32)
            vector = np.array(f[sample_key]['vector'][start_t:end_t], dtype=np.float32)
        
        # Pad to 3 channels each
        scalar_padded, n_scalar = self._pad_to_3_channels(scalar)
        vector_padded, n_vector = self._pad_to_3_channels(vector)
        
        # Concatenate: [vector (3), scalar (3)]
        # Channels: [vx, vy, vz, p, ρ, T]
        data = np.concatenate([vector_padded, scalar_padded], axis=-1)
        
        # Convert to torch tensor
        data_tensor = torch.from_numpy(data)
        
        # Create channel mask
        channel_mask = torch.zeros(6, dtype=torch.float32)
        channel_mask[:n_vector] = 1.0  # First 3 channels for vector
        channel_mask[3:3+n_scalar] = 1.0  # Last 3 channels for scalar
        
        return {
            'data': data_tensor,
            'channel_mask': channel_mask
        }
    
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
        """Generate batches grouped by dimension, dropping last incomplete batch."""
        batches = []
        
        for _, indices in self.dim_groups.items():
            if len(indices) < self.batch_size:
                continue 
            
            # Shuffle within dimension group
            if self.shuffle:
                indices = np.random.permutation(indices).tolist()
            
            num_full_batches = len(indices) // self.batch_size
            
            for i in range(num_full_batches):
                start = i * self.batch_size
                end = start + self.batch_size
                batch = indices[start:end]
                batches.append(batch)
        
        if self.shuffle:
            np.random.shuffle(batches)
        
        # Yield complete batches
        for batch in batches:
            yield batch
    
    def __len__(self) -> int:
        return len(self.dataset)


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function that stacks batch of dicts.
    
    Args:
        batch: List of dicts with keys ['data', 'channel_mask']
               - data: [T, *spatial, 6]
               - channel_mask: [6]
    
    Returns:
        Dict with:
            - data: [B, T, *spatial, 6]
            - channel_mask: [B, 6]
    """
    data = torch.stack([item['data'] for item in batch], dim=0)
    channel_mask = torch.stack([item['channel_mask'] for item in batch], dim=0)
    
    return {
        'data': data,
        'channel_mask': channel_mask
    }


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
        batch_size=16,
        shuffle=True
    )
    
    val_sampler = DimensionGroupedSampler(
        dataset=val_dataset,
        batch_size=16,
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
        logger.info(f"  Data shape: {batch['data'].shape}")
        logger.info(f"  Data dtype: {batch['data'].dtype}")
        logger.info(f"  Data device: {batch['data'].device}")
        logger.info(f"  Channel mask shape: {batch['channel_mask'].shape}")
        # logger.info(f"  Channel mask:\n{batch['channel_mask']}")
        
        # Show channel-wise statistics
        data = batch['data']
        logger.info(f"\n  Channel-wise statistics:")
        channel_names = ['vx', 'vy', 'vz', 'scalar_1', 'scalar_2', 'scalar_3']
        for ch_idx, ch_name in enumerate(channel_names):
            ch_data = data[..., ch_idx]
            logger.info(f"    {ch_name}: min={ch_data.min():.4f}, max={ch_data.max():.4f}, "
                       f"mean={ch_data.mean():.4f}, std={ch_data.std():.4f}")
        
        if i >= 2:  # Only show first 3 batches
            break
    
    logger.info("\n" + "="*50)
    logger.info("Data loading test completed successfully!")
    logger.info("="*50)