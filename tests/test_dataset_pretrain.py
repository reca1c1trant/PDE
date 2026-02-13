"""
Test dataset_pretrain.py with fake 1D and 2D data.
Verifies: metadata loading, clip generation, load_clip shapes, sampler no cross-dataset mixing.
"""
import sys
sys.path.insert(0, '/home/msai/song0304/code/PDE')

import tempfile
import numpy as np
import h5py
from pathlib import Path

from pretrain.dataset_pretrain import (
    DatasetConfig, SingleDataset, PretrainDataset,
    PretrainSampler, pretrain_collate_fn,
)


def create_fake_2d(path, N=20, T=21, H=128, W=128):
    """Create fake 2D CFD unified file: vector [N,T,H,W,3] + scalar [N,T,H,W,2]."""
    with h5py.File(path, 'w') as f:
        f.create_dataset('vector', data=np.random.randn(N, T, H, W, 3).astype(np.float32))
        f.create_dataset('scalar', data=np.random.randn(N, T, H, W, 2).astype(np.float32))
        f.create_dataset('scalar_indices', data=np.array([4, 12], dtype=np.int32))  # density, pressure
    print(f"  Created 2D: {path} [N={N}, T={T}, H={H}, W={W}]")


def create_fake_1d(path, N=20, T=101, X=1024):
    """Create fake 1D CFD unified file: vector [N,T,X,3] + scalar [N,T,X,2]."""
    with h5py.File(path, 'w') as f:
        f.create_dataset('vector', data=np.random.randn(N, T, X, 3).astype(np.float32))
        f.create_dataset('scalar', data=np.random.randn(N, T, X, 2).astype(np.float32))
        f.create_dataset('scalar_indices', data=np.array([4, 12], dtype=np.int32))
    print(f"  Created 1D: {path} [N={N}, T={T}, X={X}]")


def create_fake_scalar_only(path, N=20, T=21, H=128, W=128):
    """Create fake scalar-only dataset (like diffusion-reaction)."""
    with h5py.File(path, 'w') as f:
        f.create_dataset('scalar', data=np.random.randn(N, T, H, W, 2).astype(np.float32))
        f.create_dataset('scalar_indices', data=np.array([2, 3], dtype=np.int32))  # conc_u, conc_v
    print(f"  Created scalar-only 2D: {path} [N={N}, T={T}, H={H}, W={W}]")


def test_single_dataset_1d():
    """Test SingleDataset with 1D data."""
    print("\n[Test 1] SingleDataset 1D")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / '1d_cfd.hdf5'
        create_fake_1d(path, N=10, T=50, X=256)

        config = DatasetConfig(name='1d_cfd', path=str(path), vector_dim=1)
        ds = SingleDataset(config, split='train', temporal_length=9, clips_ratio=0.3)

        clips = ds.generate_clips(epoch=0)
        print(f"  Train clips: {len(clips)}")
        assert len(clips) > 0

        # Load one clip
        sample_idx, start_t = clips[0]
        result = ds.load_clip(sample_idx, start_t)
        data = result['data']
        mask = result['channel_mask']

        print(f"  data shape: {data.shape}")
        print(f"  channel_mask: {mask}")
        assert data.shape == (9, 256, 18), f"Expected (9, 256, 18), got {data.shape}"
        assert mask[0] == 1.0  # Vx active
        assert mask[1] == 0.0  # Vy inactive
        assert mask[2] == 0.0  # Vz inactive
        assert mask[3 + 4] == 1.0   # density (scalar idx 4)
        assert mask[3 + 12] == 1.0  # pressure (scalar idx 12)
        print("  PASSED")


def test_single_dataset_2d():
    """Test SingleDataset with 2D data (regression check)."""
    print("\n[Test 2] SingleDataset 2D")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / '2d_cfd.hdf5'
        create_fake_2d(path, N=10, T=21, H=64, W=64)

        config = DatasetConfig(name='2d_cfd', path=str(path), vector_dim=2)
        ds = SingleDataset(config, split='train', temporal_length=9, clips_ratio=0.3)

        clips = ds.generate_clips(epoch=0)
        sample_idx, start_t = clips[0]
        result = ds.load_clip(sample_idx, start_t)
        data = result['data']

        print(f"  data shape: {data.shape}")
        assert data.shape == (9, 64, 64, 18), f"Expected (9, 64, 64, 18), got {data.shape}"
        print("  PASSED")


def test_single_dataset_scalar_only():
    """Test SingleDataset with scalar-only 2D data (no vector)."""
    print("\n[Test 3] SingleDataset scalar-only 2D")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / 'diff_react.hdf5'
        create_fake_scalar_only(path, N=10, T=21, H=64, W=64)

        config = DatasetConfig(name='diff_react', path=str(path), vector_dim=0)
        ds = SingleDataset(config, split='train', temporal_length=9)

        clips = ds.generate_clips(epoch=0)
        sample_idx, start_t = clips[0]
        result = ds.load_clip(sample_idx, start_t)
        data = result['data']
        mask = result['channel_mask']

        print(f"  data shape: {data.shape}")
        assert data.shape == (9, 64, 64, 18)
        assert mask[0] == 0.0  # no vector
        assert mask[3 + 2] == 1.0  # conc_u
        assert mask[3 + 3] == 1.0  # conc_v
        print("  PASSED")


def test_pretrain_dataset_mixed():
    """Test PretrainDataset with mixed 1D + 2D datasets."""
    print("\n[Test 4] PretrainDataset mixed 1D + 2D")
    with tempfile.TemporaryDirectory() as tmpdir:
        path_1d = Path(tmpdir) / '1d_cfd.hdf5'
        path_2d = Path(tmpdir) / '2d_cfd.hdf5'
        create_fake_1d(path_1d, N=10, T=50, X=128)
        create_fake_2d(path_2d, N=10, T=21, H=64, W=64)

        configs = [
            DatasetConfig(name='1d_cfd', path=str(path_1d), vector_dim=1),
            DatasetConfig(name='2d_cfd', path=str(path_2d), vector_dim=2),
        ]

        dataset = PretrainDataset(configs, split='train', temporal_length=9, clips_ratio=0.3)
        print(f"  Total clips: {len(dataset)}")
        assert len(dataset) > 0

        # Load individual clips and check shapes
        for i in range(min(5, len(dataset))):
            item = dataset[i]
            name = item['dataset_name']
            shape = item['data'].shape
            if name == '1d_cfd':
                assert len(shape) == 3, f"1D should be 3D tensor, got {shape}"
                assert shape[-1] == 18
            else:
                assert len(shape) == 4, f"2D should be 4D tensor, got {shape}"
                assert shape[-1] == 18
            print(f"  clip {i}: {name} -> {tuple(shape)}")

        print("  PASSED")


def test_sampler_no_cross_mixing():
    """Test that sampler never mixes different datasets in one batch."""
    print("\n[Test 5] Sampler no cross-dataset mixing")
    with tempfile.TemporaryDirectory() as tmpdir:
        path_1d = Path(tmpdir) / '1d_cfd.hdf5'
        path_2d = Path(tmpdir) / '2d_cfd.hdf5'
        # Use small N and short T so clips < batch_size (triggers padding)
        create_fake_1d(path_1d, N=8, T=15, X=128)
        create_fake_2d(path_2d, N=8, T=15, H=64, W=64)

        configs = [
            DatasetConfig(name='1d_cfd', path=str(path_1d), vector_dim=1),
            DatasetConfig(name='2d_cfd', path=str(path_2d), vector_dim=2),
        ]

        batch_size = 10
        dataset = PretrainDataset(configs, split='train', temporal_length=9, clips_ratio=0.5)
        sampler = PretrainSampler(dataset, batch_size=batch_size, shuffle=True, seed=42)

        print(f"  Total clips: {len(dataset)}, Batches: {len(sampler)}")

        n_batches_checked = 0
        for batch_indices in sampler:
            # All clips in this batch must be from same dataset
            names = set()
            for idx in batch_indices:
                clip_name, _, _ = dataset.clips[idx]
                names.add(clip_name)

            assert len(names) == 1, f"Mixed datasets in batch: {names}"
            assert len(batch_indices) == batch_size, f"Batch size {len(batch_indices)} != {batch_size}"
            n_batches_checked += 1

        print(f"  Checked {n_batches_checked} batches, all same-dataset, all size={batch_size}")
        print("  PASSED")


def test_collate_same_dataset():
    """Test that collate_fn works when batch is from same dataset."""
    print("\n[Test 6] Collate within same dataset")
    with tempfile.TemporaryDirectory() as tmpdir:
        path_1d = Path(tmpdir) / '1d_cfd.hdf5'
        create_fake_1d(path_1d, N=5, T=20, X=128)

        configs = [DatasetConfig(name='1d_cfd', path=str(path_1d), vector_dim=1)]
        dataset = PretrainDataset(configs, split='train', temporal_length=9, clips_ratio=0.5)
        sampler = PretrainSampler(dataset, batch_size=4, shuffle=False, seed=42)

        # Get first batch
        batch_indices = next(iter(sampler))
        items = [dataset[i] for i in batch_indices]
        batch = pretrain_collate_fn(items)

        print(f"  data: {batch['data'].shape}")
        print(f"  channel_mask: {batch['channel_mask'].shape}")
        print(f"  names: {batch['dataset_names']}")
        assert batch['data'].shape == (4, 9, 128, 18)
        assert batch['channel_mask'].shape == (4, 18)
        print("  PASSED")


if __name__ == '__main__':
    print("=" * 60)
    print("Dataset Pretrain Tests")
    print("=" * 60)

    try:
        test_single_dataset_1d()
        test_single_dataset_2d()
        test_single_dataset_scalar_only()
        test_pretrain_dataset_mixed()
        test_sampler_no_cross_mixing()
        test_collate_same_dataset()
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("ALL DATASET TESTS PASSED!")
    print("=" * 60)
