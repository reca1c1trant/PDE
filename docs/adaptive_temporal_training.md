# 自适应时间步长 + 混合数据集训练方案

## 1. 问题背景

| 数据集 | 样本数 | 时间步数 T | 空间分辨率 |
|--------|--------|------------|------------|
| 旧数据集 | ~1000 | 101 | 128×128 |
| 新数据集 (CFD_2000) | 2000 | 21 | 128×128 |

**矛盾**：
- 当前模型需要 `temporal_length=16`（实际需要 17 个点用于 causal AR）
- 新数据集只有 21 个点，只能产生 5 个 clips，无法填满 batch_size=8
- 同样本 batching 约束必须保留

## 2. 核心思路

### 2.1 自适应时间步长

根据样本的总时间步数 `T` 动态选择 `temporal_length`：

```
if T < 24:  # 不足以支持 16+8 的 buffer
    temporal_length = 8   → 需要 9 个点/clip → clips = T - 9 + 1
else:
    temporal_length = 16  → 需要 17 个点/clip → clips = T - 17 + 1
```

| 数据集 | T | temporal_length | 可用 clips |
|--------|---|-----------------|------------|
| 旧数据集 | 101 | 16 | 85 |
| 新数据集 | 21 | 8 | 13 |

### 2.2 混合比例控制

目标：新数据集的 clips 数 ≥ 旧数据集

```
新数据集: 2000 samples × clips_per_sample_new
旧数据集: 1000 samples × clips_per_sample_old

设 ratio = 新 / 旧 = 2:1

如果 clips_per_sample_new = 13 (用满所有clips)
则 新: 2000 × 13 = 26000 clips
   旧: 需要 13000 clips → clips_per_sample_old = 13
```

## 3. 实现方案

### 3.1 数据结构

```python
# 每个 clip 的元信息
ClipInfo = {
    'sample_idx': int,      # 样本索引
    'start_t': int,         # 起始时间步
    'temporal_length': int, # 8 或 16
    'data_source': str,     # 'old' 或 'new'
}
```

### 3.2 Dataset 改动

```python
class PDEDataset:
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        adaptive_temporal: bool = True,  # 新增：是否自适应
        temporal_threshold: int = 24,    # 新增：T < 24 用 temporal_length=8
        clips_per_sample: int = None,    # None = 用满所有clips
        ...
    ):
        self.adaptive_temporal = adaptive_temporal
        self.temporal_threshold = temporal_threshold

    def _generate_clips(self):
        clips = []
        for sample_idx, sample_info in enumerate(self.samples):
            T = sample_info['num_timesteps']

            # 自适应选择 temporal_length
            if self.adaptive_temporal and T < self.temporal_threshold:
                temporal_length = 8
            else:
                temporal_length = 16

            # 计算可用起点
            required_length = temporal_length + 1  # causal AR 需要 +1
            num_available = T - required_length + 1

            # 生成 clips
            for start_t in self._select_starts(num_available):
                clips.append({
                    'sample_idx': sample_idx,
                    'start_t': start_t,
                    'temporal_length': temporal_length,
                })
        return clips
```

### 3.3 Sampler 改动

**关键**：同一个 batch 必须有相同的 `temporal_length`

```python
class AdaptiveSampler:
    """
    按 (dimension, temporal_length) 分组的采样器
    """
    def __init__(self, dataset, batch_size, new_old_ratio=2.0):
        self.batch_size = batch_size

        # 按 temporal_length 分组
        self.groups = {
            8: [],   # temporal_length=8 的 clip indices
            16: [],  # temporal_length=16 的 clip indices
        }

        for idx, clip in enumerate(dataset.clips):
            tl = clip['temporal_length']
            self.groups[tl].append(idx)

        # 根据 new_old_ratio 调整采样权重
        # ...

    def __iter__(self):
        # 交替从两个组采样，保证比例
        # 每个 batch 内的 clips 必须来自同一 temporal_length 组
        ...
```

### 3.4 Collate Function

```python
def adaptive_collate_fn(batch):
    """
    处理不同 temporal_length 的 batch
    """
    # batch 内的 temporal_length 应该一致（由 sampler 保证）
    temporal_lengths = [item['temporal_length'] for item in batch]
    assert len(set(temporal_lengths)) == 1, "Batch must have same temporal_length"

    data = torch.stack([item['data'] for item in batch])  # [B, T+1, H, W, C]

    return {
        'data': data,
        'temporal_length': temporal_lengths[0],
        'channel_mask': ...,
    }
```

### 3.5 训练循环改动

```python
for batch in train_loader:
    data = batch['data']  # [B, T+1, H, W, C], T 可能是 8 或 16
    temporal_length = batch['temporal_length']

    # Causal AR: input → target
    input_data = data[:, :-1]   # [B, T, H, W, C]
    target_data = data[:, 1:]   # [B, T, H, W, C]

    # 模型已经支持动态 T（RoPE + 动态 encoder）
    output = model(input_data)  # [B, T, H, W, C]

    # Loss 计算不变
    loss = compute_nrmse_loss(output, target_data, channel_mask)
```

## 4. 配置文件示例

```yaml
# configs/mixed_adaptive.yaml
model_name: "pde_e2e_mixed"

dataset:
  paths:
    - "./data/old_dataset/"       # 旧数据集
    - "./2D_CFD_2000_final.hdf5"  # 新数据集

  adaptive_temporal: true
  temporal_threshold: 24  # T < 24 → temporal_length=8

  # 比例控制
  clips_per_sample:
    old: 13   # 旧数据集每样本抽 13 clips
    new: 13   # 新数据集每样本用满 13 clips (实际只有13个)

  new_old_ratio: 2.0  # 新数据集采样频率是旧的 2 倍

dataloader:
  batch_size: 8
  num_workers: 4

model:
  # ... 模型配置不变
```

## 5. 预期效果

| 指标 | 值 |
|------|-----|
| 旧数据集 clips/epoch | 1000 × 13 = 13,000 |
| 新数据集 clips/epoch | 2000 × 13 = 26,000 |
| 新:旧 比例 | 2:1 |
| 总 clips/epoch | 39,000 |
| Batch size | 8 |
| Steps/epoch | ~4,875 |

## 6. 实现优先级

1. **Phase 1**: 修改 `dataset.py` 支持自适应 temporal_length
2. **Phase 2**: 修改 Sampler 支持 temporal_length 分组
3. **Phase 3**: 实现混合数据集加载
4. **Phase 4**: 比例控制与 collate 函数

## 7. 风险与对策

| 风险 | 对策 |
|------|------|
| 不同 T 的梯度量级不同 | 可考虑按 T 归一化 loss |
| GPU 利用率因 T 变化而波动 | T=8 batch 可考虑增大 batch_size |
| 模型对短序列泛化能力 | 评估时分别测试 T=8 和 T=16 |
