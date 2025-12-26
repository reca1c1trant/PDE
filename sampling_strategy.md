# Burgers2D 采样策略讨论

## 数据规模

| 参数 | 值 |
|------|------|
| 总样本数 | 100 |
| 训练集 | 90 样本 |
| 验证集 | 10 样本 |
| 每样本时间步 | 1000 |
| 每次使用 | 17 帧 |
| 可切片数/样本 | 1000 - 17 + 1 = 984 |
| **总可用片段** | 90 × 984 = **88,560** |

---

## 方案对比

### 方案A：完全随机（当前实现）

```python
# 每个step：随机选样本 → 随机选起始点
sample_idx = random.choice(90)
start_t = random.randint(0, 983)
```

**优点**：简单，时间多样性好
**缺点**：可能重复采样，无法保证覆盖所有片段

---

### 方案B：Epoch遍历所有片段

```python
# 每个epoch遍历所有 88,560 个片段
for sample_idx in range(90):
    for start_t in range(984):
        yield (sample_idx, start_t)
```

**优点**：完整覆盖
**缺点**：一个epoch太长（88k步），不实际

---

### 方案C：每样本固定采样数（推荐）

```python
# 每个epoch：每样本采样 K 个片段
K = 10  # 每样本10个片段
# 总步数/epoch = 90 × 10 = 900 片段
# batch_size=4 → 225 步/epoch
```

**优点**：
- 保证样本均衡（每个nu被同等使用）
- 时间多样性（每样本随机K个起始点）
- Epoch长度可控

---

## 推荐方案

**方案C：每样本采样K个片段**

```yaml
# 配置
sampling:
  clips_per_sample: 10   # 每样本每epoch采样10个片段
  # 90样本 × 10片段 = 900片段/epoch
  # batch_size=4 → 225步/epoch
  # 20个epoch → 4500步
```

**实现思路**：
```python
class BurgersDataset:
    def __init__(self, ..., clips_per_sample=10):
        self.clips_per_sample = clips_per_sample
        # 每个epoch开始时，为每个样本随机生成K个起始点

    def _generate_clips(self):
        """每个epoch调用一次"""
        self.clips = []
        for sample_idx in range(len(self.samples)):
            starts = np.random.choice(984, self.clips_per_sample, replace=False)
            for start_t in starts:
                self.clips.append((sample_idx, start_t))
        np.random.shuffle(self.clips)

    def __len__(self):
        return len(self.clips)  # 90 × 10 = 900
```

---

## 训练预算估算

| 参数 | 值 |
|------|------|
| clips_per_sample | 10 |
| 片段数/epoch | 900 |
| batch_size | 4 |
| 步数/epoch | 225 |
| 总epochs | 20 |
| **总训练步数** | **4,500** |

这与当前配置的 `max_steps=5000` 基本吻合。

---

## 是否需要修改代码？

**当前实现**（完全随机）对于5000步训练是可以接受的：
- 5000步 × batch_size 4 = 20,000 片段
- 平均每样本被采样 222 次
- 覆盖率足够

**如果你希望更严格的均衡采样**，我可以修改 `dataset_burgers.py` 实现方案C。

---

**请告诉我是否需要修改采样逻辑？**
