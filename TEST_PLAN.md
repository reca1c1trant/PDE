# Test Script Plan

## Overview

评估脚本，用于测试训练好的模型在未见数据上的预测能力。

## 评估逻辑

```
数据集有 T 个 timesteps (如 T=17)
随机选取起始点 t_start ∈ [0, T - input_steps - 1]
取 input_steps + 1 个连续帧: [t_start : t_start + input_steps + 1]
  - 前 input_steps 帧作为输入 (已知)
  - 最后 1 帧作为 ground truth
模型预测第 input_steps+1 帧
比较 prediction vs ground truth
```

**示例** (input_steps=4, T=17):
- 随机 t_start=5
- 取帧 [5,6,7,8,9]
- 输入: [5,6,7,8] → 模型预测 → output[3] (最后一帧的预测)
- Ground truth: [9]
- 计算 MSE(prediction, ground_truth)

## 变长输入支持

模型天然支持变长输入，原因：
1. **RoPE位置编码**: Llama使用旋转位置编码，支持任意长度（不超过max_position_embeddings=4096 tokens）
2. **Causal mask**: 只需根据输入timesteps动态生成对应大小的mask
3. **Batch内一致**: 同一batch所有sample长度相同，无需padding

**测试时**:
- 输入4步 → encoder产生 4×256=1024 tokens (2D: 4×16×16=1024)
- Causal mask: 4×4 block下三角
- 模型输出4步预测，取最后1步与ground truth比较

## Config 结构 (configs/test.yaml)

```yaml
# Test Configuration

# Model checkpoint
checkpoint: "./checkpoints_3b/best.pt"

# Model config (用于加载模型结构)
model_config: "./configs/llama_3b.yaml"

# Test dataset
dataset:
  path: "/path/to/test/data"

# Test parameters
test:
  input_steps: 4          # 用几个timestep作为输入
  batch_size: 1           # 每次评估的batch大小
  num_samples: 100        # 评估多少个样本 (null = 全部)
  seed: 42                # 随机种子 (保证可复现)

# Output
output:
  save_predictions: false  # 是否保存预测结果
  save_dir: "./test_results"
```

## 输出格式

```
============ Test Results ============
Checkpoint: ./checkpoints_3b/best.pt
Test samples: 100
Input steps: 4

Per-channel MSE:
  vx:  0.0123
  vy:  0.0145
  vz:  0.0132
  p:   0.0089
  rho: 0.0076
  T:   0.0098

Overall MSE:  0.0110
Overall RMSE: 0.1049

Per-sample stats:
  Min MSE: 0.0021
  Max MSE: 0.0342
  Std MSE: 0.0067
======================================
```

## 代码修改点

1. **pipeline.py**: `_create_causal_mask`已支持动态timesteps，无需修改

2. **test.py**: 新建测试脚本
   - 加载checkpoint
   - 随机采样timesteps
   - 前向推理
   - 计算metrics并打印

## 文件结构

```
configs/
  test.yaml          # 测试配置
test.py              # 测试脚本 (新建)
```

## 待确认

1. 评估指标除了MSE，还需要其他吗？(如RMSE、相对误差、物理量守恒等)

2. 是否需要可视化输出？(如保存prediction vs ground truth对比图)
