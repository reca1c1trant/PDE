# 分阶段训练方案

## 概述

不修改原始代码（encoder.py, pipeline.py, train.py），通过**新建独立训练脚本**实现分阶段训练。

## 三阶段设计

```
阶段1: Encoder/Decoder 预热 (冻结Transformer)
       ↓ pretrain_from
阶段2: Transformer 训练 (冻结Encoder/Decoder)
       ↓ pretrain_from
阶段3: 端到端微调 (全部解冻，小LR，nRMSE)
```

## 文件结构

```
train_staged.py          # 统一的分阶段训练脚本
configs/
  stage1.yaml           # 阶段1配置
  stage2.yaml           # 阶段2配置
  stage3.yaml           # 阶段3配置
```

## 阶段详情

### 阶段1：Encoder/Decoder 预热

**目标**：让Encoder学会提取PDE特征，Decoder学会重建

**策略**：
- 冻结Transformer所有参数
- 只训练Encoder + Decoder
- 使用MSE loss（大误差敏感，快速收敛）
- 较大LR (1e-4)

```yaml
# stage1.yaml
stage: 1
freeze:
  transformer: true
  encoder: false
  decoder: false
training:
  max_steps: 2000
  learning_rate: 1.0e-4
  loss_alpha: 0.0  # 纯MSE
  pretrain_from: null
save_dir: "./checkpoints_stage1"
```

### 阶段2：Transformer 训练

**目标**：让Transformer学会时序预测

**策略**：
- 冻结Encoder/Decoder
- 只训练Transformer
- 使用MSE loss
- 从阶段1加载权重

```yaml
# stage2.yaml
stage: 2
freeze:
  transformer: false
  encoder: true
  decoder: true
training:
  max_steps: 4000
  learning_rate: 1.0e-4
  loss_alpha: 0.0  # 纯MSE
  pretrain_from: "./checkpoints_stage1/best.pt"
save_dir: "./checkpoints_stage2"
```

### 阶段3：端到端微调

**目标**：全局优化，降低nRMSE

**策略**：
- 解冻所有参数
- 使用nRMSE loss
- 小LR防止破坏已学特征
- 从阶段2加载权重

```yaml
# stage3.yaml
stage: 3
freeze:
  transformer: false
  encoder: false
  decoder: false
training:
  max_steps: 2000
  learning_rate: 1.0e-5  # 小LR
  loss_alpha: 1.0  # 纯nRMSE
  nrmse_sigma: [...]  # 从compute_sigma.py获取
  pretrain_from: "./checkpoints_stage2/best.pt"
save_dir: "./checkpoints_stage3"
```

## 实现方案

### 方案A：单文件 + 3个yaml（推荐）

```bash
# 运行方式
python train_staged.py --config configs/stage1.yaml
python train_staged.py --config configs/stage2.yaml
python train_staged.py --config configs/stage3.yaml
```

`train_staged.py` 核心逻辑：
```python
# 根据config冻结参数
if config['freeze']['transformer']:
    for param in model.transformer.parameters():
        param.requires_grad = False

if config['freeze']['encoder']:
    for param in model.encoder_2d.parameters():
        param.requires_grad = False

if config['freeze']['decoder']:
    for param in model.decoder_2d.parameters():
        param.requires_grad = False

# 只优化requires_grad=True的参数
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), ...)
```

### 方案B：单文件自动三阶段

```bash
# 一次运行完成三阶段
python train_staged.py --config configs/staged_all.yaml
```

自动切换阶段，但灵活性较低。

## 推荐方案A

**优点**：
1. 不修改任何原始代码
2. 每个阶段可单独运行、调试
3. 可随时中断，从任意阶段继续
4. yaml清晰，易于调参

**实现文件**：
- `train_staged.py`：约200行，基于train.py简化
- `configs/stage1.yaml`
- `configs/stage2.yaml`
- `configs/stage3.yaml`

## 预期效果

| 阶段 | Steps | 预期Loss | 说明 |
|------|-------|----------|------|
| 1 | 2000 | MSE↓快 | Encoder/Decoder快速收敛 |
| 2 | 4000 | MSE继续↓ | Transformer学习时序 |
| 3 | 2000 | nRMSE↓ | 精细优化评估指标 |

总计8000步，但各阶段目标明确，训练更稳定。

---

**是否生成代码？** 确认后我将创建：
1. `train_staged.py`
2. `configs/stage1.yaml`
3. `configs/stage2.yaml`
4. `configs/stage3.yaml`
