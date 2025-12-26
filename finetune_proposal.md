# Burgers2D LoRA Finetune 方案 (修订版)

> 基于预训练PDE模型，使用LoRA微调 + 纯PDE Residual Loss

---

## 一、任务理解

**目标**：在没有ground truth的情况下，仅通过PDE物理约束对预训练模型进行LoRA微调，使其学会预测2D Burgers方程的演化。

**核心约束**：
- ❌ 无法使用解析解作为监督信号
- ✅ 只能通过PDE residual反传梯度
- ✅ 边界条件作为已知输入（不进入模型，只用于梯度计算）

---

## 二、数据流设计

```
┌─────────────────────────────────────────────────────────────────┐
│                         Training Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Burgers2D HDF5                                                  │
│  ├── interior: [T=1000, 128, 128, 2]  →  补齐到6通道             │
│  └── boundary: left/right/bottom/top   →  直接用于PDE Loss       │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  input_data [B, 16, 128, 128, 6]                         │   │
│  │       ↓                                                   │   │
│  │  ┌─────────┐   ┌───────────────────┐   ┌─────────┐       │   │
│  │  │ Encoder │ → │ Transformer+LoRA  │ → │ Decoder │       │   │
│  │  │ (frozen)│   │ (frozen + LoRA)   │   │ (frozen)│       │   │
│  │  └─────────┘   └───────────────────┘   └─────────┘       │   │
│  │       ↓                                                   │   │
│  │  output [B, 16, 128, 128, 6]                             │   │
│  └──────────────────────────────────────────────────────────┘   │
│                          ↓                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  时间对齐：prepend input[:, 0:1] 到 output 最前面          │   │
│  │  → pred_with_t0 [B, 17, 128, 128, 6]                     │   │
│  │  → 提取 u, v 两个通道 [B, 17, 128, 128, 2]               │   │
│  └──────────────────────────────────────────────────────────┘   │
│                          ↓                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  burgers_pde_loss_upwind(pred, boundaries, ν)            │   │
│  │  → 使用GT边界计算ghost cell和空间导数                     │   │
│  │  → 返回 PDE residual loss                                │   │
│  └──────────────────────────────────────────────────────────┘   │
│                          ↓                                       │
│               反传梯度 → 只更新 LoRA 参数                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 三、关键设计决策

### 3.1 通道处理

Burgers2D只有2通道 `[u, v]`，需补齐到6通道以匹配预训练模型。

**Encoder V2 的分支结构**：
```python
# PDE2DEncoderV2.forward():
x_vec = x[..., :3]  # vector branch 处理前3通道
x_sca = x[..., 3:]  # scalar branch 处理后3通道

x_vec = self.vector_conv(x_vec)  # [B*T, 256, 16, 16]
x_sca = self.scalar_conv(x_sca)  # [B*T, 256, 16, 16]
x = torch.cat([x_vec, x_sca], dim=1)  # fusion
```

**Burgers2D 通道映射**：
```python
# u, v 是速度分量 → 放入 vector channels
# 无标量场 → scalar channels 全0

data_6ch = torch.zeros(B, T, H, W, 6)
data_6ch[..., 0] = u   # vx = u
data_6ch[..., 1] = v   # vy = v
data_6ch[..., 2] = 0   # vz = 0 (2D问题)
data_6ch[..., 3:] = 0  # scalar channels 全0

channel_mask = [1, 1, 0, 0, 0, 0]  # 标记真实通道
```

**关于scalar分支处理全0**：
- 预训练时部分样本的scalar也可能接近0（已通过channel_mask处理）
- Conv层有bias，全0输入会产生非零输出，但fusion和Transformer会学会适当加权
- **Encoder/Decoder保持freeze**，复用预训练的空间特征提取能力

### 3.2 LoRA配置

在Transformer的**所有可训练层**添加LoRA：

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        # Attention
        "q_proj", "k_proj", "v_proj", "o_proj",
        # MLP
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
)
```

**可训练参数**：只有LoRA适配器
**冻结参数**：Encoder + Transformer base + Decoder

### 3.3 Forward流程 (参考train_e2e.py)

**完整训练步骤**：

```python
# ========== 1. 数据准备 ==========
# 从dataset获取17帧数据（与预训练一致）
data = batch['data']  # [B, 17, H, W, 6]，已补齐6通道
channel_mask = batch['channel_mask']  # [B, 6] = [1,1,0,0,0,0]
boundaries = batch['boundaries']  # dict: left/right/bottom/top
nu = batch['nu']  # [B] 粘度系数

# 分割 input 和 target（因果AR）
input_data = data[:, :-1]   # [B, 16, H, W, 6] 对应 t=0,...,15
target_data = data[:, 1:]   # [B, 16, H, W, 6] 对应 t=1,...,16

# ========== 2. 模型前向 ==========
# 转为bf16，与预训练一致
input_data = input_data.to(dtype=torch.bfloat16)
output = model(input_data)  # [B, 16, H, W, 6] 对应 t=1,...,16

# ========== 3. 时间对齐（用于PDE loss） ==========
# prepend input的第一帧到output前面
t0_frame = input_data[:, 0:1]  # [B, 1, H, W, 6] 对应 t=0
pred_with_t0 = torch.cat([t0_frame, output], dim=1)  # [B, 17, H, W, 6]

# 提取 u, v 通道
pred_uv = pred_with_t0[..., :2].float()  # [B, 17, H, W, 2]

# ========== 4. 边界条件对齐 ==========
# boundaries也需要17帧，与pred_with_t0对应
boundary_left = boundaries['left'][:, :17]    # [B, 17, H, 1, 2]
boundary_right = boundaries['right'][:, :17]  # [B, 17, H, 1, 2]
boundary_bottom = boundaries['bottom'][:, :17]  # [B, 17, 1, W, 2]
boundary_top = boundaries['top'][:, :17]      # [B, 17, 1, W, 2]

# ========== 5. 计算PDE Loss ==========
# dt = 1/999 (Burgers数据集的时间步长)
pde_loss, loss_u, loss_v, res_u, res_v = burgers_pde_loss_upwind(
    pred=pred_uv,
    boundary_left=boundary_left,
    boundary_right=boundary_right,
    boundary_bottom=boundary_bottom,
    boundary_top=boundary_top,
    nu=nu,
    dt=1/999
)

# ========== 6. 反向传播 ==========
loss = lambda_pde * pde_loss
accelerator.backward(loss)
```

**关键点**：
- 数据切分与预训练完全一致：`input = data[:, :-1]`, `target = data[:, 1:]`
- 模型输入16帧，输出16帧
- PDE loss需要17帧（多一个t=0用于计算时间导数）
- 边界条件也需要对应的17帧

### 3.4 边界条件处理

边界条件**不进入模型**，只用于PDE loss中计算ghost cells：

```python
# pde_loss.py 中的 ghost cell 外推：
def pad_with_boundaries(interior, boundary_left, boundary_right, ...):
    # 外推ghost cells: ghost = 2 * boundary - interior_edge
    ghost_left = 2 * boundary_left - interior[..., 0]
    ghost_right = 2 * boundary_right - interior[..., -1]
    # ...构造 padded grid [B, T, H+2, W+2]
```

**数据集需要返回的边界格式**：
```python
# 从HDF5读取原始边界 [T=1000, H, 1, 2] 或 [T=1000, 1, W, 2]
# 切片到对应的17帧
boundaries = {
    'left':   [B, 17, 128, 1, 2],   # x=0 边界
    'right':  [B, 17, 128, 1, 2],   # x=1 边界
    'bottom': [B, 17, 1, 128, 2],   # y=0 边界
    'top':    [B, 17, 1, 128, 2],   # y=1 边界
}
```

**注意**：PDE loss内部会从t=1开始计算（因为需要t-1计算时间导数），所以17帧边界对应17帧预测。

### 3.5 Loss设计

```python
# 默认：纯PDE loss
loss = λ_pde * L_PDE

# 预留接口：可选加入data loss（如果未来有ground truth）
loss = λ_data * L_nRMSE + λ_pde * L_PDE

# 默认权重
λ_data = 0.0  # 不使用
λ_pde = 1.0
```

---

## 四、评估指标

### 4.1 PDE Residual

$$\text{Residual}_u = u_t + uu_x + vu_y - \nu(u_{xx} + u_{yy})$$
$$\text{Residual}_v = v_t + uv_x + vv_y - \nu(v_{xx} + v_{yy})$$

$$\mathcal{L}_{PDE} = \|\text{Residual}_u\|_2^2 + \|\text{Residual}_v\|_2^2$$

### 4.2 约束误差

Burgers方程解析解满足 $u + v = 1.5$：

$$\mathcal{E}_{constraint} = \|u + v - 1.5\|_\infty$$

**注意**：这个约束用于**评估**，不用于训练（因为我们不应该知道解析解的性质）。

---

## 五、代码结构

```
finetune/
├── train_burgers_lora.py     # 主训练脚本
├── model_lora.py             # LoRA包装的PDECausalModel
├── dataset_burgers.py        # Burgers2D数据集加载
├── pde_loss.py               # PDE residual loss (已有)
└── configs/
    └── finetune_burgers.yaml # 配置文件
```

---

## 六、配置文件模板

```yaml
# finetune_burgers.yaml

# 模型
model:
  pretrained_path: "./checkpoints_e2e_medium_v2/best.pt"

  # 继承预训练的encoder配置
  in_channels: 6
  encoder:
    version: "v2"
    channels: [64, 128, 256]
    use_resblock: true

  # Transformer配置（必须与预训练一致）
  transformer:
    hidden_size: 768
    num_hidden_layers: 10
    num_attention_heads: 12
    num_key_value_heads: 4
    intermediate_size: 3072
    hidden_act: "silu"
    max_position_embeddings: 4096
    rms_norm_eps: 1.0e-5
    rope_theta: 500000.0
    attention_dropout: 0.0

  # LoRA配置
  lora:
    r: 16
    alpha: 32
    dropout: 0.05
    target_modules:
      - "q_proj"
      - "k_proj"
      - "v_proj"
      - "o_proj"
      - "gate_proj"
      - "up_proj"
      - "down_proj"

# 数据集
dataset:
  path: "./burgers2d_nu0.1_0.15_res128_t1000_n100.h5"
  temporal_length: 16  # 每个样本切出16+1=17帧
  train_ratio: 0.9
  seed: 42

# PDE物理参数
physics:
  dt: 0.001001001  # 1/999，Burgers数据集的时间步长
  dx: 0.0078125    # 1/128，空间步长

# DataLoader
dataloader:
  batch_size: 4
  num_workers: 4
  pin_memory: true

# 训练
training:
  max_steps: 5000
  warmup_steps: 100
  learning_rate: 1.0e-4
  weight_decay: 1.0e-4
  grad_clip: 1.0

  # Loss权重
  lambda_data: 0.0   # 默认不使用data loss
  lambda_pde: 1.0

  mixed_precision: "bf16"
  gradient_accumulation_steps: 1
  eval_every_steps: 50

# 日志
logging:
  project: "pde-burgers-lora"
  entity: "your-wandb-entity"
  save_dir: "./checkpoints_burgers_lora"
  log_interval: 10
```

---

## 七、训练策略

### 7.1 Warmup阶段

前100步使用较小学习率，让LoRA适配器稳定：

```python
def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    else:
        # cosine decay
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return 0.5 * (1 + cos(π * progress))
```

### 7.2 梯度裁剪

PDE loss涉及二阶导数，梯度可能很大：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 7.3 数值稳定性

```python
# 在PDE loss计算中添加eps
eps = 1e-8
u_t = (u[:, 1:] - u[:, :-1]) / (dt + eps)
```

---

## 八、补充建议与提醒

### 8.1 潜在风险

| 风险 | 说明 | 缓解措施 |
|------|------|----------|
| **收敛到平凡解** | 模型可能学到 $u=v=const$（满足 $u_t=0$） | 监控预测的空间变化，添加正则项 |
| **梯度爆炸** | 二阶导数放大误差 | 严格的gradient clipping |
| **边界依赖** | 模型过度依赖边界信息 | 评估时检查内部域独立预测能力 |
| **过拟合ν** | 只在 $\nu \in [0.1, 0.15]$ 训练 | 评估时测试其他ν值的泛化 |

### 8.2 调试建议

1. **先用1-2个样本过拟合**：验证pipeline正确性
2. **监控每个残差项**：分别看 $u_t$, $uu_x$, $vu_y$, $\nu u_{xx}$ 等项的量级
3. **可视化预测**：每100步保存预测场的图像
4. **检查LoRA权重**：确保LoRA参数在更新

### 8.3 评估时额外检查

```python
# 1. 约束误差（不用于训练）
constraint_error = torch.abs(u + v - 1.5).max()

# 2. 空间变化（确保不是平凡解）
spatial_std = pred.std(dim=(2, 3))  # 应该 > 0

# 3. 时间变化
temporal_diff = (pred[:, 1:] - pred[:, :-1]).abs().mean()
```

### 8.4 关于纯PDE Loss训练的担忧

**问题**：PINN文献显示纯物理约束可能收敛困难，因为：
- 解空间太大
- PDE残差为0有无穷多解（包括非物理解）

**Burgers方程的有利因素**：
- 有解析解，解是光滑的
- 初始条件（input的第一帧）提供了隐式约束
- 边界条件提供了额外约束

**建议**：如果训练不收敛，可以考虑：
1. 降低学习率
2. 增加warmup步数
3. 尝试引入少量带标签数据（λ_data > 0）

### 8.5 LoRA参数量估计

```
Transformer: h=768, L=10
每层LoRA:
  - q_proj: 768 × 768 → LoRA: 768×16 + 16×768 = 24,576
  - k_proj: 768 × 768 → LoRA: 24,576
  - v_proj: 768 × 768 → LoRA: 24,576
  - o_proj: 768 × 768 → LoRA: 24,576
  - gate_proj: 768 × 3072 → LoRA: 768×16 + 16×3072 = 61,440
  - up_proj: 768 × 3072 → LoRA: 61,440
  - down_proj: 3072 × 768 → LoRA: 61,440

每层总计: 24,576 × 4 + 61,440 × 3 = 98,304 + 184,320 = 282,624
10层总计: 2,826,240 ≈ 2.8M 可训练参数

对比原始Transformer: ~85M 参数
LoRA比例: 2.8M / 85M ≈ 3.3%
```

---

## 九、待实现检查清单

- [ ] `dataset_burgers.py`: 加载Burgers2D HDF5，补齐6通道，返回边界
- [ ] `model_lora.py`: 加载预训练权重，应用LoRA，freeze非LoRA参数
- [ ] `train_burgers_lora.py`: 训练循环，时间对齐，PDE loss计算
- [ ] `configs/finetune_burgers.yaml`: 配置文件
- [ ] 评估脚本：PDE residual + 约束误差

---

**请确认方案，确认后我将开始编写代码。**
