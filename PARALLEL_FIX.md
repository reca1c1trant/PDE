# 并行训练问题修复方案

## 一、当前错误分析

### OOM错误
```
CUDA out of memory. Tried to allocate 26.01 GiB.
GPU 0 has a total capacity of 44.43 GiB of which 18.00 GiB is free.
```

**原因**：7B模型 + DDP需要复制梯度 → 单卡46GB不够

**解决方案**：使用FSDP（Fully Sharded Data Parallel）替代DDP

---

## 二、修复方案

### 1. 使用FSDP替代DDP

| 方案 | 显存占用 | 说明 |
|------|---------|------|
| DDP | 模型×N卡 | 每卡完整模型+梯度 |
| **FSDP** | 模型/N卡 | 模型分片到各卡 |

FSDP可将7B模型分片到8卡，每卡只需~10GB模型参数。

### 2. 指定torch_dtype解决FlashAttention警告

```python
# pipeline.py - LlamaConfig中添加
llama_config = LlamaConfig(
    ...,
    torch_dtype=torch.bfloat16,  # 新增
)
```

### 3. 主进程控制print

```python
# 使用accelerator.is_main_process
if accelerator.is_main_process:
    logger.info(...)
    print(...)
```

### 4. 同步屏障

```python
# 关键点添加同步
accelerator.wait_for_everyone()
```

### 5. Loss聚合（all_reduce）

```python
# 多卡loss平均
loss = accelerator.gather(loss).mean()
```

---

## 三、具体修改

### config.yaml 新增
```yaml
training:
  # FSDP配置
  use_fsdp: true
  fsdp_sharding_strategy: "FULL_SHARD"  # 完全分片
```

### train.py 修改

```python
# Accelerator初始化
from accelerate import FullyShardedDataParallelPlugin
from torch.distributed.fsdp import ShardingStrategy

fsdp_plugin = FullyShardedDataParallelPlugin(
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    cpu_offload=False,
)

accelerator = Accelerator(
    mixed_precision="bf16",
    fsdp_plugin=fsdp_plugin,  # 使用FSDP
    log_with="wandb"
)
```

### pipeline.py 修改

```python
# LlamaConfig添加torch_dtype
llama_config = LlamaConfig(
    ...,
    torch_dtype=torch.bfloat16,
)

# 模型初始化后转换dtype
self.transformer = LlamaModel(llama_config)
self.transformer = self.transformer.to(torch.bfloat16)
```

### 主进程控制（train.py中已有，pipeline.py需修改）

```python
# pipeline.py - _log_info中
def _log_info(self, ...):
    # 只在rank 0打印
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(...)
```

---

## 四、完整修改清单

| 文件 | 修改内容 |
|------|---------|
| `config.yaml` | 添加 `use_fsdp: true` |
| `train.py` | 添加FSDP plugin，添加同步屏障，loss聚合 |
| `pipeline.py` | 添加 `torch_dtype=torch.bfloat16`，主进程print控制 |

---

## 五、预估效果

| 指标 | DDP（当前） | FSDP（修复后） |
|------|-----------|---------------|
| 单卡显存 | ~35GB (OOM) | ~15-20GB |
| Batch Size | 1 | 2-4 |
| 通信开销 | 低 | 中等 |

---

## 六、额外优化建议

### 1. CPU Offload（可选）
```python
fsdp_plugin = FullyShardedDataParallelPlugin(
    cpu_offload=True,  # 将优化器状态offload到CPU
)
```
进一步减少显存，但会降低速度。

### 2. Activation Checkpointing
已启用 `gradient_checkpointing: true`，保持。

### 3. 内存碎片优化
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## 七、待确认

1. **是否使用FSDP**？（推荐是）
2. **是否使用CPU Offload**？（batch_size够用则不需要）
3. **batch_size调整**？使用FSDP后可尝试增加到2-4

确认后我开始修改代码。
