#!/bin/bash
#SBATCH --job-name=bs3d
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=00:05:00
#SBATCH --output=logs_1d3d/bs3d_%j.out
#SBATCH --error=logs_1d3d/bs3d_%j.err

cd /home/msai/song0304/code/PDE
python3 -c "
import torch, yaml
from finetune.model_lora_v3 import PDELoRAModelV3
torch.set_float32_matmul_precision('high')

with open('configs/finetune_burgers_3d_apebench_rescaled_norm.yaml') as f:
    config = yaml.safe_load(f)

model = PDELoRAModelV3(config=config, pretrained_path=config['model']['pretrained_path']).cuda().float()
model.train()

for bs in [4, 8, 16, 32, 64]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        x = torch.randn(bs, 8, 32, 32, 32, 18, device='cuda', dtype=torch.float32)
        out, m, s = model(x, return_normalized=True)
        out.mean().backward()
        for p in model.get_trainable_params():
            if p.grad is not None: p.grad = None
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f'  bs={bs}: OK, peak={mem:.1f}GB')
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print(f'  bs={bs}: OOM')
            break
        else:
            raise
"
