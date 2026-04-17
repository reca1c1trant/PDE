#!/bin/bash
#SBATCH --job-name=bs_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=00:10:00
#SBATCH --output=logs_1d3d/bs_test_v2_%j.out
#SBATCH --error=logs_1d3d/bs_test_v2_%j.err

cd /home/msai/song0304/code/PDE

python3 -c "
import torch
from finetune.model_lora_v3 import PDELoRAModelV3
torch.set_float32_matmul_precision('high')

config = {
    'model': {
        'in_channels': 18, 'hidden_dim': 768, 'patch_size': 16,
        'num_layers': 12, 'num_heads': 12, 'dropout': 0.0,
        'encoder': {'stem_hidden': 128, 'stem_out': 256, 'use_cnn_pool': False},
        'intra_patch': {'num_layers': 2, 'temporal_window': 3, 'num_heads': 8},
        'na': {'base_kernel': 5},
        'decoder': {'stem_channels': 256, 'hidden_channels': 128, 'post_smooth_kernel': 5},
        'vector_channels': 3, 'scalar_channels': 15,
        'enable_1d': False, 'enable_3d': False,
        'lora': {'r': 16, 'alpha': 32, 'dropout': 0.05, 'target_modules': ['qkv','proj','gate_proj','up_proj','down_proj']},
    }
}

model = PDELoRAModelV3(config=config, pretrained_path='./checkpoints_v4_2donly_postsmooth/best_tf.pt').cuda().float()
model.train()
params = model.get_trainable_params()
print('=== 2D Burgers (160x160) ===')
for bs in [2, 4, 8, 12, 16, 20, 24]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        x = torch.randn(bs, 8, 160, 160, 18, device='cuda', dtype=torch.float32, requires_grad=False)
        out, m, s = model(x, return_normalized=True)
        loss = out.mean()
        loss.backward()
        for p in params: 
            if p.grad is not None: p.grad = None
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f'  bs={bs}: OK, peak={mem:.1f}GB')
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print(f'  bs={bs}: OOM')
            torch.cuda.empty_cache()
            break
        else:
            raise

del model; torch.cuda.empty_cache()

print()
print('=== 1D Burgers (160) ===')
config['model']['enable_1d'] = True
model = PDELoRAModelV3(config=config, pretrained_path='./checkpoints_v4_s_3d_with_best1-2d/best_tf.pt').cuda().float()
model.train()
for bs in [8, 16, 32, 64, 128, 256, 512]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        x = torch.randn(bs, 8, 160, 18, device='cuda', dtype=torch.float32, requires_grad=False)
        out, m, s = model(x, return_normalized=True)
        loss = out.mean()
        loss.backward()
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
