#!/bin/bash
#SBATCH --job-name=b3d_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=00:30:00
#SBATCH --output=logs_1d3d/burgers_3d_test_%j.out
#SBATCH --error=logs_1d3d/burgers_3d_test_%j.err

cd /home/msai/song0304/code/PDE

# 1. GT verification
echo "=== GT Verification ==="
python tools/test_gt_pde_burgers_3d_apebench.py

# 2. Test training (just a few steps)
echo ""
echo "=== Training Test (5 steps) ==="
python -c "
import torch, yaml
from finetune.model_lora_v3 import PDELoRAModelV3
from finetune.dataset_finetune import create_finetune_dataloaders
from finetune.pde_loss_verified import APEBenchBurgers3DPDELoss

torch.set_float32_matmul_precision('high')

with open('configs/finetune_burgers_3d_apebench_rescaled_norm.yaml') as f:
    config = yaml.safe_load(f)

# Create dataloader
train_loader, val_loader, train_sampler, _ = create_finetune_dataloaders(
    data_path=config['dataset']['path'],
    batch_size=config['dataloader']['batch_size'],
    num_workers=2, pin_memory=False,
    seed=42, temporal_length=9,
    train_ratio=0.8,
    vector_dim=config['dataset'].get('vector_dim', 3),
    val_time_interval=20,
)
print(f'Train batches: {len(train_sampler)}, Val batches: {len(val_loader)}')

# Load model
model = PDELoRAModelV3(
    config=config,
    pretrained_path=config['model']['pretrained_path'],
).cuda().float()
model.train()
print(f'Model loaded. Params: {sum(p.numel() for p in model.parameters()):,}')

# PDE loss
physics = config.get('physics', {})
pde_loss_fn = APEBenchBurgers3DPDELoss(
    nx=physics.get('nx', 32),
    Lx=physics.get('Lx', 1.0),
    dt=physics.get('dt', 1.0),
    alpha_2=physics.get('alpha_2'),
    beta_1=physics.get('beta_1'),
    eq_scales=physics.get('eq_scales'),
).cuda()

# Test 3 steps
optimizer = torch.optim.AdamW(model.get_trainable_params(), lr=1e-4)
for i, batch in enumerate(train_loader):
    if i >= 3:
        break
    data = batch['data'].cuda().float()
    input_data = data[:, :8]
    target = data[:, 1:9]
    
    out_norm, mean, std = model(input_data, return_normalized=True)
    output = out_norm * std + mean
    
    # PDE loss
    t0_u = input_data[:, 0:1, ..., 0].float()
    t0_v = input_data[:, 0:1, ..., 1].float()
    t0_w = input_data[:, 0:1, ..., 2].float()
    u = torch.cat([t0_u, output[..., 0].float()], dim=1)
    v = torch.cat([t0_v, output[..., 1].float()], dim=1)
    w = torch.cat([t0_w, output[..., 2].float()], dim=1)
    pde_loss, losses = pde_loss_fn(u, v, w)
    
    loss = pde_loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f'Step {i}: pde_loss={pde_loss.item():.4f}, mem={mem:.1f}GB')

print('Training test passed!')
"
