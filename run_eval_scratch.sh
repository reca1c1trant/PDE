#!/bin/bash
# Evaluate 4 from-scratch checkpoints + 1 wave norm LoRA checkpoint
set -e
cd /home/msai/song0304/code/PDE

LOGDIR="logs_eval_scratch"
mkdir -p "$LOGDIR"

echo "=========================================="
echo "Starting evaluation at $(date)"
echo "=========================================="

# 1. Taylor-Green scratch
echo "[1/5] Taylor-Green (scratch) - $(date)"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 tools/eval_scratch.py \
    --config configs/finetune_taylor_green_2d_v3_scratch.yaml \
    --checkpoint checkpoints_taylor_green_2d_lora_v3_scratch/best_scratch.pt \
    2>&1 | tee "$LOGDIR/taylor_green_scratch.log"
echo "[1/5] Done at $(date)"

# 2. Wave scratch
echo "[2/5] Wave (scratch) - $(date)"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 tools/eval_scratch.py \
    --config configs/finetune_wave_2d_v3_scratch.yaml \
    --checkpoint checkpoints_wave_2d_lora_v3_scratch/best_scratch.pt \
    2>&1 | tee "$LOGDIR/wave_scratch.log"
echo "[2/5] Done at $(date)"

# 3. AdvDiff scratch
echo "[3/5] AdvDiff (scratch) - $(date)"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 tools/eval_scratch.py \
    --config configs/finetune_advdiff_2d_v3_scratch.yaml \
    --checkpoint checkpoints_advdiff_2d_lora_v3_scratch/best_scratch.pt \
    2>&1 | tee "$LOGDIR/advdiff_scratch.log"
echo "[3/5] Done at $(date)"

# 4. Burgers scratch
echo "[4/5] Burgers (scratch) - $(date)"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 tools/eval_scratch.py \
    --config configs/finetune_burgers_2d_ch_v3_scratch.yaml \
    --checkpoint checkpoints_burgers_2d_ch_lora_v3_scratch/best_scratch.pt \
    2>&1 | tee "$LOGDIR/burgers_scratch.log"
echo "[4/5] Done at $(date)"

# 5. Wave norm (LoRA checkpoint, not scratch)
echo "[5/5] Wave (norm, LoRA) - $(date)"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 tools/visualize_wave_2d_lora.py \
    --config configs/finetune_wave_2d_v3.yaml \
    --checkpoint checkpoints_wave_2d_lora_v3/best_lora.pt \
    --scan_all 2>&1 | tee "$LOGDIR/wave_norm.log"
echo "[5/5] Done at $(date)"

echo "=========================================="
echo "All evaluations completed at $(date)"
echo "=========================================="
