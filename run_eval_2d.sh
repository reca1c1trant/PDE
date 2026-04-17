#!/bin/bash
# Evaluate 7 checkpoints (skip wave_norm which is being retrained on node 6)
set -e
cd /home/msai/song0304/code/PDE

LOGDIR="logs_2d_eval"
mkdir -p "$LOGDIR"

echo "=========================================="
echo "Starting 7 evaluation jobs at $(date)"
echo "=========================================="

# 1. Taylor-Green (norm)
echo "[1/7] Taylor-Green (norm) - $(date)"
torchrun --nproc_per_node=4 tools/visualize_taylor_green_2d_lora.py \
    --config configs/finetune_taylor_green_2d_v3.yaml \
    --checkpoint checkpoints_taylor_green_2d_lora_v3/best_lora.pt \
    --scan_all 2>&1 | tee "$LOGDIR/taylor_green_norm.log"
echo "[1/7] Done at $(date)"

# 2. Taylor-Green (no norm)
echo "[2/7] Taylor-Green (no norm) - $(date)"
torchrun --nproc_per_node=4 tools/visualize_taylor_green_2d_lora.py \
    --config configs/finetune_taylor_green_2d_v3_no_norm.yaml \
    --checkpoint checkpoints_taylor_green_2d_lora_v3_no_norm/best_lora.pt \
    --scan_all 2>&1 | tee "$LOGDIR/taylor_green_no_norm.log"
echo "[2/7] Done at $(date)"

# 3. Wave (no norm) — skip wave norm (being retrained)
echo "[3/7] Wave (no norm) - $(date)"
torchrun --nproc_per_node=4 tools/visualize_wave_2d_lora.py \
    --config configs/finetune_wave_2d_v3_no_norm.yaml \
    --checkpoint checkpoints_wave_2d_lora_v3_no_norm/best_lora.pt \
    --scan_all 2>&1 | tee "$LOGDIR/wave_no_norm.log"
echo "[3/7] Done at $(date)"

# 4. AdvDiff (norm)
echo "[4/7] AdvDiff (norm) - $(date)"
torchrun --nproc_per_node=4 tools/visualize_advdiff_2d_lora.py \
    --config configs/finetune_advdiff_2d_v3.yaml \
    --checkpoint checkpoints_advdiff_2d_lora_v3/best_lora.pt \
    --scan_all 2>&1 | tee "$LOGDIR/advdiff_norm.log"
echo "[4/7] Done at $(date)"

# 5. AdvDiff (no norm)
echo "[5/7] AdvDiff (no norm) - $(date)"
torchrun --nproc_per_node=4 tools/visualize_advdiff_2d_lora.py \
    --config configs/finetune_advdiff_2d_v3_no_norm.yaml \
    --checkpoint checkpoints_advdiff_2d_lora_v3_no_norm/best_lora.pt \
    --scan_all 2>&1 | tee "$LOGDIR/advdiff_no_norm.log"
echo "[5/7] Done at $(date)"

# 6. Burgers2D (norm)
echo "[6/7] Burgers2D (norm) - $(date)"
torchrun --nproc_per_node=4 tools/visualize_burgers_2d_ch_lora.py \
    --config configs/finetune_burgers_2d_ch_v3.yaml \
    --checkpoint checkpoints_burgers_2d_ch_lora_v3/best_lora.pt \
    --scan_all 2>&1 | tee "$LOGDIR/burgers2d_norm.log"
echo "[6/7] Done at $(date)"

# 7. Burgers2D (no norm)
echo "[7/7] Burgers2D (no norm) - $(date)"
torchrun --nproc_per_node=4 tools/visualize_burgers_2d_ch_lora.py \
    --config configs/finetune_burgers_2d_ch_v3_no_norm.yaml \
    --checkpoint checkpoints_burgers_2d_ch_lora_v3_no_norm/best_lora.pt \
    --scan_all 2>&1 | tee "$LOGDIR/burgers2d_no_norm.log"
echo "[7/7] Done at $(date)"

echo "=========================================="
echo "All 7 evaluation jobs completed at $(date)"
echo "=========================================="
