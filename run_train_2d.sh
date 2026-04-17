#!/bin/bash
# Train 4 datasets x 2 variants = 8 jobs sequentially on 4 GPUs
# Datasets: Taylor-Green, Wave, AdvDiff, Burgers2D (skip KP-II per user request)

set -e
cd /home/msai/song0304/code/PDE

LOGDIR="logs_2d_train"
mkdir -p "$LOGDIR"

echo "=========================================="
echo "Starting 8 training jobs at $(date)"
echo "=========================================="

# 1. Taylor-Green (norm)
echo "[1/8] Taylor-Green (norm) - $(date)"
torchrun --nproc_per_node=4 finetune/train_taylor_green_2d_lora_v3.py \
    --config configs/finetune_taylor_green_2d_v3.yaml \
    2>&1 | tee "$LOGDIR/taylor_green_norm.log"
echo "[1/8] Done at $(date)"

# 2. Taylor-Green (no norm)
echo "[2/8] Taylor-Green (no norm) - $(date)"
torchrun --nproc_per_node=4 finetune/train_taylor_green_2d_lora_v3.py \
    --config configs/finetune_taylor_green_2d_v3_no_norm.yaml \
    2>&1 | tee "$LOGDIR/taylor_green_no_norm.log"
echo "[2/8] Done at $(date)"

# 3. Wave (norm)
echo "[3/8] Wave (norm) - $(date)"
torchrun --nproc_per_node=4 finetune/train_wave_2d_lora_v3.py \
    --config configs/finetune_wave_2d_v3.yaml \
    2>&1 | tee "$LOGDIR/wave_norm.log"
echo "[3/8] Done at $(date)"

# 4. Wave (no norm)
echo "[4/8] Wave (no norm) - $(date)"
torchrun --nproc_per_node=4 finetune/train_wave_2d_lora_v3.py \
    --config configs/finetune_wave_2d_v3_no_norm.yaml \
    2>&1 | tee "$LOGDIR/wave_no_norm.log"
echo "[4/8] Done at $(date)"

# 5. AdvDiff (norm)
echo "[5/8] AdvDiff (norm) - $(date)"
torchrun --nproc_per_node=4 finetune/train_advdiff_2d_lora_v3.py \
    --config configs/finetune_advdiff_2d_v3.yaml \
    2>&1 | tee "$LOGDIR/advdiff_norm.log"
echo "[5/8] Done at $(date)"

# 6. AdvDiff (no norm)
echo "[6/8] AdvDiff (no norm) - $(date)"
torchrun --nproc_per_node=4 finetune/train_advdiff_2d_lora_v3.py \
    --config configs/finetune_advdiff_2d_v3_no_norm.yaml \
    2>&1 | tee "$LOGDIR/advdiff_no_norm.log"
echo "[6/8] Done at $(date)"

# 7. Burgers2D (norm)
echo "[7/8] Burgers2D (norm) - $(date)"
torchrun --nproc_per_node=4 finetune/train_burgers_2d_ch_lora_v3.py \
    --config configs/finetune_burgers_2d_ch_v3.yaml \
    2>&1 | tee "$LOGDIR/burgers2d_norm.log"
echo "[7/8] Done at $(date)"

# 8. Burgers2D (no norm)
echo "[8/8] Burgers2D (no norm) - $(date)"
torchrun --nproc_per_node=4 finetune/train_burgers_2d_ch_lora_v3.py \
    --config configs/finetune_burgers_2d_ch_v3_no_norm.yaml \
    2>&1 | tee "$LOGDIR/burgers2d_no_norm.log"
echo "[8/8] Done at $(date)"

echo "=========================================="
echo "All 8 jobs completed at $(date)"
echo "=========================================="
