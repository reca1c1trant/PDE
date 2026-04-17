#!/bin/bash
# Run 4 supervised LoRA training jobs sequentially on one node
set -e
cd /home/msai/song0304/code/PDE

LOGDIR="logs_supervised"
mkdir -p "$LOGDIR"

echo "=========================================="
echo "Supervised LoRA Training (Sequential)"
echo "Start: $(date)"
echo "=========================================="

# 1. Taylor-Green 2D
echo "[1/4] Taylor-Green 2D - $(date)"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 finetune/train_supervised_lora.py \
    --config configs/finetune_taylor_green_2d_supervised.yaml \
    2>&1 | tee "$LOGDIR/taylor_green_2d.log"
echo "[1/4] Done at $(date)"

# 2. Wave 2D
echo "[2/4] Wave 2D - $(date)"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 finetune/train_supervised_lora.py \
    --config configs/finetune_wave_2d_supervised.yaml \
    2>&1 | tee "$LOGDIR/wave_2d.log"
echo "[2/4] Done at $(date)"

# 3. AdvDiff 2D
echo "[3/4] AdvDiff 2D - $(date)"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 finetune/train_supervised_lora.py \
    --config configs/finetune_advdiff_2d_supervised.yaml \
    2>&1 | tee "$LOGDIR/advdiff_2d.log"
echo "[3/4] Done at $(date)"

# 4. Burgers 2D
echo "[4/4] Burgers 2D - $(date)"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 finetune/train_supervised_lora.py \
    --config configs/finetune_burgers_2d_ch_supervised.yaml \
    2>&1 | tee "$LOGDIR/burgers_2d.log"
echo "[4/4] Done at $(date)"

echo "=========================================="
echo "All done at $(date)"
echo "=========================================="
