#!/bin/bash
set -e
cd /home/msai/song0304/code/PDE

echo "=========================================="
echo "Full-FT Physics-Only Ablation"
echo "Start: $(date)"
echo "=========================================="

# Round 1: TG on node 3, Wave on node 4
echo "[Round 1] TG + Wave - $(date)"

ssh 3 "mkdir -p logs_fullft_physonly && nohup bash -c 'CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=29700 finetune/train_fullft_physonly.py --config configs/finetune_taylor_green_2d_v3_fullft_physonly.yaml' > logs_fullft_physonly/taylor_green_2d.log 2>&1 &" &

ssh 4 "mkdir -p logs_fullft_physonly && nohup bash -c 'CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=29700 finetune/train_fullft_physonly.py --config configs/finetune_wave_2d_v3_fullft_physonly.yaml' > logs_fullft_physonly/wave_2d.log 2>&1 &" &

wait
echo "[Round 1] Submitted at $(date)"

# Wait 6 hours for round 1 to finish
sleep 6h

# Round 2: AdvDiff on node 3, Burgers on node 4
echo "[Round 2] AdvDiff + Burgers - $(date)"

ssh 3 "mkdir -p logs_fullft_physonly && nohup bash -c 'CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=29700 finetune/train_fullft_physonly.py --config configs/finetune_advdiff_2d_v3_fullft_physonly.yaml' > logs_fullft_physonly/advdiff_2d.log 2>&1 &" &

ssh 4 "mkdir -p logs_fullft_physonly && nohup bash -c 'CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=29700 finetune/train_fullft_physonly.py --config configs/finetune_burgers_2d_ch_v3_fullft_physonly.yaml' > logs_fullft_physonly/burgers_2d.log 2>&1 &" &

wait
echo "[Round 2] Submitted at $(date)"
echo "=========================================="
echo "All done at $(date)"
echo "=========================================="
