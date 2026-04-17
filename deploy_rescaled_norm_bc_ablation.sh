#!/bin/bash
# Deploy rescaled_norm lambda_bc ablation: bc1000 first, wait 6h, then bc100
set -e
cd /home/msai/song0304/code/PDE

echo "=========================================="
echo "rescaled_norm lambda_bc ablation"
echo "Start: $(date)"
echo "=========================================="

# ---- Round 1: lambda_bc=1000 ----
echo "[Round 1] lambda_bc=1000 - $(date)"

ssh 3 "mkdir -p logs_rescaled_norm_bc1000 && nohup bash -c 'CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=29700 finetune/train_taylor_green_2d_lora_v3.py --config configs/finetune_taylor_green_2d_v3_rescaled_norm_bc1000.yaml' > logs_rescaled_norm_bc1000/taylor_green_2d.log 2>&1 &" &

ssh 4 "mkdir -p logs_rescaled_norm_bc1000 && nohup bash -c 'CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=29700 finetune/train_wave_2d_lora_v3.py --config configs/finetune_wave_2d_v3_rescaled_norm_bc1000.yaml' > logs_rescaled_norm_bc1000/wave_2d.log 2>&1 &" &

ssh 5 "mkdir -p logs_rescaled_norm_bc1000 && nohup bash -c 'CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=29700 finetune/train_advdiff_2d_lora_v3.py --config configs/finetune_advdiff_2d_v3_rescaled_norm_bc1000.yaml' > logs_rescaled_norm_bc1000/advdiff_2d.log 2>&1 &" &

ssh 6 "mkdir -p logs_rescaled_norm_bc1000 && nohup bash -c 'CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=29700 finetune/train_burgers_2d_ch_lora_v3.py --config configs/finetune_burgers_2d_ch_v3_rescaled_norm_bc1000.yaml' > logs_rescaled_norm_bc1000/burgers_2d.log 2>&1 &" &

wait
echo "[Round 1] All 4 bc1000 submitted at $(date)"

# ---- Wait 6 hours ----
echo "Sleeping 6 hours..."
sleep 6h
echo "Woke up at $(date)"

# ---- Round 2: lambda_bc=100 ----
echo "[Round 2] lambda_bc=100 - $(date)"

ssh 3 "mkdir -p logs_rescaled_norm_bc100 && nohup bash -c 'CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=29700 finetune/train_taylor_green_2d_lora_v3.py --config configs/finetune_taylor_green_2d_v3_rescaled_norm_bc100.yaml' > logs_rescaled_norm_bc100/taylor_green_2d.log 2>&1 &" &

ssh 4 "mkdir -p logs_rescaled_norm_bc100 && nohup bash -c 'CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=29700 finetune/train_wave_2d_lora_v3.py --config configs/finetune_wave_2d_v3_rescaled_norm_bc100.yaml' > logs_rescaled_norm_bc100/wave_2d.log 2>&1 &" &

ssh 5 "mkdir -p logs_rescaled_norm_bc100 && nohup bash -c 'CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=29700 finetune/train_advdiff_2d_lora_v3.py --config configs/finetune_advdiff_2d_v3_rescaled_norm_bc100.yaml' > logs_rescaled_norm_bc100/advdiff_2d.log 2>&1 &" &

ssh 6 "mkdir -p logs_rescaled_norm_bc100 && nohup bash -c 'CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=29700 finetune/train_burgers_2d_ch_lora_v3.py --config configs/finetune_burgers_2d_ch_v3_rescaled_norm_bc100.yaml' > logs_rescaled_norm_bc100/burgers_2d.log 2>&1 &" &

wait
echo "[Round 2] All 4 bc100 submitted at $(date)"
echo "=========================================="
echo "All done at $(date)"
echo "=========================================="
