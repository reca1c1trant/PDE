#!/bin/bash
set -e
cd /home/msai/song0304/code/PDE

echo "=========================================="
echo "The Well: no_norm (after 1h) + full FT (after 6h)"
echo "Start: $(date)"
echo "=========================================="

# ---- Wait 1 hour before starting no_norm ----
echo "Sleeping 1 hour before no_norm..."
sleep 1h
echo "Woke up at $(date)"

# ---- Round 1: no_norm ----
echo "[Round 1] no_norm - $(date)"

ssh 3 "mkdir -p logs_no_norm_thewell && nohup bash -c 'CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=29700 finetune/train_gray_scott_lora_v3.py --config configs/finetune_gray_scott_v3_no_norm.yaml' > logs_no_norm_thewell/gray_scott.log 2>&1 &" &

ssh 4 "mkdir -p logs_no_norm_thewell && nohup bash -c 'CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=29700 finetune/train_rayleigh_benard_lora_v3.py --config configs/finetune_rayleigh_benard_v3_no_norm.yaml' > logs_no_norm_thewell/rayleigh_benard.log 2>&1 &" &

ssh 5 "mkdir -p logs_no_norm_thewell && nohup bash -c 'CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=29700 finetune/train_shear_flow_lora_v3.py --config configs/finetune_shear_flow_v3_no_norm.yaml' > logs_no_norm_thewell/shear_flow.log 2>&1 &" &

ssh 6 "mkdir -p logs_no_norm_thewell && nohup bash -c 'CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=29700 finetune/train_active_matter_lora_v3.py --config configs/finetune_active_matter_v3_no_norm.yaml' > logs_no_norm_thewell/active_matter.log 2>&1 &" &

wait
echo "[Round 1] All 4 no_norm submitted at $(date)"

# ---- Wait 6 hours ----
echo "Sleeping 6 hours before full FT..."
sleep 6h
echo "Woke up at $(date)"

# ---- Round 2: full finetune ----
echo "[Round 2] full FT - $(date)"

ssh 3 "mkdir -p logs_fullft_thewell && nohup bash -c 'CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=29700 finetune/train_fullft_physonly.py --config configs/finetune_gray_scott_v3_fullft_physonly.yaml' > logs_fullft_thewell/gray_scott.log 2>&1 &" &

ssh 4 "mkdir -p logs_fullft_thewell && nohup bash -c 'CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=29700 finetune/train_fullft_physonly.py --config configs/finetune_rayleigh_benard_v3_fullft_physonly.yaml' > logs_fullft_thewell/rayleigh_benard.log 2>&1 &" &

ssh 5 "mkdir -p logs_fullft_thewell && nohup bash -c 'CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=29700 finetune/train_fullft_physonly.py --config configs/finetune_shear_flow_v3_fullft_physonly.yaml' > logs_fullft_thewell/shear_flow.log 2>&1 &" &

ssh 6 "mkdir -p logs_fullft_thewell && nohup bash -c 'CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=29700 finetune/train_fullft_physonly.py --config configs/finetune_active_matter_v3_fullft_physonly.yaml' > logs_fullft_thewell/active_matter.log 2>&1 &" &

wait
echo "[Round 2] All 4 full FT submitted at $(date)"
echo "=========================================="
echo "All done at $(date)"
echo "=========================================="
