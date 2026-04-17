#!/bin/bash
set -e
cd /home/msai/song0304/code/PDE

echo "=========================================="
echo "The Well: rescaled_norm + supervised LoRA"
echo "Start: $(date)"
echo "=========================================="

# ---- Round 1: rescaled_norm ----
echo "[Round 1] rescaled_norm - $(date)"

ssh 3 "mkdir -p logs_rescaled_norm_thewell && nohup bash -c 'CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=29700 finetune/train_gray_scott_lora_v3.py --config configs/finetune_gray_scott_v3_rescaled_norm.yaml' > logs_rescaled_norm_thewell/gray_scott.log 2>&1 &" &

ssh 4 "mkdir -p logs_rescaled_norm_thewell && nohup bash -c 'CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=29700 finetune/train_rayleigh_benard_lora_v3.py --config configs/finetune_rayleigh_benard_v3_rescaled_norm.yaml' > logs_rescaled_norm_thewell/rayleigh_benard.log 2>&1 &" &

ssh 5 "mkdir -p logs_rescaled_norm_thewell && nohup bash -c 'CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=29700 finetune/train_shear_flow_lora_v3.py --config configs/finetune_shear_flow_v3_rescaled_norm.yaml' > logs_rescaled_norm_thewell/shear_flow.log 2>&1 &" &

ssh 6 "mkdir -p logs_rescaled_norm_thewell && nohup bash -c 'CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=29700 finetune/train_active_matter_lora_v3.py --config configs/finetune_active_matter_v3_rescaled_norm.yaml' > logs_rescaled_norm_thewell/active_matter.log 2>&1 &" &

wait
echo "[Round 1] All 4 rescaled_norm submitted at $(date)"

# ---- Wait 6 hours ----
echo "Sleeping 6 hours..."
sleep 6h
echo "Woke up at $(date)"

# ---- Round 2: supervised LoRA ----
echo "[Round 2] supervised LoRA - $(date)"

ssh 3 "mkdir -p logs_supervised_thewell && nohup bash -c 'CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=29700 finetune/train_supervised_lora.py --config configs/finetune_gray_scott_v3_supervised.yaml' > logs_supervised_thewell/gray_scott.log 2>&1 &" &

ssh 4 "mkdir -p logs_supervised_thewell && nohup bash -c 'CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=29700 finetune/train_supervised_lora.py --config configs/finetune_rayleigh_benard_v3_supervised.yaml' > logs_supervised_thewell/rayleigh_benard.log 2>&1 &" &

ssh 5 "mkdir -p logs_supervised_thewell && nohup bash -c 'CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=29700 finetune/train_supervised_lora.py --config configs/finetune_shear_flow_v3_supervised.yaml' > logs_supervised_thewell/shear_flow.log 2>&1 &" &

ssh 6 "mkdir -p logs_supervised_thewell && nohup bash -c 'CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=29700 finetune/train_supervised_lora.py --config configs/finetune_active_matter_v3_supervised.yaml' > logs_supervised_thewell/active_matter.log 2>&1 &" &

wait
echo "[Round 2] All 4 supervised submitted at $(date)"
echo "=========================================="
echo "All done at $(date)"
echo "=========================================="
