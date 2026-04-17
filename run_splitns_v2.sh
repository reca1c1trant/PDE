#!/bin/bash
# Split-NS v2: fair comparison with rescaled_norm baselines
# 3 datasets, each on 4 GPUs, parallel across nodes
#
# GPU allocation:
#   ssh 2 GPU 0,1,2,3 — Burgers (vanilla baseline first, then Split-NS)
#   ssh 2 GPU 4,5,6,7 — AdvDiff Split-NS
#   ssh 3 GPU 0,1,2,3 — Wave Split-NS

set -e

echo "=== $(date): Starting Split-NS v2 experiments ==="

# --- AdvDiff Split-NS (ssh 2, GPU 4,5,6,7) ---
echo "$(date): AdvDiff Split-NS starting..."
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29501 \
  finetune/train_muon_lora.py --config configs/finetune_advdiff_2d_splitns_v2.yaml \
  2>&1 | tee logs/advdiff_splitns_v2.log &
PID_ADVDIFF=$!

# --- Wave Split-NS (ssh 3, GPU 0,1,2,3) ---
echo "$(date): Wave Split-NS starting..."
ssh 3 "cd /home/msai/song0304/code/PDE && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29502 \
  finetune/train_muon_lora.py --config configs/finetune_wave_2d_splitns_v2.yaml" \
  2>&1 | tee logs/wave_splitns_v2.log &
PID_WAVE=$!

# --- Burgers: first train vanilla rescaled_norm to get checkpoint ---
echo "$(date): Burgers vanilla rescaled_norm starting..."
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 \
  finetune/train_burgers_2d_ch_lora_v3.py --config configs/finetune_burgers_2d_ch_v3_rescaled_norm.yaml \
  2>&1 | tee logs/burgers_rescaled_norm.log

# After vanilla finishes, run Burgers Split-NS
echo "$(date): Burgers Split-NS starting..."
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 \
  finetune/train_muon_lora.py --config configs/finetune_burgers_2d_ch_splitns_v2.yaml \
  2>&1 | tee logs/burgers_splitns_v2.log

# Wait for parallel jobs
echo "$(date): Waiting for AdvDiff and Wave..."
wait $PID_ADVDIFF
wait $PID_WAVE

echo "=== $(date): All training complete ==="
