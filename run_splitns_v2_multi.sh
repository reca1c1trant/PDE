#!/bin/bash
# Split-NS v2 across nodes 4,5,6
set -e
cd /home/msai/song0304/code/PDE

echo "=== $(date): Launching 3 experiments ==="

# ssh 5 (GPU 0,1,3): AdvDiff Split-NS
ssh 5 "cd /home/msai/song0304/code/PDE && \
  CUDA_VISIBLE_DEVICES=0,1,3 nohup torchrun --nproc_per_node=3 --master_port=29501 \
  finetune/train_muon_lora.py --config configs/finetune_advdiff_2d_splitns_v2.yaml \
  > logs/advdiff_splitns_v2.log 2>&1 &" && echo "AdvDiff launched on ssh 5"

# ssh 6 (GPU 1,2,3): Wave Split-NS
ssh 6 "cd /home/msai/song0304/code/PDE && \
  CUDA_VISIBLE_DEVICES=1,2,3 nohup torchrun --nproc_per_node=3 --master_port=29502 \
  finetune/train_muon_lora.py --config configs/finetune_wave_2d_splitns_v2.yaml \
  > logs/wave_splitns_v2.log 2>&1 &" && echo "Wave launched on ssh 6"

# ssh 4 (GPU 0,1,2,3): Burgers vanilla first, then Split-NS
ssh 4 "cd /home/msai/song0304/code/PDE && \
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 \
  finetune/train_burgers_2d_ch_lora_v3.py --config configs/finetune_burgers_2d_ch_v3_rescaled_norm.yaml \
  > logs/burgers_rescaled_norm.log 2>&1 && \
  echo 'Burgers vanilla done, starting Split-NS...' && \
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 \
  finetune/train_muon_lora.py --config configs/finetune_burgers_2d_ch_splitns_v2.yaml \
  > logs/burgers_splitns_v2.log 2>&1" &
echo "Burgers launched on ssh 4 (vanilla then Split-NS)"

echo "=== $(date): All launched. Burgers is sequential (vanilla then Split-NS). ==="
wait
echo "=== $(date): All complete ==="
