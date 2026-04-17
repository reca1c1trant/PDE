#!/bin/bash
# Run Split-NS + Vanilla baseline for Burgers, AdvDiff, Wave (sequential)
# GPU 3,5,6,7, 4 cards each run

export CUDA_VISIBLE_DEVICES=3,5,6,7
PORT=29503

echo "=== Starting Split-NS experiments ==="
echo "$(date): Burgers Split-NS"
torchrun --nproc_per_node=4 --master_port=$PORT finetune/train_muon_lora.py --config configs/finetune_burgers_2d_ch_splitns.yaml 2>&1 | tee logs/burgers_splitns.log

echo "$(date): Burgers Vanilla"
torchrun --nproc_per_node=4 --master_port=$PORT finetune/train_vanilla_lora_frozen.py --config configs/finetune_burgers_2d_ch_vanilla_fair.yaml 2>&1 | tee logs/burgers_vanilla_fair.log

echo "$(date): AdvDiff Split-NS"
torchrun --nproc_per_node=4 --master_port=$PORT finetune/train_muon_lora.py --config configs/finetune_advdiff_2d_splitns.yaml 2>&1 | tee logs/advdiff_splitns.log

echo "$(date): AdvDiff Vanilla"
torchrun --nproc_per_node=4 --master_port=$PORT finetune/train_vanilla_lora_frozen.py --config configs/finetune_advdiff_2d_vanilla_fair.yaml 2>&1 | tee logs/advdiff_vanilla_fair.log

echo "$(date): Wave Split-NS"
torchrun --nproc_per_node=4 --master_port=$PORT finetune/train_muon_lora.py --config configs/finetune_wave_2d_splitns.yaml 2>&1 | tee logs/wave_splitns.log

echo "$(date): Wave Vanilla"
torchrun --nproc_per_node=4 --master_port=$PORT finetune/train_vanilla_lora_frozen.py --config configs/finetune_wave_2d_vanilla_fair.yaml 2>&1 | tee logs/wave_vanilla_fair.log

echo "=== All training complete. Running evaluations ==="

# Eval all 6
for ds in burgers_2d_ch advdiff_2d wave_2d; do
  echo "$(date): Eval ${ds} Split-NS"
  CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --master_port=$PORT tools/eval_splitns.py \
    --config configs/finetune_${ds}_splitns.yaml \
    --checkpoint checkpoints_${ds}_splitns/best_lora.pt --scan_all 2>&1 | tee logs/eval_${ds}_splitns.log

  echo "$(date): Eval ${ds} Vanilla"
  CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --master_port=$PORT tools/eval_vanilla_frozen.py \
    --config configs/finetune_${ds}_vanilla_fair.yaml \
    --checkpoint checkpoints_${ds}_vanilla_fair/best_lora.pt --scan_all 2>&1 | tee logs/eval_${ds}_vanilla_fair.log
done

echo "=== ALL DONE ==="
