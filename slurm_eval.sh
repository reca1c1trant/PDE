#!/bin/bash
#SBATCH --job-name=eval_splitns
#SBATCH --partition=MGPU-TC2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm_eval_%j.log

echo "=== Job info ==="
echo "Node: $(hostname)"
echo "IP: $(hostname -i)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "=== SSH test ==="
echo "You can ssh to this node: ssh $(hostname)"
echo "=================="

cd /home/msai/song0304/code/PDE

# Eval all 3 vanilla v2
for ds in burgers_2d_ch advdiff_2d wave_2d; do
  echo "=== ${ds} Vanilla ==="
  python tools/eval_vanilla_frozen.py \
    --config configs/finetune_${ds}_vanilla_v2.yaml \
    --checkpoint checkpoints_${ds}_vanilla_v2/best_lora.pt --scan_all 2>&1 | grep -E 'Loaded.*keys|Results|VRMSE|nRMSE|RMSE|PDE'
  echo ""
done

echo "=== All eval done ==="

# Keep alive for SSH access (sleep 3.5 hours)
echo "Node available for SSH for 3.5 hours..."
sleep 12600
