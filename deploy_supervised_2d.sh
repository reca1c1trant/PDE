#!/bin/bash
# Deploy 4 supervised LoRA training jobs to nodes 3,4,5,6
# Each job uses GPU 1,2,3 (leaving GPU 0 free)
# Usage: nohup bash deploy_supervised_2d.sh [--delay 5h] > logs_supervised/deploy.log 2>&1 &

set -e

DELAY="${1:-5h}"  # default 5 hours delay
WORKDIR="/home/msai/song0304/code/PDE"
LOGDIR="logs_supervised"
mkdir -p "$LOGDIR"

echo "=========================================="
echo "Supervised LoRA Deploy Script"
echo "Delay: $DELAY"
echo "Start time: $(date)"
echo "=========================================="

if [ "$DELAY" != "0" ]; then
    echo "Sleeping for $DELAY before launching..."
    sleep "$DELAY"
fi

echo "Waking up at $(date). Deploying to 4 nodes..."

# Node 3: Taylor-Green 2D
echo "[1/4] Deploying Taylor-Green 2D to node 3 at $(date)"
ssh 3 "cd $WORKDIR && mkdir -p $LOGDIR && nohup bash -c 'CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 finetune/train_supervised_lora.py --config configs/finetune_taylor_green_2d_supervised.yaml' > $LOGDIR/taylor_green_2d.log 2>&1 &" &

# Node 4: Wave 2D
echo "[2/4] Deploying Wave 2D to node 4 at $(date)"
ssh 4 "cd $WORKDIR && mkdir -p $LOGDIR && nohup bash -c 'CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 finetune/train_supervised_lora.py --config configs/finetune_wave_2d_supervised.yaml' > $LOGDIR/wave_2d.log 2>&1 &" &

# Node 5: AdvDiff 2D
echo "[3/4] Deploying AdvDiff 2D to node 5 at $(date)"
ssh 5 "cd $WORKDIR && mkdir -p $LOGDIR && nohup bash -c 'CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 finetune/train_supervised_lora.py --config configs/finetune_advdiff_2d_supervised.yaml' > $LOGDIR/advdiff_2d.log 2>&1 &" &

# Node 6: Burgers 2D
echo "[4/4] Deploying Burgers 2D to node 6 at $(date)"
ssh 6 "cd $WORKDIR && mkdir -p $LOGDIR && nohup bash -c 'CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 finetune/train_supervised_lora.py --config configs/finetune_burgers_2d_ch_supervised.yaml' > $LOGDIR/burgers_2d.log 2>&1 &" &

# Wait for all SSH commands to finish
wait

echo "=========================================="
echo "All 4 jobs deployed at $(date)"
echo "Monitor: ssh N 'tail -f $WORKDIR/$LOGDIR/*.log'"
echo "=========================================="
