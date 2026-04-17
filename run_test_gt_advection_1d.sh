#!/bin/bash
#SBATCH --job-name=gt_adv1d
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=00:30:00
#SBATCH --output=logs_gen/test_gt_advection_1d_%j.out
#SBATCH --error=logs_gen/test_gt_advection_1d_%j.err

cd /home/msai/song0304/code/PDE
mkdir -p logs_gen data/gt_scales

echo "=========================================="
echo "GT PDE Verification - 1D Advection"
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

source activate icml

python tools/test_gt_pde_advection_1d.py

echo "Done: $(date)"
