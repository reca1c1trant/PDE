#!/bin/bash
#SBATCH --job-name=gt_3d
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=00:10:00
#SBATCH --output=logs_1d3d/gt_3d_%j.out
#SBATCH --error=logs_1d3d/gt_3d_%j.err

cd /home/msai/song0304/code/PDE
python tools/test_gt_pde_burgers_3d_apebench.py
