#!/bin/bash
#SBATCH --job-name=gen_adv3d
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=02:00:00
#SBATCH --output=logs_gen/gen_advection_3d_%j.out
#SBATCH --error=logs_gen/gen_advection_3d_%j.err

cd /home/msai/song0304/code/PDE
mkdir -p logs_gen

echo "=========================================="
echo "Generate 3D Advection Dataset"
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

source activate icml

echo "Config: N_GRID=64, N_T=21, dt=0.05, N_SAMPLES=50, |v| in [0.5, 2.0]"
python tools/generate_advection_3d.py

echo "Done: $(date)"
