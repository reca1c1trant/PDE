#!/bin/bash
#SBATCH --job-name=gen_adv1d
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=00:30:00
#SBATCH --output=logs_gen/gen_advection_1d_%j.out
#SBATCH --error=logs_gen/gen_advection_1d_%j.err

cd /home/msai/song0304/code/PDE
mkdir -p logs_gen

echo "=========================================="
echo "Generate 1D Advection Dataset"
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

source activate icml

echo "Config: N_GRID=256, N_T=101, N_SAMPLES=1000, a~U(0.5,3.0)"
python tools/generate_advection_1d.py

echo "Done: $(date)"
