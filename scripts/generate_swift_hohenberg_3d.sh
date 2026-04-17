#!/bin/bash
#SBATCH --job-name=gen_sh3d
#SBATCH --output=logs/gen_sh3d_%j.out
#SBATCH --error=logs/gen_sh3d_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=6:00:00

cd /home/msai/song0304/code/PDE

# Step 1: Generate with apebench env
source activate apebench
python tools/generate_swift_hohenberg_3d_apebench.py

# Step 2: Convert with icml env
source activate icml
python tools/convert_swift_hohenberg_3d_apebench.py
