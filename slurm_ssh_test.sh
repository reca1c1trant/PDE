#!/bin/bash
#SBATCH --job-name=ssh_test
#SBATCH --partition=MGPU-TC2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm_ssh_test_%j.log

echo "I am on: $(hostname) ($(hostname -i))"
echo "Testing SSH to other compute nodes..."

for node in TC2N01 TC2N02 TC2N03 TC2N04 TC2N05 TC2N06; do
  result=$(ssh -o ConnectTimeout=3 -o BatchMode=yes $node "echo ok" 2>&1 | tail -1)
  echo "  $node: $result"
done

echo "=== Done ==="
