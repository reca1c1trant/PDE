#!/bin/bash
# Auto-submit heat_1d and advection_3d after the first 2 jobs finish
cd /home/msai/song0304/code/PDE

echo "Waiting for burgers_1d (18678) and advection_1d (18679) to finish..."

while true; do
    running=$(squeue -u song0304 -h | wc -l)
    if [ "$running" -eq 0 ]; then
        echo "All done. Submitting remaining jobs at $(date)"
        break
    fi
    sleep 60
done

# Submit heat_1d
sbatch run_heat_1d.sh
echo "Submitted heat_1d at $(date)"

# Wait for it to be accepted, then submit advection_3d
sleep 5
sbatch run_advection_3d.sh 2>/dev/null || true
echo "Submitted advection_3d at $(date)"

echo "Waiting for heat_1d and advection_3d to finish..."
while true; do
    running=$(squeue -u song0304 -h | wc -l)
    if [ "$running" -eq 0 ]; then
        echo "All 4 jobs complete at $(date)"
        break
    fi
    sleep 60
done
