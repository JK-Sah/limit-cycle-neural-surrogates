#!/bin/bash -l
#SBATCH --job-name=downstr
#SBATCH --account=flowlab
#SBATCH --partition=debug
#SBATCH --time=0-03:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=24g
#SBATCH --output=logs/downstr_%A_%a.out
#SBATCH --error=logs/downstr_%A_%a.err
# What does the period error cost? Phase-invariant statistics should be immune
# to delta; phase-resolved ones should follow 2|sin(pi delta N)| and first
# decorrelate at N = 1/(4 delta). One seed per task.
# Usage: sbatch --array=0-2 cluster/downstream_job.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; . .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
mkdir -p results/downstream
python src/downstream.py --data data/resweep --Re 100 --modes 8 \
    --epochs 4000 --seed-list "$SLURM_ARRAY_TASK_ID" \
    --horizons 10 30 100 300 1000 3000 6000 10000 20000 30000 \
    --out "results/downstream/seed${SLURM_ARRAY_TASK_ID}.json"
