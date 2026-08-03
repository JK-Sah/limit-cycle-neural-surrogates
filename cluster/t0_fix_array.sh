#!/bin/bash -l
#SBATCH --job-name=t0fix
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --time=0-03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4g
#SBATCH --output=logs/fix_%A_%a.out
#SBATCH --error=logs/fix_%A_%a.err

# Constructive test: anchored limit-cycle parametrization vs free neural ODE.
# Usage: sbatch --array=0-19 cluster/t0_fix_array.sh

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
. .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

mkdir -p results/fix
python src/t0_fix.py \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --epochs 3000 \
    --periods 200 \
    --seeds 5 \
    --out results/fix/trial_$(printf '%03d' "$SLURM_ARRAY_TASK_ID").json

echo "task $SLURM_ARRAY_TASK_ID done on $(hostname)"
