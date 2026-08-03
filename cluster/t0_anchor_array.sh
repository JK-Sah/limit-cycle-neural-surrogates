#!/bin/bash -l
#SBATCH --job-name=t0anch
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --time=0-03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4g
#SBATCH --output=logs/anch_%A_%a.out
#SBATCH --error=logs/anch_%A_%a.err

# Does delta for an anchored model fall to the precision with which the
# frequency can be MEASURED from data, rather than to whatever the fit leaves?
# Grid: supervision weight x observation noise x seed.
# Usage: sbatch --array=0-17 cluster/t0_anchor_array.sh

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
. .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

mkdir -p results/anchor
python src/t0_anchor_sweep.py \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --epochs 3000 \
    --periods 200 \
    --out results/anchor/trial_$(printf '%03d' "$SLURM_ARRAY_TASK_ID").json

echo "task $SLURM_ARRAY_TASK_ID done on $(hostname)"
