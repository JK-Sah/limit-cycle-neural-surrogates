#!/bin/bash -l
#SBATCH --job-name=t0lc
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4g
#SBATCH --output=logs/t0_%A_%a.out
#SBATCH --error=logs/t0_%A_%a.err

# Tier-0 limit-cycle falsification sweep. One array task per trial config.
# Usage: sbatch --array=0-74 cluster/t0_array.sh all

set -euo pipefail
MODE="${1:-all}"
cd "$SLURM_SUBMIT_DIR"
. .venv/bin/activate

# tiny float64 nets: threads hurt more than they help
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

mkdir -p results/"$MODE"
python src/t0_falsify.py \
    --mode "$MODE" \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --epochs 3000 \
    --periods 200 \
    --out results/"$MODE"/trial_$(printf '%03d' "$SLURM_ARRAY_TASK_ID").json

echo "task $SLURM_ARRAY_TASK_ID done on $(hostname)"
