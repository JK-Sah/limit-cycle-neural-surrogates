#!/bin/bash -l
#SBATCH --job-name=parosc
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16g
#SBATCH --output=logs/parosc_%A_%a.out
#SBATCH --error=logs/parosc_%A_%a.err
# Parametric van der Pol: does delta blow up when omega must be interpolated?
# One seed per array task; each task runs all four model configurations.
# Usage: sbatch --array=0-4 cluster/param_osc_array.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; . .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
mkdir -p results/parosc
python src/param_osc.py --epochs 6000 --periods 100 --data-periods 30 \
    --seed-list "$SLURM_ARRAY_TASK_ID" \
    --free-widths 96 150 224 \
    --out "results/parosc/seed${SLURM_ARRAY_TASK_ID}.json"
