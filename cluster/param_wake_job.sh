#!/bin/bash -l
#SBATCH --job-name=parwake
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64g
#SBATCH --output=logs/parwake_%A_%a.out
#SBATCH --error=logs/parwake_%A_%a.err
# One latent operator across Reynolds number. Free baselines at width
# 128/200/288 bracket the anchored model's parameter count (~112k).
# Usage: sbatch --array=0-4 cluster/param_wake_job.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; . .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
mkdir -p results/parwake
python src/param_wake.py --data data/resweep --modes 16 --epochs 6000 \
    --periods 100 --seed-list "$SLURM_ARRAY_TASK_ID" \
    --free-widths 128 200 288 \
    --out "results/parwake/seed${SLURM_ARRAY_TASK_ID}.json"
