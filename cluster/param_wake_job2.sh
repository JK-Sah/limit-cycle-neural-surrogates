#!/bin/bash -l
#SBATCH --job-name=parwake2
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --time=0-14:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64g
#SBATCH --output=logs/parwake2_%A_%a.out
#SBATCH --error=logs/parwake2_%A_%a.err
# Re-run with FROZEN, constrained frequency interpolants.
# The previous run learned omega with a free MLP supervised at the training Re
# and it wiggled between them: interpolation delta median 1.96e-3, worst
# 1.42e-1. Those numbers are kept as the failed ablation; not re-run here
# because that configuration also cost 10-35x the runtime.
# Usage: sbatch --array=0-4 cluster/param_wake_job2.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; . .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
mkdir -p results/parwake2
python src/param_wake.py --data data/resweep --modes 16 --epochs 6000 \
    --periods 100 --seed-list "$SLURM_ARRAY_TASK_ID" \
    --free-widths 128 200 288 --omega-modes williamson poly4 \
    --out "results/parwake2/seed${SLURM_ARRAY_TASK_ID}.json"
