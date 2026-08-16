#!/bin/bash -l
#SBATCH --job-name=quasi
#SBATCH --account=flowlab
#SBATCH --partition=debug
#SBATCH --time=0-03:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16g
#SBATCH --output=logs/quasi_%A_%a.out
#SBATCH --error=logs/quasi_%A_%a.err
# Where does the single-period argument stop holding?
# Coupling sweep across the locking transition: quasi-periodic at eps <= 0.3
# (phase drift 0.77 to 0.49), phase-locked at eps >= 0.4 (drift exactly 0).
# One coupling value per task, 3 seeds each.
# Usage: sbatch --array=0-5 cluster/quasi_job.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; . .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
EPS=(0.0 0.1 0.2 0.3 0.5 0.8)
E=${EPS[$SLURM_ARRAY_TASK_ID]}
mkdir -p results/quasi
python src/quasiperiodic.py --eps-list "$E" --epochs 4000 \
    --seed-list 0 1 2 --eval-time 20000 \
    --out "results/quasi/eps${E}.json"
echo "eps=$E done"
