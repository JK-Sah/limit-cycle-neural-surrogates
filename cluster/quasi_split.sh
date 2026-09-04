#!/bin/bash -l
#SBATCH --job-name=qsplit
#SBATCH --account=flowlab
#SBATCH --partition=debug
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16g
#SBATCH --output=logs/qsplit_%A.out
#SBATCH --error=logs/qsplit_%A.err
# Save section 9's split-half numbers, and test them against record length.
# The original check printed and saved nothing, and used a single window.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; . .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
export PYTHONPATH=src
mkdir -p results/quasi_split
python src/quasi_split.py --out results/quasi_split/sweep.json
