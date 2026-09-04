#!/bin/bash -l
#SBATCH --job-name=flapres
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --time=0-20:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16g
#SBATCH --output=logs/flapres_%A_%a.out
#SBATCH --error=logs/flapres_%A_%a.err
# Is the residual period jitter of the flapping filament numerical or physical?
#
# At L = 16, KB* = 2e-2 the cycle is genuinely settled -- tip amplitude constant
# to five figures over 150k steps -- but its jitter is 1.8e-3 and its split-half
# refit 5.8e-4, against 1.1e-6 and 1.6e-8 for the cylinder. Two explanations:
#
#   numerical: the filament is resolved by 16 nodes and the immersed boundary
#     smears over a lattice unit, so refining L should drive the jitter down
#     like a discretization error.
#   physical: a flapping filament couples a structural oscillator to a wake
#     oscillator, and if the two do not lock the attractor is quasi-periodic.
#     Then the jitter is real, it will NOT fall with resolution, and by the
#     paper's own section 9 a single delta does not describe the system.
#
# Refining L separates them. Steps scale with L because the period does.
# Signals are saved so the spectrum can be checked for a second incommensurate
# peak, which is the independent test of the same question.
#
# Usage: sbatch --array=0-2 cluster/flap_resolution.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; . .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
export PYTHONPATH=src
LS=(16 24 32)
ST=(250000 375000 500000)
L=${LS[$SLURM_ARRAY_TASK_ID]}
S=${ST[$SLURM_ARRAY_TASK_ID]}
mkdir -p results/flapres
python src/flapping.py --L "$L" --Re 200 --u-in 0.08 --KB 2e-2 --rho 1.0 \
    --steps "$S" --n-ib 5 \
    --out "results/flapres/L${L}.json" \
    --save-signal "results/flapres/L${L}_signal.npz"
