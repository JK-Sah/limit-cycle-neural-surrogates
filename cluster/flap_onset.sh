#!/bin/bash -l
#SBATCH --job-name=flapons
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --time=0-09:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16g
#SBATCH --output=logs/flapons_%A_%a.out
#SBATCH --error=logs/flapons_%A_%a.err
# Locate the flutter onset in a relaxation time the solver is actually
# validated at.
#
# The first onset scan ran at L=16, Re=200, u=0.08, which puts tau at 0.5192 --
# close enough to the LBM floor of 0.5 that the scheme is at its least accurate.
# Refining to L=24 at the same nondimensional parameters made the flapping go
# away entirely (envelope decayed by 10x over 375k steps), so the onset measured
# at L=16 was at least partly a discretization artifact, not physics.
#
# This runs at u=0.1, Re=100 -> tau = 0.572, the same relaxation time as the
# cylinder work that reproduced Williamson's Strouhal number. Onset is bracketed
# by whether the tip envelope grows or decays, which needs far fewer steps than
# a converged limit cycle, so this is the cheap question asked first.
#
# Runtime: ~5 h per task at L=24 (480x240, 200k steps, ~1.4 MLUPS observed).
# Asking 16 h because the last two estimates from laptop timings were both low
# and both jobs got killed.
#
# Usage: sbatch --array=0-3 cluster/flap_onset.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; . .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
export PYTHONPATH=src
KBS=(5e-3 1e-2 1.5e-2 2e-2)
KB=${KBS[$SLURM_ARRAY_TASK_ID]}
mkdir -p results/flapons
python src/flapping.py --L 24 --Re 100 --u-in 0.1 --KB "$KB" --rho 1.0 \
    --steps 200000 --n-ib 5 \
    --out "results/flapons/KB${KB}.json" \
    --save-signal "results/flapons/KB${KB}_signal.npz"
