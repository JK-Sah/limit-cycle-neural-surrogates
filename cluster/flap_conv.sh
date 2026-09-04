#!/bin/bash -l
#SBATCH --job-name=flapconv
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --time=0-20:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=24g
#SBATCH --output=logs/flapconv_%A_%a.out
#SBATCH --error=logs/flapconv_%A_%a.err
# Is the KB* = 1e-2 limit cycle resolution-converged?
#
# At L=24, tau=0.572 it is clean: envelope constant to five figures, jitter
# 6.0e-5, split-half 1.5e-6. But the previous attempt looked fine at one
# resolution too and fell apart at the next -- L=16 flapped, L=24 decayed --
# because tau was 0.5192, near the lattice Boltzmann floor. That failure is the
# reason this check exists rather than being assumed.
#
# Here tau = 0.003 L + 0.5 at fixed Re=100, u=0.1: 0.548 at L=16, 0.572 at
# L=24, 0.596 at L=32. All comfortably above the floor, so if the cycle is
# physical the period and amplitude should agree across the three and the
# jitter should fall with resolution. If instead the state changes character
# again, the flapping filament does not have a usable limit cycle at these
# parameters and the case should be reconsidered rather than pushed.
#
# Runtimes from the measured 0.088 s/step at L=24 scaled by cell count:
# L=16 ~1.5 h, L=32 ~12 h. Asking 20 h.
#
# Usage: sbatch --array=0-1 cluster/flap_conv.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; . .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
export PYTHONPATH=src
LS=(16 32)
ST=(150000 300000)
L=${LS[$SLURM_ARRAY_TASK_ID]}
S=${ST[$SLURM_ARRAY_TASK_ID]}
mkdir -p results/flapconv
python src/flapping.py --L "$L" --Re 100 --u-in 0.1 --KB 1e-2 --rho 1.0 \
    --steps "$S" --n-ib 5 \
    --out "results/flapconv/L${L}.json" \
    --save-signal "results/flapconv/L${L}_signal.npz"
