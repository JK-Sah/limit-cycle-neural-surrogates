#!/bin/bash -l
#SBATCH --job-name=kolmons
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --time=0-14:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=16g
#SBATCH --output=logs/kolmons_%A_%a.out
#SBATCH --error=logs/kolmons_%A_%a.err
# Resolve the transition band properly before concluding there is no periodic
# window in the FNO forcing geometry.
#
# The 20-point sweep found steady flow for nu >= 2.5e-3 and non-periodic motion
# for nu <= 2.0e-3, with no periodic band between. That conclusion is not safe:
# at nu = 2.0e-3 the amplitude is 1.9e-6, which is a perturbation still growing
# rather than a saturated attractor, and the split-half number there is fitting
# numerical noise. Near a Hopf bifurcation the growth rate vanishes and the time
# to saturate diverges, so the 200-unit transient used in that sweep is far too
# short precisely where the answer matters.
#
# This is the same failure that produced a wrong verdict on the flapping
# filament -- reading a regime off runs that had not settled -- so the fix is
# the same: a transient long enough that the envelope stops moving, on a fine
# grid through the band.
#
# transient 3000, record 800, seven viscosities from 1.6e-3 to 2.4e-3.
# Usage: sbatch --array=0-6 cluster/kolm_onset.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; . .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
export PYTHONPATH=src
NUS=(2.4e-3 2.2e-3 2.0e-3 1.9e-3 1.8e-3 1.7e-3 1.6e-3)
NU=${NUS[$SLURM_ARRAY_TASK_ID]}
mkdir -p results/kolm_onset
python src/kolmogorov.py --nu "$NU" --N 128 --dt 2e-3 \
    --transient 3000 --record 800 --forcing fno \
    --out "results/kolm_onset/nu${NU}.json"
