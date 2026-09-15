#!/bin/bash -l
#SBATCH --job-name=kolmdata
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32g
#SBATCH --output=logs/kolmdata_%A.out
#SBATCH --error=logs/kolmdata_%A.err
# Generate the training set on the periodic Navier-Stokes attractor.
#
# nu = 1.7e-3 is the limit cycle located by the transition sweep: split-half
# 1.12e-8, period 29.38, in the FNO paper's own equations and forcing on a
# 128^2 grid. This is the case that lets delta be measured on a PDE field
# rather than on eight POD coefficients, which is the scope question the whole
# study has been open on.
#
# 3000 time units of transient (the shorter one used earlier misclassified a
# decaying perturbation as a growing one), then 1500 units of record -- about
# 51 periods -- sampled 40 times per period.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; . .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
export PYTHONPATH=src
mkdir -p data/kolm results/kolm_data
python src/kolmogorov.py --nu 1.7e-3 --N 128 --dt 2e-3 \
    --transient 3000 --record 1500 --forcing fno \
    --snap-every 367 --save-snaps data/kolm/nu1.7e-3_snaps.npz \
    --out results/kolm_data/nu1.7e-3.json
