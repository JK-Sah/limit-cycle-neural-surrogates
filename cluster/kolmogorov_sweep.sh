#!/bin/bash -l
#SBATCH --job-name=kolm
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=16g
#SBATCH --output=logs/kolm_%A_%a.out
#SBATCH --error=logs/kolm_%A_%a.err
# Is delta even defined on the Fourier neural operator's Navier-Stokes setting?
#
# The FNO benchmark ships at nu = 1e-3, 1e-4, 1e-5, roughly Re = 20, 200, 2000.
# Two-dimensional Kolmogorov-type flow has its first Hopf bifurcation near
# Re = 32 and sustained chaos above Re ~ 103. The benchmark's three viscosities
# therefore sit below the first bifurcation and above the onset of chaos: they
# bracket the periodic windows without landing in one. A delta quoted on a
# chaotic trajectory measures nothing, which is the whole point of section 9.
#
# A first sweep from nu = 1e-3 DOWNWARD found no periodic window: all ten points
# were quasi-periodic or chaotic, with split-half instability between 3.7e-4 and
# 1.1e-1, never near the 1e-7 that marks a well-posed period fit. That sweep went
# the wrong way. The benchmark's mildest setting is already past the transition,
# so the periodic window, if it exists for this forcing, lies at HIGHER viscosity
# than any point the benchmark ships. This sweeps upward.
#
# So rather than assume, this sweeps nu across the range the benchmark skips and
# applies the split-half well-posedness test at each point -- the same test a
# practitioner would run before trusting a delta. Output is a regime map:
# steady, periodic, or neither. Only the periodic band can carry the
# measurement, and finding it is the precondition for the experiment, not the
# experiment.
#
# Solver is the FNO paper's own equations and forcing, integrated by
# integrating-factor RK4 with 2/3 dealiasing. Verified: inviscid unforced
# enstrophy drift 1.2e-16 over 200 steps, pure diffusion matching the analytic
# decay to 5e-15.
#
# Usage: sbatch --array=0-9 cluster/kolmogorov_sweep.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; . .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
export PYTHONPATH=src
NUS=(1.2e-3 1.5e-3 2e-3 2.5e-3 3e-3 4e-3 5e-3 7e-3 1e-2 1.5e-2)
NU=${NUS[$SLURM_ARRAY_TASK_ID]}
mkdir -p results/kolmogorov
python src/kolmogorov.py --nu "$NU" --N 128 --dt 2e-3 \
    --transient 200 --record 400 --forcing fno \
    --out "results/kolmogorov/nu${NU}.json"
