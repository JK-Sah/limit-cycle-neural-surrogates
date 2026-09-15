#!/bin/bash -l
#SBATCH --job-name=fnokolm
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --time=1-12:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=48g
#SBATCH --output=logs/fnokolm_%A_%a.out
#SBATCH --error=logs/fnokolm_%A_%a.err
# delta on a Navier-Stokes field, not on eight POD coefficients.
#
# This answers the scope objection the study has carried throughout: every other
# system it measures is a two-dimensional ODE or a cylinder wake compressed to a
# small latent space, which leaves open whether the period error is a property
# of the problem or of low-dimensional latent dynamics. Here the operator acts
# on a 128 x 128 vorticity field of the 2D Navier-Stokes equations, in the
# forcing geometry of the FNO benchmark, at the one viscosity where a limit
# cycle exists (nu = 1.7e-3, split-half 4.8e-9 over 51 periods).
#
# Three configurations x three seeds. The configurations mirror the wake sweep
# so the two are comparable, and the seeds are not optional: seed spread in
# delta is a reported quantity, and on the wake it reached 890x for the noise
# variant while one residual seed diverged at a training loss indistinguishable
# from the seeds that survived.
#
# Usage: sbatch --array=0-8 cluster/fno_kolm.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; . .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
export PYTHONPATH=src
mkdir -p results/fno_kolm
CFG=$((SLURM_ARRAY_TASK_ID / 3))
SEED=$((SLURM_ARRAY_TASK_ID % 3))
case $CFG in
  0) ARGS="--variant residual --noise 0.0"  ; TAG=residual ;;
  1) ARGS="--variant direct   --noise 0.0"  ; TAG=direct ;;
  2) ARGS="--variant residual --noise 1e-3" ; TAG=noise ;;
esac
python src/fno_kolm.py --data data/kolm/nu1.7e-3_snaps.npz \
    --modes-pod 8 --width 32 --fmodes 16 16 --depth 4 \
    --epochs 3000 --seg-len 25 --batch 8 --periods 100 \
    --seed-list "$SEED" --threads "$SLURM_CPUS_PER_TASK" \
    $ARGS --out "results/fno_kolm/${TAG}_s${SEED}.json"
