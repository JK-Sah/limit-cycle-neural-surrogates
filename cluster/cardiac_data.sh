#!/bin/bash -l
#SBATCH --job-name=apdata
#SBATCH --account=flowlab
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16g
#SBATCH --time=06:00:00
#SBATCH --output=logs/apdata_%A_%a.out
#SBATCH --error=logs/apdata_%A_%a.err
#
# Aliev-Panfilov spiral wave: the training record, and two refinements of it.
#
# Task 0 is the record the surrogates are trained on: 256 x 256 at dx = 0.4,
# dt = 0.02, coarsened to 128 x 128, 60 periods after a 600 t.u. transient.
# Tasks 1 and 2 halve nothing but one discretisation parameter each (dx 0.4 ->
# 0.32 at fixed domain size, dt 0.02 -> 0.01) and record 20 periods, so the
# period's discretisation error can be reported. The surrogate study does not
# depend on it -- delta is measured against the discrete system the surrogate
# was trained on, as on Navier-Stokes -- but the filament solver in this repo
# was set aside precisely because its period did not converge, so the number
# is stated rather than assumed.
#
# Usage: sbatch -p debug,tier3 --qos=qos_tier3 --array=0-2 cluster/cardiac_data.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; . .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1 PYTHONPATH=src
mkdir -p data/cardiac results/cardiac logs
case $SLURM_ARRAY_TASK_ID in
  0) ARGS="--N 256 --dx 0.4  --dt 0.02 --periods 60"; TAG=ap_main ;;
  1) ARGS="--N 320 --dx 0.32 --dt 0.02 --periods 20"; TAG=ap_dx032 ;;
  2) ARGS="--N 256 --dx 0.4  --dt 0.01 --periods 20"; TAG=ap_dt001 ;;
esac
python src/cardiac.py $ARGS --out "data/cardiac/$TAG"
cp "data/cardiac/$TAG.json" "results/cardiac/$TAG.json"
