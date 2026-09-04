#!/bin/bash -l
#SBATCH --job-name=dodirect
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=32g
#SBATCH --output=logs/dodirect_%A_%a.out
#SBATCH --error=logs/dodirect_%A_%a.err
# Why does the direct DeepONet train 34x worse than its residual form?
#
# Three seeds land at train MSE 1.181e-2, 1.104e-2, 1.179e-2 -- a plateau, not
# seed scatter, so it is hitting something rather than getting unlucky. Two
# candidate causes with opposite consequences for the paper:
#
#   optimisation  it needs longer or a different step size, in which case the
#                 number is an artefact of my configuration and must not be
#                 reported as a property of the architecture
#   structural    the residual form only has to represent an increment, while
#                 the direct form must reproduce the whole field INCLUDING the
#                 mean flow, with a single scalar per channel to help it. If so
#                 the trunk is spending its basis on a component that never
#                 varies, and a learnable mean field should fix it
#
# Four tasks separate them. Task 0 repeats the baseline as a control so the
# comparison is against a run from this same code, not a remembered number.
#
# Usage: sbatch --array=0-3 cluster/donet_direct.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; . .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
export PYTHONPATH=src
mkdir -p results/donet_direct
case $SLURM_ARRAY_TASK_ID in
  0) ARGS="--epochs 3000 --lr 1e-3"              ; TAG=control ;;
  1) ARGS="--epochs 3000 --lr 3e-3"              ; TAG=lr3e-3 ;;
  2) ARGS="--epochs 9000 --lr 1e-3"              ; TAG=ep9000 ;;
  3) ARGS="--epochs 3000 --lr 1e-3 --field-bias" ; TAG=fieldbias ;;
esac
python src/deeponet.py --Re 100 --modes-pod 8 --p 128 --n-freq 16 \
    --seg-len 25 --batch 8 --periods 100 --variant direct \
    --seed-list 0 --threads "$SLURM_CPUS_PER_TASK" \
    $ARGS --out "results/donet_direct/${TAG}.json"
