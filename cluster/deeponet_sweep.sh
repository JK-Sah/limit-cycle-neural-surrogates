#!/bin/bash -l
#SBATCH --job-name=donet
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=32g
#SBATCH --output=logs/donet_%A_%a.out
#SBATCH --error=logs/donet_%A_%a.err
# DeepONet, matched to the FNO sweep so the two operator families differ only
# in architecture.
#
# FNO and DeepONet factorise the problem differently -- global mixing through a
# truncated spectrum versus a branch/trunk split with a learned spatial basis.
# If delta is a property of the problem rather than of one inductive bias, both
# should show it. Demonstrating it on only one leaves the obvious objection
# open, and the trunk here is especially worth testing because on a fixed grid
# it amounts to a LEARNED basis weighted by branch coefficients, which is
# structurally close to what the POD latent models do with a computed one.
#
# Same three configurations as tasks 1-3 of the FNO sweep:
#   0  seg-len 25, residual
#   1  seg-len 25, direct
#   2  seg-len 25, noise injection
#
# Usage: sbatch --array=0-2 cluster/deeponet_sweep.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; . .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
export PYTHONPATH=src
mkdir -p results/deeponet
case $SLURM_ARRAY_TASK_ID in
  0) ARGS="--variant residual --noise 0.0"  ; TAG=seg25 ;;
  1) ARGS="--variant direct   --noise 0.0"  ; TAG=seg25direct ;;
  2) ARGS="--variant residual --noise 1e-3" ; TAG=seg25noise ;;
esac
python src/deeponet.py --Re 100 --modes-pod 8 --p 128 \
    --epochs 3000 --seg-len 25 --batch 8 --periods 100 \
    --seed-list 0 --threads "$SLURM_CPUS_PER_TASK" \
    $ARGS --out "results/deeponet/${TAG}.json"
