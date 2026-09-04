#!/bin/bash -l
#SBATCH --job-name=donetseed
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=32g
#SBATCH --output=logs/donetseed_%A_%a.out
#SBATCH --error=logs/donetseed_%A_%a.err
# DeepONet with a Fourier-embedded trunk, three configs x three seeds.
#
# The first attempt trained 4700x worse than the FNO on identical data and was
# not reported, correctly: the cause was the trunk, not the architecture. A
# plain MLP on raw (x, y) has a strong low-frequency spectral bias, and fitting
# one shedding-like field sin(2pi 6x)cos(2pi 2y) with it reaches MSE 2.2e-1
# against a target variance of 0.251 -- essentially no explanatory power. With
# an axis-aligned Fourier embedding the same MLP reaches 2.9e-5. The rebuilt
# model overfits a single transition to 9.9e-5 against variance 1.0e-2, so it
# now has the capacity the comparison requires.
#
# 3 configs x seeds 0-2 = 9 tasks.
# Usage: sbatch --array=0-8 cluster/deeponet_seeds.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; . .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
export PYTHONPATH=src
mkdir -p results/deeponet_v2
CFG=$((SLURM_ARRAY_TASK_ID / 3))
SEED=$((SLURM_ARRAY_TASK_ID % 3))
case $CFG in
  0) ARGS="--variant residual --noise 0.0"  ; TAG=seg25 ;;
  1) ARGS="--variant direct   --noise 0.0"  ; TAG=seg25direct ;;
  2) ARGS="--variant residual --noise 1e-3" ; TAG=seg25noise ;;
esac
python src/deeponet.py --Re 100 --modes-pod 8 --p 128 --n-freq 16 \
    --epochs 3000 --seg-len 25 --batch 8 --periods 100 \
    --seed-list "$SEED" --threads "$SLURM_CPUS_PER_TASK" \
    $ARGS --out "results/deeponet_v2/${TAG}_s${SEED}.json"
