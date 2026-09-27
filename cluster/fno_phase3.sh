#!/bin/bash -l
#SBATCH --job-name=fnophase3
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=48g
#SBATCH --output=logs/fnophase3_%A_%a.out
#SBATCH --error=logs/fnophase3_%A_%a.err
#
# Seeds 3-7 for the control and w = 1e-1, the weight the sweep picked out.
#
# The sweep at three seeds each gave:
#
#     w      med |delta|   seed spread   med train MSE
#     0        2.70e-4        102x          1.23e-5
#     1e-1     8.23e-5        2.1x          1.32e-5
#     1e0      1.71e-4      63000x          1.77e-5
#     1e1      9.48e-4        4.8x          4.30e-5
#
# w = 1e-1 improves the median 3.3x for a 7% fit cost. But an exact permutation
# test on 3 v 3 medians gives p = 0.20, and with three seeds per arm the
# SMALLEST attainable p is 0.05 -- the design cannot produce a significant
# result whatever the data. Hence more seeds.
#
# What the effect actually is, stated carefully, because it is not what the
# median suggests. The control's BEST seed (1.35e-5) beats every w = 1e-1 seed.
# The term does not improve the best achievable delta; it deletes the bad tail.
# Control spans 1.35e-5 to 1.38e-3, two decades. w = 1e-1 spans 5.45e-5 to
# 1.15e-4, a factor of 2.1. Worst case improves 12x.
#
# That is worth having on its own terms. This paper's own recommendation is to
# report more than one seed because delta spreads up to 6e4x at fixed
# architecture and data, and no loss-based criterion can select against that.
# A term that collapses the spread to 2.1x turns delta from a quantity you
# cannot rely on into one you can.
#
# Five more seeds per arm takes n to 8 and lets the permutation test reach
# p ~ 2e-4 if the effect is real. Seeds 3-7 so nothing already run is repeated.
#
# 2 arms x 5 seeds = 10 tasks, ~13.5 h each.
#
# Usage: sbatch --array=0-9 cluster/fno_phase3.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; . .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
export PYTHONPATH=src
mkdir -p results/fno_phase3 logs

CFG=$((SLURM_ARRAY_TASK_ID / 5))
SEED=$((SLURM_ARRAY_TASK_ID % 5 + 3))
case $CFG in
  0) W=0.0  ; TAG=seg32ctrl  ;;
  1) W=1e-1 ; TAG=seg32p1em1 ;;
esac

python src/fno.py --Re 100 --modes-pod 8 \
    --width 32 --fmodes 16 12 --depth 4 \
    --epochs 3000 --seg-len 32 --batch 8 --periods 100 \
    --variant residual --noise 0.0 \
    --phase-weight "$W" --phase-warmup 0.5 \
    --seed-list "$SEED" --threads "$SLURM_CPUS_PER_TASK" \
    --out "results/fno_phase3/${TAG}_s${SEED}.json"

echo "done cfg=$CFG w=$W seed=$SEED on $(hostname)"
