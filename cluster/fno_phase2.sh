#!/bin/bash -l
#SBATCH --job-name=fnophase2
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=48g
#SBATCH --output=logs/fnophase2_%A_%a.out
#SBATCH --error=logs/fnophase2_%A_%a.err
#
# The phase term at the weights it can actually carry.
#
# The first array (21784478) guessed {1e1, 1e3, 1e5} and two of those were
# hopeless. Pilot 21784818 showed w = 1e3 takes the fit from 4.7e-4 to 8.4e+0
# within 50 epochs of engagement, on all three seeds. Scale probe 21785421 then
# swept downward at the same budget and found the ceiling:
#
#     w       fit at ep 300     vs control (1.7e-4)
#     1e-2      1.8e-4          no damage
#     1e-1      1.6e-4          no damage
#     1e0       3.0e-4          mild, transient, recovering
#     1e3       ~1e-2           catastrophic
#
# So the usable range is w <= ~1, and this array runs the two weights that
# bracket it at the production budget. The gradient argument says 1e-1 is the
# balanced one: the penalty gradient is about w*delta, so at delta ~ 5e-4 that
# is 5e-5 against an MSE gradient of order the loss, ~1e-5 by epoch 1500. At
# w = 1e0 it is 50x the fit gradient, which is why the probe saw a bump.
#
# NOTE this differs from the probe in one way that matters: warmup is 0.5 here
# rather than 0.25, so the term engages at epoch 1500 where the loss is ~1e-5
# rather than at epoch 150 where it is ~4e-4. The MSE gradient is ~40x smaller
# at that point, so the phase term is RELATIVELY stronger than the probe saw.
# If 1e-1 also proves too strong here, 1e-2 is the next step down and the probe
# already shows it costs the fit nothing.
#
# Together with the surviving arms of 21784478 -- three phase-off controls and
# two seeds at w = 1e1 -- this completes a sweep at 0, 1e-1, 1e0, 1e1 on one
# budget, one segment length and shared seeds.
#
# 2 weights x 3 seeds = 6 tasks, ~13.5 h each at the 15.8 s/epoch measured on
# the running controls.
#
# Usage: sbatch --array=0-5 cluster/fno_phase2.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; . .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
export PYTHONPATH=src
mkdir -p results/fno_phase2 logs

CFG=$((SLURM_ARRAY_TASK_ID / 3))
SEED=$((SLURM_ARRAY_TASK_ID % 3))
case $CFG in
  0) W=1e-1 ; TAG=seg32p1em1 ;;
  1) W=1e0  ; TAG=seg32p1e0  ;;
esac

python src/fno.py --Re 100 --modes-pod 8 \
    --width 32 --fmodes 16 12 --depth 4 \
    --epochs 3000 --seg-len 32 --batch 8 --periods 100 \
    --variant residual --noise 0.0 \
    --phase-weight "$W" --phase-warmup 0.5 \
    --seed-list "$SEED" --threads "$SLURM_CPUS_PER_TASK" \
    --out "results/fno_phase2/${TAG}_s${SEED}.json"

echo "done cfg=$CFG w=$W seed=$SEED on $(hostname)"
