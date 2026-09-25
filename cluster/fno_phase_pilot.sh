#!/bin/bash -l
#SBATCH --job-name=fnopilot
#SBATCH --account=flowlab
#SBATCH --partition=debug
#SBATCH --time=06:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=48g
#SBATCH --output=logs/fnopilot_%A_%a.out
#SBATCH --error=logs/fnopilot_%A_%a.err
#
# PILOT. Not a result. Does the phase term move delta in the right direction?
#
# The production array (cluster/fno_phase.sh) orders its tasks CFG = id/3, so
# the first three are all phase-off controls and the first weight that matters
# does not start until the third wave. On a busy tier3 running three tasks at a
# time that is better than a day of waiting to find out whether the idea works
# at all. This is the cheap early read: 600 epochs instead of 3000, warmup 0.25
# so the term is active for 450 of them, control against w = 1e3 only.
#
# It runs on `debug` deliberately -- 24 h limit, two 36-core/380 GB nodes, one
# of them idle -- so it does not compete with the production array for tier3.
#
# READ IT FOR DIRECTION, NOTHING ELSE. At 600 epochs the model is badly
# undertrained: the production controls reach ~2.4e-5 training loss by epoch
# 1050 and keep falling, so delta here is dominated by undertraining rather
# than by the phase term. Three seeds per arm because this paper's own finding
# is that a single-seed delta says almost nothing -- spread runs 1.9x to 6e4x
# at fixed everything -- and a one-seed pilot could easily point the wrong way.
# Seeds are shared between the arms so the comparison is paired.
#
# If delta at w = 1e3 is not below the control on a majority of seeds, the term
# is not working and the production array is worth cancelling before it burns
# two more days.
#
# 2 configs x 3 seeds = 6 tasks, ~3 h each (600 epochs at the 15.8 s/epoch
# measured on the production runs, plus the 3215-step measurement rollout).
#
# Usage: sbatch --array=0-5 cluster/fno_phase_pilot.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; . .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
export PYTHONPATH=src
mkdir -p results/fno_phase_pilot logs

CFG=$((SLURM_ARRAY_TASK_ID / 3))
SEED=$((SLURM_ARRAY_TASK_ID % 3))
case $CFG in
  0) W=0.0   ; TAG=pilot_ctrl   ;;
  1) W=1e3   ; TAG=pilot_p1e3   ;;
esac

python src/fno.py --Re 100 --modes-pod 8 \
    --width 32 --fmodes 16 12 --depth 4 \
    --epochs 600 --seg-len 32 --batch 8 --periods 100 \
    --variant residual --noise 0.0 \
    --phase-weight "$W" --phase-warmup 0.25 \
    --seed-list "$SEED" --threads "$SLURM_CPUS_PER_TASK" \
    --out "results/fno_phase_pilot/${TAG}_s${SEED}.json"

echo "done pilot cfg=$CFG w=$W seed=$SEED on $(hostname)"
