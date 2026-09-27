#!/bin/bash -l
#SBATCH --job-name=kolmphase
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --time=1-12:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=48g
#SBATCH --output=logs/kolmphase_%A_%a.out
#SBATCH --error=logs/kolmphase_%A_%a.err
#
# The phase-consistency term on the Navier-Stokes field, at nu = 1.7e-3.
#
# Same question as cluster/fno_phase.sh, on the system where the residual
# operator fits fifteen times better than the direct one and has a fifty times
# worse period. If a term in the objective can close that gap, the paper has a
# constructive result on the architecture it is aimed at rather than only on a
# linear ROM. If it cannot, that is section 8's structural argument holding on
# an operator, which is also worth reporting.
#
# SEG LEN 40, NOT 32. The bias argument in fno_phase.sh is about covering a
# CLOSED orbit, so the right segment length is one period in snapshot units,
# and here T_snap is 40.02 rather than the wake's 32.15 (100 periods = 4002
# steps, per Extended Data Table 5). Config 0 is again a phase-off control at
# the same segment length, because seg_len is otherwise a confound against the
# seg25 rows already in the table.
#
# The seg25 runs here took ~10.3 h median; 40 steps is ~60% more work per
# epoch, so budget ~16 h, hence the longer wall clock than the wake array.
#
# WEIGHTS CORRECTED. This script originally swept {0, 1e1, 1e3, 1e5}, chosen
# before any of them had been tested. The wake sweep has since shown every one
# of those except the control sits at or above the ceiling the term can carry:
# w = 1e3 takes the fit from 4.7e-4 to 8.4e+0 within 50 epochs of engagement,
# and w = 1e1 leaves delta 3.5x WORSE than control. The weight that worked was
# 1e-1, three orders below the original guess, so this runs the control against
# that alone.
#
# On the wake, w = 1e-1 gave median |delta| 8.23e-5 against the control's
# 2.70e-4 for a 7% fit cost, and collapsed the seed spread from 102x to 2.1x.
# The question here is whether that transfers to a second system, which is what
# would make it a design recommendation rather than an idiosyncrasy of one
# problem -- the same standard the direct-parameterisation result was held to.
#
# 2 arms x 3 seeds = 6 tasks.
#
# Memory: NOT measured on this configuration. The equivalent probe on the
# WAKE (seg_len 32, phase term active) peaked at 22.4 GB resident, and 16g
# OOMed outright. This field is 128x128x1 against the wake's 2x192x96, so
# per-state activations are smaller while the segment is 40 rather than 32;
# 48g should be ample either way, but check MaxRSS on the first task rather
# than trusting the extrapolation.
#
# Usage: sbatch --array=0-5 cluster/fno_kolm_phase.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; . .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
export PYTHONPATH=src
mkdir -p results/fno_kolm_phase logs

CFG=$((SLURM_ARRAY_TASK_ID / 3))
SEED=$((SLURM_ARRAY_TASK_ID % 3))
case $CFG in
  0) W=0.0   ; TAG=seg40ctrl  ;;
  1) W=1e-1  ; TAG=seg40p1em1 ;;
esac

python src/fno_kolm.py --data data/kolm/nu1.7e-3_snaps.npz \
    --modes-pod 8 --width 32 --fmodes 16 16 --depth 4 \
    --epochs 3000 --seg-len 40 --batch 8 --periods 100 \
    --variant residual --noise 0.0 \
    --phase-weight "$W" --phase-warmup 0.5 \
    --seed-list "$SEED" --threads "$SLURM_CPUS_PER_TASK" \
    --out "results/fno_kolm_phase/${TAG}_s${SEED}.json"

echo "done cfg=$CFG w=$W seed=$SEED on $(hostname)"
