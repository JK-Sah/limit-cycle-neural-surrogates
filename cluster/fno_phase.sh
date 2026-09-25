#!/bin/bash -l
#SBATCH --job-name=fnophase
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=48g
#SBATCH --output=logs/fnophase_%A_%a.out
#SBATCH --error=logs/fnophase_%A_%a.err
#
# Can delta be controlled during training of a neural operator?
#
# Everything else in this study either measures delta or repairs it after the
# fact on a linear operator. Section 5 shows that imposing neutrality and a
# measured frequency drives delta to the precision of the frequency estimate,
# but only for a 13-point parametric ROM repaired by eigenvalue surgery. That
# route does not exist for an FNO, so this puts the frequency into the
# objective instead: a penalty on the relative period error implied by the
# training segment, measured from the angular advance of the leading POD pair.
#
# SEG LEN 32, NOT 25. The estimator averages the per-step angle increment over
# a segment, and over a partial arc that average is biased by the orbit's
# non-uniform instantaneous rate -- the same non-uniformity that killed the
# earlier pointwise phase-anchoring attempt. Over a CLOSED orbit the
# accumulated angle is exactly 2*pi whatever the shape. Measured on a
# wake-like synthetic orbit (axis ratio 0.95, 2% third harmonic):
#
#     seg_len   periods   bias       per-batch sd
#        25      0.78     -1.4e-4      2.0e-3
#        32      1.00     -2.5e-6      4.8e-5
#        48      1.49     -2.5e-6      4.8e-5
#
# T_snap is 32.15 snapshots per period here, so 32 covers 0.995 of a cycle and
# the bias drops 56x. It saturates past one period, so there is no reason to
# pay for a longer segment.
#
# That makes seg_len a confound against the existing seg25 rows, so config 0 is
# a seg32 phase-off control. Compare the phase rows against THAT, not against
# Extended Data Table 4.
#
# Weights: the penalty is Huber in delta, so its gradient is ~w*delta near
# zero, against a fit term around 1e-5. w = 10 is a nudge, 1e3 is comparable,
# 1e5 is dominant -- but clip_grad_norm_ at 1.0 caps the whole update, so 1e5
# is expected to look much like 1e3. That saturation is worth confirming
# rather than assuming.
#
# 4 configs x 3 seeds = 12 tasks. Expect 13-15 h each: the seg25 runs took ~10 h
# median and 12.2 h worst, and 32 steps is ~28% more work per epoch.
#
# Memory: a 6-epoch probe of this exact configuration peaked at 22.4 GB
# resident (seg_len 32, phase term active), so 32g leaves only 30% headroom and
# 16g OOMs outright. 48g is requested to keep a long run from dying late.
#
# Usage: sbatch --array=0-11 cluster/fno_phase.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; . .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
export PYTHONPATH=src
mkdir -p results/fno_phase logs

CFG=$((SLURM_ARRAY_TASK_ID / 3))
SEED=$((SLURM_ARRAY_TASK_ID % 3))
case $CFG in
  0) W=0.0   ; TAG=seg32ctrl  ;;
  1) W=1e1   ; TAG=seg32p1e1  ;;
  2) W=1e3   ; TAG=seg32p1e3  ;;
  3) W=1e5   ; TAG=seg32p1e5  ;;
esac

python src/fno.py --Re 100 --modes-pod 8 \
    --width 32 --fmodes 16 12 --depth 4 \
    --epochs 3000 --seg-len 32 --batch 8 --periods 100 \
    --variant residual --noise 0.0 \
    --phase-weight "$W" --phase-warmup 0.5 \
    --seed-list "$SEED" --threads "$SLURM_CPUS_PER_TASK" \
    --out "results/fno_phase/${TAG}_s${SEED}.json"

echo "done cfg=$CFG w=$W seed=$SEED on $(hostname)"
