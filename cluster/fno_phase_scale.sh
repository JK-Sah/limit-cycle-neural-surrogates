#!/bin/bash -l
#SBATCH --job-name=fnoscale
#SBATCH --account=flowlab
#SBATCH --partition=debug
#SBATCH --time=06:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=48g
#SBATCH --output=logs/fnoscale_%A_%a.out
#SBATCH --error=logs/fnoscale_%A_%a.err
#
# What weight can the phase term actually carry?
#
# The first pilot (21784818) settled this negatively at w = 1e3: the term
# engages at epoch 150 with the orbit already good, delta_batch = +1.7e-3, and
# within 50 epochs the fit has gone from 4.7e-4 to 8.4e+0 -- four orders of
# magnitude -- on all three seeds. The cause is a scale mismatch, not a bug.
# The penalty gradient runs about w * delta, so at w = 1e3 and delta ~ 1e-3
# that is ~1, against an MSE gradient of order the loss itself, ~1e-4. With
# clip_grad_norm_ at 1.0 the whole update becomes the phase direction and the
# fit is abandoned. Ironically delta_batch then gets WORSE, reaching 2.6e-1,
# because the model is being yanked off its orbit.
#
# So the useful weight is set by the gradient ratio, and the first array's
# {1e1, 1e3, 1e5} was chosen two to five orders of magnitude too high. This
# probes downward: 1e-2, 1e-1, 1e0.
#
# One seed per weight is enough here because the question is whether the FIT
# survives, and fit damage was not seed-dependent in the pilot -- all three
# w = 1e3 seeds landed at 2.2e-2 to 2.9e-1 against a control at 1.0e-4. Once a
# weight is shown to leave the fit intact, it gets the full seed treatment.
#
# Budget and warm-up match pilot 21784818 exactly (600 epochs, warmup 0.25) so
# the existing control arm and the w = 1e3 arm both serve as reference without
# being re-run. Damage shows within 50 epochs of engagement, so the verdict is
# readable long before the job ends.
#
# Usage: sbatch --array=0-2 cluster/fno_phase_scale.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; . .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
export PYTHONPATH=src
mkdir -p results/fno_phase_scale logs

case $SLURM_ARRAY_TASK_ID in
  0) W=1e-2 ; TAG=scale_w1em2 ;;
  1) W=1e-1 ; TAG=scale_w1em1 ;;
  2) W=1e0  ; TAG=scale_w1e0  ;;
esac

python src/fno.py --Re 100 --modes-pod 8 \
    --width 32 --fmodes 16 12 --depth 4 \
    --epochs 600 --seg-len 32 --batch 8 --periods 100 \
    --variant residual --noise 0.0 \
    --phase-weight "$W" --phase-warmup 0.25 \
    --seed-list 0 --threads "$SLURM_CPUS_PER_TASK" \
    --out "results/fno_phase_scale/${TAG}_s0.json"

echo "done scale w=$W on $(hostname)"
