#!/bin/bash -l
#SBATCH --job-name=fnosweep
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=32g
#SBATCH --output=logs/fnosweep_%A_%a.out
#SBATCH --error=logs/fnosweep_%A_%a.err
# FNO on the wake, trained properly, with the two confounds separated.
#
# The pilot trained to mse 1.7e-4 and then diverged at step 1221 of 3215 -- 38
# of 100 periods. That number is uninterpretable as it stands, because 300
# epochs at batch 8 is about 1.5 passes over 1546 segments while every other
# model in the table got 4000 epochs at batch 64. Undertraining and genuine
# rollout instability look identical from one run, and reporting the second
# when it was the first is the error that produced the retracted 400-epoch
# decorrelation result.
#
# Four tasks separate them:
#   0  seg-len 8,  3000 epochs          is it just undertrained?
#   1  seg-len 25, 3000 epochs          matched to the other models' objective
#                                       horizon, which section 4 shows matters
#                                       more than the model class
#   2  seg-len 25, direct variant       does the residual parameterisation carry it
#   3  seg-len 25, noise injection      the paper's own critique, applied here:
#                                       if noise rescues the rollout but leaves
#                                       delta unmoved, the thesis is demonstrated
#                                       on the field's flagship architecture
#
# One seed each -- this establishes whether the model is usable at all. Seeds
# come after, since delta spread across seeds is itself a reported quantity.
#
# Cost: the pilot ran 10.1 s/epoch at seg-len 8. Task 0 is ~8 h; the seg-len 25
# tasks are ~3x that, so ~26 h. Asking 48 h.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; . .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
export PYTHONPATH=src
mkdir -p results/fno
case $SLURM_ARRAY_TASK_ID in
  0) ARGS="--seg-len 8  --variant residual --noise 0.0"   ; TAG=seg8 ;;
  1) ARGS="--seg-len 25 --variant residual --noise 0.0"   ; TAG=seg25 ;;
  2) ARGS="--seg-len 25 --variant direct   --noise 0.0"   ; TAG=seg25direct ;;
  3) ARGS="--seg-len 25 --variant residual --noise 1e-3"  ; TAG=seg25noise ;;
esac
python src/fno.py --Re 100 --modes-pod 8 \
    --width 32 --fmodes 16 12 --depth 4 \
    --epochs 3000 --batch 8 --periods 100 \
    --seed-list 0 --threads "$SLURM_CPUS_PER_TASK" \
    $ARGS --out "results/fno/${TAG}.json"
