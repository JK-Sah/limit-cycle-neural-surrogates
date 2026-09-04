#!/bin/bash -l
#SBATCH --job-name=fnoseed
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=32g
#SBATCH --output=logs/fnoseed_%A_%a.out
#SBATCH --error=logs/fnoseed_%A_%a.err
# Seeds for the three FNO configurations that completed their rollout.
#
# Seed spread is not a robustness footnote in this paper, it is a reported
# quantity: the breadth table shows learned models spanning 3.5x to 6e4x in
# delta at fixed architecture and data, and the practical recommendation is that
# a single-seed delta says almost nothing. Quoting one-seed FNO numbers next to
# five-seed numbers for everything else would be exactly the inconsistency the
# paper warns about.
#
# 3 configs x seeds 1-4 (seed 0 already done) = 12 tasks, one seed each so they
# run in parallel rather than serially inside one job.
#
# Usage: sbatch --array=0-11 cluster/fno_seeds.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; . .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
export PYTHONPATH=src
mkdir -p results/fno_seeds
CFG=$((SLURM_ARRAY_TASK_ID / 4))
SEED=$(( (SLURM_ARRAY_TASK_ID % 4) + 1 ))
case $CFG in
  0) ARGS="--variant residual --noise 0.0"  ; TAG=seg25 ;;
  1) ARGS="--variant direct   --noise 0.0"  ; TAG=seg25direct ;;
  2) ARGS="--variant residual --noise 1e-3" ; TAG=seg25noise ;;
esac
python src/fno.py --Re 100 --modes-pod 8 \
    --width 32 --fmodes 16 12 --depth 4 \
    --epochs 3000 --seg-len 25 --batch 8 --periods 100 \
    --seed-list "$SEED" --threads "$SLURM_CPUS_PER_TASK" \
    $ARGS --out "results/fno_seeds/${TAG}_s${SEED}.json"
