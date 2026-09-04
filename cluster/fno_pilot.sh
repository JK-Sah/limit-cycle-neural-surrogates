#!/bin/bash -l
#SBATCH --job-name=fnopilot
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=32g
#SBATCH --output=logs/fnopilot_%A.out
#SBATCH --error=logs/fnopilot_%A.err
# Pilot before committing the full FNO sweep.
#
# The architecture-breadth table has no neural OPERATOR in it, which is the
# obvious gap for the venue: FNO and DeepONet are what the neural-PDE community
# actually deploys, and the paper's claim is aimed at them. This trains an FNO
# on the wake FIELDS (not on POD coefficients -- that would test a different
# object) and measures delta by projecting the rollout onto the same 8-mode POD
# basis and fitting the period with the same estimator, so the number drops
# straight into the table.
#
# Purpose here is narrow: does the loss fall, and does a 3215-step rollout stay
# finite? A full run is ~6 x 20 h and is not worth starting until both hold.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; . .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
export PYTHONPATH=src
mkdir -p results/fno
python src/fno.py --Re 100 --modes-pod 8 --variant residual \
    --width 32 --fmodes 16 12 --depth 4 \
    --epochs 300 --seg-len 8 --batch 8 --periods 100 \
    --seed-list 0 --threads "$SLURM_CPUS_PER_TASK" \
    --out results/fno/pilot.json
