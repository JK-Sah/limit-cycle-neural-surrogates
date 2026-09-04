#!/bin/bash -l
#SBATCH --job-name=ibcyl
#SBATCH --account=flowlab
#SBATCH --partition=debug
#SBATCH --time=0-14:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16g
#SBATCH --output=logs/ibcyl_%A_%a.out
#SBATCH --error=logs/ibcyl_%A_%a.err
# Does the immersed-boundary path reproduce the Strouhal number the bounce-back
# solver already gives? Same box (8D upstream / 22D downstream, ny_D = 20), same
# Re, same probe, same period estimator -- only the body representation differs.
# Bounce-back reference at this box: 0.16643 (D=16), 0.16851 (D=24),
# 0.16889 (D=32). Agreement to well under a percent means the coupling and the
# force scaling are right; disagreement means nothing downstream is worth
# running.
#
# D=16 is already done and gave St = 0.16308, 2.0% below bounce-back. If that
# gap is the immersed boundary's effective-diameter offset -- the delta function
# smears no-slip over about a lattice unit, so the body the flow sees is
# slightly larger than the marker circle -- it is a fixed offset in lattice
# units and must shrink as 1/D: about 1.4% at D=24 and 1.0% at D=32. If instead
# it stays near 2%, it is not geometry and the coupling is wrong.
#
# One D per task. Runtime is ~4 h at D=24 and ~7 h at D=32 on this partition;
# the first attempt asked for 3 h and both were killed.
# Usage: sbatch --array=0-1 cluster/ib_cyl_validate.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; . .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
export PYTHONPATH=src
DS=(24 32)
D=${DS[$SLURM_ARRAY_TASK_ID]}
mkdir -p results/ibcyl
python src/iblbm.py --D "$D" --Re 100 --steps 60000 --n-iter 5 \
    --out "results/ibcyl/D${D}.json" \
    --save-signal "results/ibcyl/D${D}_signal.npz"
