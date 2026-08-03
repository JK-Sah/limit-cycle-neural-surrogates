#!/bin/bash -l
#SBATCH --job-name=cylres
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16g
#SBATCH --output=logs/cyl_%A_%a.out
#SBATCH --error=logs/cyl_%A_%a.err

# Resolution study for the cylinder wake at Re=100.
# Strouhal number must converge toward the literature value (0.1643) as the
# staircase cylinder is refined, otherwise the reference period is not
# trustworthy and delta cannot be measured against it.
# Usage: sbatch --array=0-3 cluster/cyl_res_array.sh

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
. .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

D_LIST=(16 24 32 40)
D=${D_LIST[$SLURM_ARRAY_TASK_ID]}
# hold the number of shed cycles roughly fixed as D grows
STEPS=$(( 60000 + 3000 * D ))

mkdir -p results/cyl signals
python src/cylinder_lbm.py \
    --D "$D" \
    --steps "$STEPS" \
    --out results/cyl/D${D}.json \
    --save-signal signals/D${D}_signal.npz

echo "D=$D done on $(hostname)"
