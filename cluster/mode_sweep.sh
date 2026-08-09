#!/bin/bash -l
#SBATCH --job-name=modesw
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --time=0-16:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48g
#SBATCH --output=logs/modesw_%A_%a.out
#SBATCH --error=logs/modesw_%A_%a.err
# Does delta track how dynamically closed the retained POD subspace is?
# DMD gives 2e-8 at 16 modes and 4.58e-4 at 8, so the classical-vs-learned
# conclusion depends on truncation and has to be pinned down.
# One mode count per array task, 3 seeds each.
# Usage: sbatch --array=0-5 cluster/mode_sweep.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; . .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
MODES=(4 8 12 16 24 32)
M=${MODES[$SLURM_ARRAY_TASK_ID]}
mkdir -p results/modesweep
python src/arch_breadth.py --data data/resweep --Re 100 --modes "$M" \
    --epochs 4000 --periods 100 --seed-list 0 1 2 \
    --out "results/modesweep/modes${M}.json"
echo "modes=$M done"
