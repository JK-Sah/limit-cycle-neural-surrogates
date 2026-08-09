#!/bin/bash -l
#SBATCH --job-name=modesw
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --time=0-02:30:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16g
#SBATCH --output=logs/modesw_%A_%a.out
#SBATCH --error=logs/modesw_%A_%a.err
# Does delta track how dynamically closed the retained POD subspace is?
# DMD gives 2e-8 at 16 modes and 4.58e-4 at 8, so the classical-versus-learned
# ordering depends on truncation and has to be pinned down.
#
# One (mode count, seed) pair per task: 6 x 3 = 18 short tasks. The earlier
# version asked for 16h/8cpu/48g in 6 tasks and sat unscheduled for two days on
# a partition with 182 jobs pending. Observed runtime is ~15 min per task.
# Usage: sbatch --array=0-17 cluster/mode_sweep.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; . .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
MODES=(4 8 12 16 24 32)
M=${MODES[$((SLURM_ARRAY_TASK_ID / 3))]}
S=$((SLURM_ARRAY_TASK_ID % 3))
mkdir -p results/modesweep
python src/arch_breadth.py --data data/resweep --Re 100 --modes "$M" \
    --epochs 4000 --periods 100 --seed-list "$S" \
    --out "results/modesweep/modes${M}_seed${S}.json"
echo "modes=$M seed=$S done"
