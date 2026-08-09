#!/bin/bash -l
#SBATCH --job-name=archbr
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48g
#SBATCH --output=logs/arch_%A_%a.out
#SBATCH --error=logs/arch_%A_%a.err
# Does the period error survive a change of architecture, and is it the model
# class or the fitting procedure that loses the frequency?
# One seed per array task. Usage: sbatch --array=0-4 cluster/arch_job.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; . .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
mkdir -p results/arch
python src/arch_breadth.py --data data/resweep --Re 100 --modes 8 \
    --epochs 4000 --periods 100 --seed-list "$SLURM_ARRAY_TASK_ID" \
    --out "results/arch/seed${SLURM_ARRAY_TASK_ID}.json"
