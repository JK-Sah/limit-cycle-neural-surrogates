#!/bin/bash -l
#SBATCH --job-name=cyllat
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16g
#SBATCH --output=logs/cyllat_%j.out
#SBATCH --error=logs/cyllat_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; . .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
python src/cyl_latent.py --snaps data/D16_snaps.npz --modes 8 \
    --epochs 4000 --periods 200 --seeds 5 --free-widths 96 168 256 \
    --out results/cyl_latent_D16.json
