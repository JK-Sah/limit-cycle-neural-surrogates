#!/bin/bash -l
#SBATCH --job-name=romwake
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48g
#SBATCH --output=logs/romwake_%j.out
#SBATCH --error=logs/romwake_%j.err
# Classical ROM baselines (DMD, operator inference) measured with the same
# delta as the neural models. Answers whether the period error is a property
# of learned dynamics or of reduced-order modelling of limit cycles generally.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; . .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
mkdir -p results/rom
python src/rom_wake.py --data data/resweep --modes 16 --periods 100 \
    --out results/rom/rom_wake.json
