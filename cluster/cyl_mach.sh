#!/bin/bash -l
#SBATCH --job-name=cylmach
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=48g
#SBATCH --output=logs/mach_%A_%a.out
#SBATCH --error=logs/mach_%A_%a.err
# Is the +2.9% Strouhal offset compressibility? Blockage was ruled out.
# D is scaled inversely with u so tau stays 0.572 and only Mach changes.
# Usage: sbatch --array=0-2 cluster/cyl_mach.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; . .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
U_LIST=(0.10 0.0667 0.05)
D_LIST=(24   36     48)
U=${U_LIST[$SLURM_ARRAY_TASK_ID]}; D=${D_LIST[$SLURM_ARRAY_TASK_ID]}
# hold the number of shed cycles fixed: T ~ D/(St*u)
STEPS=$(python -c "print(int(60*$D/(0.169*$U)))")
mkdir -p results/mach
python src/cylinder_lbm.py --D "$D" --u-in "$U" --steps "$STEPS" \
    --out results/mach/u${U}_D${D}.json
echo "u=$U D=$D steps=$STEPS done"
