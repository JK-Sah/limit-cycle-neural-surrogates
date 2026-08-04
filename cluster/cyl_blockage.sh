#!/bin/bash -l
#SBATCH --job-name=cylblk
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=24g
#SBATCH --output=logs/blk_%A_%a.out
#SBATCH --error=logs/blk_%A_%a.err
# Is the +2.9% Strouhal offset blockage? Fix D=24, open the domain out.
# Usage: sbatch --array=0-3 cluster/cyl_blockage.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; . .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
NYD_LIST=(20 30 40 60)
NYD=${NYD_LIST[$SLURM_ARRAY_TASK_ID]}
mkdir -p results/blockage
python src/cylinder_lbm.py --D 24 --ny-D "$NYD" --steps 132000 \
    --out results/blockage/nyD${NYD}.json
echo "ny_D=$NYD done"
