#!/bin/bash -l
#SBATCH --job-name=cyldom
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --time=0-10:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64g
#SBATCH --output=logs/dom_%A_%a.out
#SBATCH --error=logs/dom_%A_%a.err
# Blockage and Mach are both ruled out. Remaining suspect is the streamwise
# domain: inlet proximity and the crude zero-gradient outflow. Vary upstream
# distance and downstream length at fixed D=24, ny_D=20, u=0.1.
# Usage: sbatch --array=0-3 cluster/cyl_domain.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; . .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
NX_LIST=(30 50 50 80)
XC_LIST=(8  8  16 20)
NX=${NX_LIST[$SLURM_ARRAY_TASK_ID]}; XC=${XC_LIST[$SLURM_ARRAY_TASK_ID]}
mkdir -p results/domain
python src/cylinder_lbm.py --D 24 --nx-D "$NX" --xc-D "$XC" --steps 132000 \
    --out "results/domain/nx${NX}_xc${XC}.json"
echo "nx_D=$NX xc_D=$XC done"
