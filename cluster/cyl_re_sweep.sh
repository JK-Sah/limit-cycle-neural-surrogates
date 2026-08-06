#!/bin/bash -l
#SBATCH --job-name=cylre
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --time=0-10:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=48g
#SBATCH --output=logs/cylre_%A_%a.out
#SBATCH --error=logs/cylre_%A_%a.err
# Wake snapshots across Reynolds number, for the parametric operator.
# Strouhal varies 0.153 -> 0.202 over this range along the Williamson curve,
# so omega(Re) has an independent reference.
# Domain is the validated one: 20D upstream, 60D downstream (the 8D/22D box
# used earlier is truncated at both ends).
# Usage: sbatch --array=0-12 cluster/cyl_re_sweep.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; . .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
RE_LIST=(80 100 120 140 160 180 90 110 130 150 170 200 220)
RE=${RE_LIST[$SLURM_ARRAY_TASK_ID]}
mkdir -p results/resweep data/resweep
python src/cylinder_lbm.py \
    --D 24 --Re "$RE" --nx-D 80 --xc-D 20 --ny-D 20 \
    --steps 140000 --snap-every 45 --snap-stride 2 \
    --out "results/resweep/Re${RE}.json" \
    --save-signal "data/resweep/Re${RE}_signal.npz" \
    --save-snaps "data/resweep/Re${RE}_snaps.npz"
echo "Re=$RE done on $(hostname)"
