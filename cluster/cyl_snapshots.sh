#!/bin/bash -l
#SBATCH --job-name=cylsnap
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --time=0-10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --output=logs/cylsnap_%j.out
#SBATCH --error=logs/cylsnap_%j.err

# Wake snapshots for latent surrogate training, D=32, Re=100.
# Shedding period is ~1948 lattice steps, so snap_every=39 gives ~50 samples
# per period. Sampling starts halfway through the run, ~45 periods in, which is
# well past the transient.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
. .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

mkdir -p results/cyl data
python src/cylinder_lbm.py \
    --D 32 \
    --steps 175000 \
    --snap-every 39 \
    --snap-stride 2 \
    --out results/cyl/D32_snaprun.json \
    --save-signal data/D32_signal.npz \
    --save-snaps data/D32_snaps.npz

echo "snapshots done on $(hostname)"
ls -lh data/D32_snaps.npz
