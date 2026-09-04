#!/bin/bash -l
#SBATCH --job-name=rerun
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32g
#SBATCH --output=logs/rerun_%A_%a.out
#SBATCH --error=logs/rerun_%A_%a.err
# Re-measure the two tables that still predate the period-estimator correction.
#
# Section 5 (parametric ROM repair) and section 6 (parametric wake) were both
# measured with the old Hilbert phase-regression estimator, which carries a
# 1.2e-5 bias on harmonic-rich signals. Both sit at 4e-4 or above, one to three
# orders clear of that bias, so the draft states them as unaffected -- and that
# is almost certainly right. But "almost certainly right" is a caveat a referee
# can pull on, and section 3's DMD column showed what happens when a small
# quantity is measured against an estimator-supplied reference: delta moved by
# up to 9x while the eigenvalue did not move at all.
#
# These are cheap next to what they buy. Task 0 is closed-form and takes
# minutes; task 1 retrains the parametric wake. Removing the caveat entirely is
# worth more than defending it.
#
# Usage: sbatch --array=0-4 cluster/rerun_estimator.sh
#   0    section 5, parametric ROM (closed form)
#   1-4  section 6, parametric wake, one seed each
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; . .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
export PYTHONPATH=src
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    mkdir -p results/rom
    python src/rom_parametric.py --data data/resweep --modes 16 --periods 100 \
        --out results/rom/rom_parametric_v2.json
else
    S=$((SLURM_ARRAY_TASK_ID - 1))
    mkdir -p results/parwake_v2
    python src/param_wake.py --data data/resweep --modes 16 --epochs 6000 \
        --periods 100 --seed-list "$S" \
        --free-widths 128 200 288 --omega-modes williamson poly4 \
        --out "results/parwake_v2/seed${S}.json"
fi
