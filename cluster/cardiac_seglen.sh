#!/bin/bash -l
#SBATCH --job-name=apseg
#SBATCH --account=flowlab
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=24g
#SBATCH --time=06:00:00
#SBATCH --output=logs/apseg_%A_%a.out
#SBATCH --error=logs/apseg_%A_%a.err
#
# The segment-length dose-response on a non-fluid limit cycle: a residual FNO
# on the two-channel (u, v) Aliev-Panfilov field, 128 x 128, with exactly the
# protocol of cluster/seglen_sweep.sh (A100, float64, seeded initialisation,
# checkpointed backward, 3000 iterations of 8 segments, 100-period rollout,
# delta from POD coefficient 1). K is set from the measured period so the
# fractions of a period match the Navier-Stokes arms: 0.25, 0.5, 0.625, 0.75,
# 1.0 and 1.5 T. Five seeds each, task = 5 * config + seed.
#
# Refuses to run unless the training record passed the split-half test: a
# delta on a record whose single-period fit is ill-posed would measure nothing.
#
# Usage: sbatch -p debug,tier3 --qos=qos_tier3 --array=0-29 \
#            --dependency=afterok:<cardiac_data task 0> cluster/cardiac_seglen.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; . .venv-gpu/bin/activate
export PYTHONPATH=src PYTHONUNBUFFERED=1
RES=${RES:-results/seglen}
EPOCHS=${EPOCHS:-3000}
PERIODS=${PERIODS:-100}
mkdir -p "$RES/coefs" logs
META=data/cardiac/ap_main.json
read -r REGIME TPHYS TSNAP < <(python -c "import json;j=json.load(open('$META'));print(j['regime'],j['T'],j['T_snap'])")
[ "$REGIME" = "periodic" ] || { echo "training record is '$REGIME', not periodic; refusing"; exit 3; }

i=$SLURM_ARRAY_TASK_ID
SEED=$((i % 5)); C=$((i / 5))
FR=(0.25 0.5 0.625 0.75 1.0 1.5)
K=$(python -c "print(max(2, round(${FR[$C]} * $TSNAP)))")
TAG="cardiac-fno-K${K}-b8_s${SEED}"
OUT="$RES/${TAG}.json"
[ -f "$OUT" ] && { echo "exists: $OUT"; exit 0; }

python src/fno_kolm.py --data data/cardiac/ap_main_snaps.npz --period "$TPHYS" \
    --modes-pod 8 --width 32 --fmodes 16 16 --depth 4 \
    --epochs "$EPOCHS" --seg-len "$K" --batch 8 --periods "$PERIODS" \
    --variant residual --noise 0.0 --device cuda --seed-init --ckpt \
    --seed-list "$SEED" --save-coefs "$RES/coefs/${TAG}.npz" --out "$OUT"
echo "done $TAG (K/T = $(python -c "print(round($K/$TSNAP,3))")) on $(hostname)"
