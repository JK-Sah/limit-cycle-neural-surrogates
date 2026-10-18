#!/bin/bash -l
#SBATCH --job-name=seglen
#SBATCH --account=flowlab
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16g
#SBATCH --time=06:00:00
#SBATCH --output=logs/seglen_%A_%a.out
#SBATCH --error=logs/seglen_%A_%a.err
#
# Segment length as a dose-response, with the controls a referee will ask for.
#
# The paper's lead result rests on two points per system: 25 vs 40 steps on
# Navier-Stokes (3 seeds each) and 25 vs 32 on the wake (5 vs 7). Under the
# one-sided median permutation test the paper itself uses to reject the phase
# penalty (p = 0.18), those comparisons give p = 0.10 and p = 0.19. Two
# alternative readings are also open with two points:
#
#   1. "Longer is better", smoothly. Phase error over a K-step segment grows
#      like K, so the loss pressure on delta grows like K^2 with nothing special
#      at one period. Unrolled-training studies already report that longer
#      horizons help. Only a sweep through and past one period separates a knee
#      at K = T from a smooth power law.
#   2. More data per step. Each "epoch" is one batch of 8 segments, so a 40-step
#      objective sees 1.6x the transitions of a 25-step one. The two batch-
#      matched arms (K=25 at batch 13, K=40 at batch 5) hold transitions per
#      step fixed while changing K.
#
# The U-Net arms ask whether the effect belongs to the objective or to
# spectral convolutions. The wake arms repeat the sweep on the second system.
#
# Everything runs on A100s in float64 with the weight initialisation seeded
# (--seed-init), so the sweep is self-contained and re-runnable; the earlier
# CPU rows are an independent replication, not part of these arms. --ckpt
# recomputes activations per step (same arithmetic) so K = 80 fits in 40 GB.
# Rolled-out POD coefficients are saved so delta can be re-measured on a
# ten-period window afterwards.
#
# Task layout, 5 seeds per configuration (task = 5 * config + seed):
#   cfg  0-9   NS  FNO   K = 10 20 25 30 35 40 45 50 60 80, batch 8
#   cfg 10     NS  FNO   K = 25, batch 13   (transitions matched to K = 40)
#   cfg 11     NS  FNO   K = 40, batch 5    (transitions matched to K = 25)
#   cfg 12-14  NS  U-Net K = 25 40 60, batch 8 (60 not submitted: ~5 h a seed
#              in float64, and 25 vs 40 is the comparison the claim needs)
#   cfg 15-20  wake FNO  K = 16 24 32 40 48 64, batch 8
# T is 40.02 snapshots on NS and 32.15 on the wake.
#
# Measured on an A100 (float64): NS FNO ~1 min of training per unit of K,
# so K=40 is ~40 min and K=80 ~75 min; wake FNO ~80 s per unit of K; the U-Net
# ~3.3 h at K=40. About 90 GPU-hours in all.
#
# Usage, most decisive arms first:
#   sbatch -p debug,tier3 --qos=qos_tier3 --array=0-59   cluster/seglen_sweep.sh
#   sbatch -p debug,tier3 --qos=qos_tier3 --array=75-104 cluster/seglen_sweep.sh
#   sbatch -p debug,tier3 --qos=qos_tier3 --array=60-69  cluster/seglen_sweep.sh
# Smoke test into a scratch directory, never results/seglen (the skip-if-exists
# guard would then keep the toy runs):
#   sbatch -p debug --qos=qos_tier3 --array=0,55,60,75 \
#       --export=ALL,EPOCHS=5,PERIODS=3,RES=scratch/seglen_smoke cluster/seglen_sweep.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; . .venv-gpu/bin/activate
export PYTHONPATH=src PYTHONUNBUFFERED=1
RES=${RES:-results/seglen}
EPOCHS=${EPOCHS:-3000}
PERIODS=${PERIODS:-100}
mkdir -p "$RES/coefs" logs

i=$SLURM_ARRAY_TASK_ID
SEED=$((i % 5))
C=$((i / 5))
NSK=(10 20 25 30 35 40 45 50 60 80)
WKK=(16 24 32 40 48 64)
UNK=(25 40 60)

NS="src/fno_kolm.py --data data/kolm/nu1.7e-3_snaps.npz --modes-pod 8 --width 32 --fmodes 16 16 --depth 4"
WK="src/fno.py --data data/resweep --Re 100 --modes-pod 8 --width 32 --fmodes 16 12 --depth 4"
B=8
if   [ $C -le 9 ];  then K=${NSK[$C]};        SCRIPT=$NS; ARCH=fno;  SYS=ns
elif [ $C -eq 10 ]; then K=25; B=13;          SCRIPT=$NS; ARCH=fno;  SYS=ns
elif [ $C -eq 11 ]; then K=40; B=5;           SCRIPT=$NS; ARCH=fno;  SYS=ns
elif [ $C -le 14 ]; then K=${UNK[$((C-12))]}; SCRIPT="$NS --arch unet"; ARCH=unet; SYS=ns
else                     K=${WKK[$((C-15))]}; SCRIPT=$WK; ARCH=fno;  SYS=wake
fi
TAG="${SYS}-${ARCH}-K${K}-b${B}_s${SEED}"
OUT="$RES/${TAG}.json"
[ -f "$OUT" ] && { echo "exists: $OUT"; exit 0; }

python $SCRIPT --epochs "$EPOCHS" --seg-len "$K" --batch "$B" --periods "$PERIODS" \
    --variant residual --noise 0.0 --device cuda --seed-init --ckpt \
    --seed-list "$SEED" --save-coefs "$RES/coefs/${TAG}.npz" \
    --out "$OUT"
echo "done $TAG on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
