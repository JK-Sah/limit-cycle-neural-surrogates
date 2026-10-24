#!/bin/bash -l
#SBATCH --job-name=seglen2
#SBATCH --account=flowlab
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=24g
#SBATCH --time=08:00:00
#SBATCH --output=logs/seglen2_%A_%a.out
#SBATCH --error=logs/seglen2_%A_%a.err
#
# Every neural-operator arm in the study, on one protocol, with delta measured
# past the operator's own transient and split-half tested.
#
# Why v2. The first sweep (cluster/seglen_sweep.sh, results/seglen) fitted delta
# on periods 50-100 of a 100-period rollout, as every earlier operator row did.
# Its saved rollouts showed that fit is not settled for field operators: the two
# quarters of the window disagreed by a large fraction of delta, sometimes in sign
# (on the v2 rollouts: median 25% for residual operators, 0.0% for direct),
# and 0 of 35 runs passed the split-half threshold the paper applies to data.
# A pure amplitude transient does not do that (checked: late_delta recovers a
# known delta to 1e-8 through one), so the learned orbits carry phase modulation
# that 100 periods do not average out. v2 rolls each model out 400 periods,
# keeps the old delta (second half of the first 100) for continuity, adds
# delta on periods 200-400 with its split-half instability, and saves the
# weights so no delta ever needs a retrain to re-measure.
#
# It also re-runs the direct-parameterisation and noise-injection arms, because
# the paper's operator claims rest on them too, and the zero-initialised U-Net
# (the default-initialised one sat on a loss plateau for most of training).
#
# Configurations, most decisive first (task = 5 * config + seed):
#    0 NS K25           1 NS K40           2 NS K25 batch 13  3 NS K40 batch 5
#    4 cardiac 0.625T   5 cardiac 1.0T     6 wake K24         7 wake K32
#    8 NS K10   9 NS K20  10 NS K30  11 NS K35  12 NS K45  13 NS K50
#   14 NS K60  15 NS K80
#   16 cardiac 0.25T  17 cardiac 0.5T  18 cardiac 0.75T  19 cardiac 1.5T
#   20 wake K16  21 wake K40  22 wake K48  23 wake K64
#   24 NS direct K25   25 NS noise K25   26 wake direct K24   27 wake noise K24
#   28 NS U-Net K25 (zero-init)          29 NS U-Net K40 (zero-init)
#   Added 2026-10-12 after the first 150 showed direct operators keep a well-posed
#   cycle (10/10) where residual ones mostly do not, to make parameterisation x
#   segment length a full 2 x 2 on every system and test it outside fluids:
#   30 cardiac direct 0.625T   31 cardiac direct 1.0T
#   32 NS direct K40           33 wake direct K32
#   Added 2026-10-24 to test the Discussion's account of the residual
#   multipliers: noise injection at 1e-3 left them at 0.96, which the account
#   must explain. These raise the amplitude tenfold and thirtyfold:
#   34 NS noise 1e-2 K25   35 NS noise 3e-2 K25   36 wake noise 1e-2 K24
#   37 wake noise 3e-2 K24
# Residual parameterisation unless stated; noise = injection at 1e-3.
# T = 40.02 snapshots (NS), 32.15 (wake), measured (cardiac, ~39.7).
#
# Usage:
#   sbatch -p debug,tier3 --qos=qos_tier3 --array=0-149 cluster/seglen_v2.sh
#   sbatch -p debug,tier3 --qos=qos_tier3 --array=150-169 cluster/seglen_v2.sh
#   sbatch -p debug,tier3 --qos=qos_tier3 --array=170-189 cluster/seglen_v2.sh
# Smoke test into scratch (never results/seglen_v2), on CPU, one task per
# system and variant: scratch/v2_smoke.sh runs tasks 0 20 30 120 135 140 with
# DEVICE=cpu EPOCHS=2 PERIODS=2 PLONG=4 RES=scratch/v2_smoke.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; . .venv-gpu/bin/activate
export PYTHONPATH=src PYTHONUNBUFFERED=1
RES=${RES:-results/seglen_v2}
EPOCHS=${EPOCHS:-3000}
PERIODS=${PERIODS:-100}
PLONG=${PLONG:-400}
DEVICE=${DEVICE:-cuda}
mkdir -p "$RES/coefs" "$RES/models" logs

i=$SLURM_ARRAY_TASK_ID
SEED=$((i % 5)); C=$((i / 5))
NS="src/fno_kolm.py --data data/kolm/nu1.7e-3_snaps.npz --modes-pod 8 --width 32 --fmodes 16 16 --depth 4"
WK="src/fno.py --data data/resweep --Re 100 --modes-pod 8 --width 32 --fmodes 16 12 --depth 4"
read -r TC TCS < <(python -c "import json;j=json.load(open('data/cardiac/ap_main.json'));print(j['T'],j['T_snap'])")
CD="src/fno_kolm.py --data data/cardiac/ap_main_snaps.npz --period $TC --modes-pod 8 --width 32 --fmodes 16 16 --depth 4"
ck(){ python -c "print(max(2, round($1 * $TCS)))"; }

B=8; VAR=residual; NOISE=0.0; ARCH=fno; EXTRA=""
case $C in
  0) SYS=ns; K=25 ;;            1) SYS=ns; K=40 ;;
  2) SYS=ns; K=25; B=13 ;;      3) SYS=ns; K=40; B=5 ;;
  4) SYS=cardiac; K=$(ck 0.625) ;;  5) SYS=cardiac; K=$(ck 1.0) ;;
  6) SYS=wake; K=24 ;;          7) SYS=wake; K=32 ;;
  8) SYS=ns; K=10 ;;  9) SYS=ns; K=20 ;;  10) SYS=ns; K=30 ;;  11) SYS=ns; K=35 ;;
  12) SYS=ns; K=45 ;; 13) SYS=ns; K=50 ;; 14) SYS=ns; K=60 ;;  15) SYS=ns; K=80 ;;
  16) SYS=cardiac; K=$(ck 0.25) ;;  17) SYS=cardiac; K=$(ck 0.5) ;;
  18) SYS=cardiac; K=$(ck 0.75) ;;  19) SYS=cardiac; K=$(ck 1.5) ;;
  20) SYS=wake; K=16 ;; 21) SYS=wake; K=40 ;; 22) SYS=wake; K=48 ;; 23) SYS=wake; K=64 ;;
  24) SYS=ns; K=25; VAR=direct; ARCH=fnodir ;;
  25) SYS=ns; K=25; NOISE=1e-3; ARCH=fnonz ;;
  26) SYS=wake; K=24; VAR=direct; ARCH=fnodir ;;
  27) SYS=wake; K=24; NOISE=1e-3; ARCH=fnonz ;;
  28) SYS=ns; K=25; ARCH=unetz; EXTRA="--arch unet --zero-init-out" ;;
  29) SYS=ns; K=40; ARCH=unetz; EXTRA="--arch unet --zero-init-out" ;;
  30) SYS=cardiac; K=$(ck 0.625); VAR=direct; ARCH=fnodir ;;
  31) SYS=cardiac; K=$(ck 1.0); VAR=direct; ARCH=fnodir ;;
  32) SYS=ns; K=40; VAR=direct; ARCH=fnodir ;;
  33) SYS=wake; K=32; VAR=direct; ARCH=fnodir ;;
  34) SYS=ns; K=25; NOISE=1e-2; ARCH=fnonz10 ;;
  35) SYS=ns; K=25; NOISE=3e-2; ARCH=fnonz30 ;;
  36) SYS=wake; K=24; NOISE=1e-2; ARCH=fnonz10 ;;
  37) SYS=wake; K=24; NOISE=3e-2; ARCH=fnonz30 ;;
  *) echo "no config $C"; exit 2 ;;
esac
case $SYS in ns) SCRIPT=$NS ;; wake) SCRIPT=$WK ;; cardiac) SCRIPT=$CD ;; esac
TAG="${SYS}-${ARCH}-K${K}-b${B}_s${SEED}"
OUT="$RES/${TAG}.json"
[ -f "$OUT" ] && { echo "exists: $OUT"; exit 0; }

python $SCRIPT $EXTRA --epochs "$EPOCHS" --seg-len "$K" --batch "$B" \
    --periods "$PERIODS" --periods-long "$PLONG" \
    --variant "$VAR" --noise "$NOISE" --device "$DEVICE" --seed-init --ckpt \
    --seed-list "$SEED" --save-coefs "$RES/coefs/${TAG}.npz" \
    --save-model "$RES/models/${TAG}.pt" --out "$OUT"
echo "done $TAG on $(hostname) ($DEVICE)"
