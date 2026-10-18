#!/bin/bash -l
#SBATCH --job-name=opfloq
#SBATCH --account=flowlab
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=24g
#SBATCH --time=04:00:00
#SBATCH --output=logs/opfloq_%A_%a.out
#SBATCH --error=logs/opfloq_%A_%a.err
#
# Lyapunov exponents (Floquet multipliers per period) of every saved operator
# in results/seglen_v2, residual against direct. See src/operator_floquet.py.
# Each task takes every NTASKS-th model; finished models are skipped, so the
# same command picks up models added later (the direct arms 150-169).
# Validated on a Stuart-Landau map embedded with known decay rates: recovers
# 0, -0.3, -1, -1 as 0, -0.305, -0.995, -1.001.
#
# Usage: sbatch -p debug,tier3 --qos=qos_tier3 --array=0-13 cluster/operator_floquet.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; . .venv-gpu/bin/activate
export PYTHONPATH=src PYTHONUNBUFFERED=1
mkdir -p results/floquet_ops logs
python src/operator_floquet.py --task "$SLURM_ARRAY_TASK_ID" --ntasks "${NTASKS:-14}"
