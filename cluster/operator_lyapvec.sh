#!/bin/bash -l
#SBATCH --job-name=lyapvec
#SBATCH --account=flowlab
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32g
#SBATCH --time=06:00:00
#SBATCH --output=logs/lyapvec_%A_%a.out
#SBATCH --error=logs/lyapvec_%A_%a.err
#
# Where the weakest transverse contraction of each trained operator lives:
# inside or outside the subspace the training data span, and at which Fourier
# wavenumbers (src/operator_lyapvec.py). ONLY selects the arms; finished models
# are skipped.
#
# Usage: sbatch -p debug,tier3 --qos=qos_tier3 --export=ALL,ONLY=K25-b8,K24-b8 cluster/operator_lyapvec.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; . .venv-gpu/bin/activate
export PYTHONPATH=src PYTHONUNBUFFERED=1
mkdir -p results/lyapvec logs
python src/operator_lyapvec.py --only "${ONLY:-K25-b8,K24-b8}"
