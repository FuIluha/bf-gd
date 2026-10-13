#!/usr/bin/env bash

#SBATCH --job-name=mpgd-tune
#SBATCH --output=logs/tune_mpgd_%j.out
#SBATCH --error=logs/tune_mpgd_%j.err
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    project_dir=$SLURM_SUBMIT_DIR
else
    project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
fi
cd "$project_dir"
mkdir -p logs
python_bin="$project_dir/.venv/bin/python3"
if [[ ! -x "$python_bin" ]] || ! "$python_bin" --version >/dev/null 2>&1; then
    echo "Working Python interpreter not found: $python_bin" >&2
    exit 1
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec srun "$python_bin" -m tuning.tune_mpgd \
    --workers "${SLURM_CPUS_PER_TASK:-128}" "$@"
