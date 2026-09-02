#!/usr/bin/env bash

#SBATCH --job-name=epmgdbf-tune
#SBATCH --output=logs/tune_epmgdbf_%j.out
#SBATCH --error=logs/tune_epmgdbf_%j.err
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    project_dir=$SLURM_SUBMIT_DIR
else
    project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
fi

cd "$project_dir"
mkdir -p logs

python_bin="$project_dir/.venv/bin/python3"
config_file=${EPMGDBF_CONFIG:-experiments/experiment_cpp_epmgdbf.json}
qc_code_config="codes/ldpc_savin_4_8_12_24_Zc54.json"
qc_code_pcm="codes/ldpc_savin_4_8_12_24_Zc54_pcm.alist"
if [[ ! -x "$python_bin" ]] || ! "$python_bin" --version >/dev/null 2>&1; then
    echo "Working Python interpreter not found: $python_bin" >&2
    echo "Create .venv and install requirements before submitting this job." >&2
    exit 1
fi
if ! command -v g++ >/dev/null 2>&1; then
    echo "C++ compiler not found: g++" >&2
    echo "Load the GCC module available on the server and resubmit the job." >&2
    exit 1
fi
if [[ ! -f "$config_file" ]]; then
    echo "Experiment config not found: $project_dir/$config_file" >&2
    exit 1
fi
if [[ ! -f "$qc_code_config" || ! -f "$qc_code_pcm" ]]; then
    echo "Generating the Savin (4,8) QC-LDPC code..."
    "$python_bin" ldpc_qc_matrix.py \
        -c ldpc_savin_4_8_12_24.txt \
        --Zc 54
fi

exec srun "$python_bin" tune_epmgdbf.py \
    -c "$config_file" \
    --workers "${SLURM_CPUS_PER_TASK:-128}" \
    "$@"
