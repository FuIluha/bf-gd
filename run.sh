#!/usr/bin/env bash

# Run this file directly on entropy1. It starts (or reuses) the standalone
# dashboard on the login node and then submits itself as the Slurm job.

#SBATCH --job-name=ldpc
#SBATCH --output=logs/ldpc_%j.out
#SBATCH --error=logs/ldpc_%j.err
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
config_file=${LDPC_CONFIG:-experiments/experiment_pmgdbf.json}
dashboard_port=${DASHBOARD_PORT:-8888}
python_bin="$project_dir/.venv/bin/python3"

prepare_python() {
    if [[ -x "$python_bin" ]] && "$python_bin" --version >/dev/null 2>&1; then
        return
    fi

    if command -v module >/dev/null 2>&1; then
        module load python/3.12.3 >/dev/null 2>&1 || true
    fi

    if [[ ! -x "$python_bin" ]] || ! "$python_bin" --version >/dev/null 2>&1; then
        echo "Working Python interpreter not found: $python_bin" >&2
        echo "Create .venv and install requirements before running this script." >&2
        exit 1
    fi
}

# This branch is executed by Slurm after the launcher submits this same file.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    cd "$project_dir"
    prepare_python
    exec srun "$python_bin" main.py -c "$config_file" --no-run-plots
fi

cd "$project_dir"
mkdir -p logs

if [[ ! -f "$config_file" ]]; then
    echo "Experiment config not found: $project_dir/$config_file" >&2
    exit 1
fi

prepare_python

dashboard_url="http://127.0.0.1:$dashboard_port"
dashboard_log="$project_dir/logs/dashboard.log"
dashboard_pid_file="$project_dir/logs/dashboard.pid"

if curl --fail --silent "$dashboard_url/_dash-layout" >/dev/null 2>&1; then
    echo "Dashboard is already running on port $dashboard_port."
else
    echo "Starting dashboard on port $dashboard_port..."
    nohup "$python_bin" plot_results.py \
        -c "$config_file" \
        --host 0.0.0.0 \
        --port "$dashboard_port" \
        >"$dashboard_log" 2>&1 </dev/null &
    dashboard_pid=$!
    printf '%s\n' "$dashboard_pid" >"$dashboard_pid_file"

    dashboard_ready=false
    for _ in {1..20}; do
        if curl --fail --silent "$dashboard_url/_dash-layout" >/dev/null 2>&1; then
            dashboard_ready=true
            break
        fi
        if ! kill -0 "$dashboard_pid" 2>/dev/null; then
            break
        fi
        sleep 0.5
    done

    if [[ "$dashboard_ready" != true ]]; then
        echo "Dashboard failed to start. Last log lines:" >&2
        tail -n 30 "$dashboard_log" >&2 || true
        exit 1
    fi
fi

server_ip=''
if [[ -n "${SSH_CONNECTION:-}" ]]; then
    read -r _ _ server_ip _ <<<"$SSH_CONNECTION"
fi
if [[ -z "$server_ip" ]]; then
    server_ip=$(hostname -I 2>/dev/null | awk '{print $1}' || true)
fi
server_ip=${server_ip:-127.0.0.1}

job_id=$(sbatch \
    --parsable \
    "$project_dir/run.sh")

echo "Slurm job submitted: $job_id"
echo "Dashboard: http://$server_ip:$dashboard_port"
echo "Dashboard log: $dashboard_log"
