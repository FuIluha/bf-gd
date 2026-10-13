# Hyperparameter tuning

This directory contains the standalone grid-search entry points and their
Slurm launchers. Run commands from the repository root so relative experiment
and virtual-environment paths remain consistent.

For a local run:

```console
python3 -m tuning.tune_epmgdbf --help
```

For a Slurm run:

```console
sbatch tuning/tune_epmgdbf.sh
```

Each `tune_*.sh` launcher invokes the matching `tune_*.py` script. Decoder
implementations and experiment configurations remain in their existing
top-level modules and `experiments/` directory.

## MPGD

The tuner uses `cpp mpgd` and the existing experiment runner. It takes the
code, channel, modulation and LLR dtype from `--config` (by default
`experiments/experiment_cpp_gdms.json`), replacing its decoder in memory.
It does not modify that config or ordinary simulation result files.

```console
python3 -m tuning.tune_mpgd --dry-run
mkdir -p logs
sbatch tuning/tune_mpgd.sh --snr 0.0 --iterations 50 --max-errors 50
```

The launcher requests 128 CPUs. Every candidate collects 50 actual frame
errors at SNR 0.0 dB, with a safety cap of 100,000,000 frames. Wrong codewords
after early stopping are counted as errors. All candidates receive the same
channel realizations, controlled by `--seed`; scheduling does not change them.
There is no refinement or adaptive candidate elimination.

The refined default grid has **45,056 combinations** and independently
searches all five decoder parameters:

| Parameter | CLI option | Values |
| --- | --- | --- |
| `alpha` | `--alphas` | 0.25 to 1.5 inclusive, step 0.125 |
| `eta_r` | `--eta-rs` | 0.7, 1, 1.25, 1.5, 1.75, 2, 2.5, 3 |
| `eta_v` | `--eta-vs` | 0.4, 0.5, 0.6, 0.7, 0.8, 1, 1.5, 2 |
| `tau_r` | `--tau-rs` | 0, 0.003, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1 |
| `tau_v` | `--tau-vs` | 0.1 to 0.45 inclusive, step 0.05 |

Specify comma-separated values to replace any axis. Only the requested grid
is tested; no baseline outside the lists is silently added. The iteration
budget defaults to 50 complete CN/VN cycles and is set by `--iterations`;
clipping epsilon is a numerical constant.

Each result is printed and flushed to `params_mpgd_snr0.jsonl`. Improvements are
atomically saved to `params_mpgd_snr0.txt`, including the complete codec settings,
observed trial/error counts, seed and channel-output type. Ranking uses FER,
then BER, then mean iterations; exact ties use grid order, independently of
worker scheduling. Results that reach the trial cap are marked
explicitly. `--output` changes the best-result path and the adjacent JSONL path.

Local smoke test (not the full experiment):

```console
python3 -m tuning.tune_mpgd --workers 2 --max-configs 4 \
    --trials 8 --max-errors 3 --iterations 5 --output /tmp/mpgd_smoke.txt
```

Monitor the server search with `tail -f logs/tune_mpgd_<jobid>.out` or inspect
`params_mpgd_snr0.jsonl` for every completed candidate. The new default paths
keep the earlier 0.8 dB sweep in `params_mpgd.txt` / `params_mpgd.jsonl` intact.
