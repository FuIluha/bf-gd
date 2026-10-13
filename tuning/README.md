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

The L2 default grid has **20,625 combinations** around the previous winner
(alpha=0.875, eta_r=eta_v=l2_r=l2_v=max_value=1). It searches all six
parameters using a fixed grid.

| Parameter | CLI option | Values |
| --- | --- | --- |
| `alpha` | `--alphas` | 0.75–1.0, step 0.025 |
| `eta_r` | `--eta-rs` | 0.9, 0.95, 1, 1.05, 1.1 |
| `eta_v` | `--eta-vs` | 0.9, 0.95, 1, 1.05, 1.1 |
| `l2_r` | `--l2-rs` | 0.9, 0.95, 1, 1.05, 1.1 |
| `l2_v` | `--l2-vs` | 0.9, 0.95, 1, 1.05, 1.1 |
| `max_value` | `--max-values` | 0.9, 1, 1.1 |

Specify comma-separated values to replace any axis. Only the requested grid
is tested; no baseline outside the lists is silently added. The iteration
budget defaults to 50 complete CN/VN cycles and is set by `--iterations`;
the message bound is independently searched with `--max-values`. An explicit
`--max-values 1000000` tests a very wide range, but still uses a finite bound. Increasing the
bound can amplify the CN products; numerical failures are logged as invalid
and excluded from best-parameter selection.

Each result is printed and flushed to `params_mpgd_l2_snr0_fine50.jsonl`.
Improvements are atomically saved to `params_mpgd_l2_snr0_fine50.txt`, including
the regularization type (`l2`), all six parameters, complete codec settings,
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
`params_mpgd_l2_snr0_fine50.jsonl` for every completed candidate. The new default
paths preserve the earlier broad L2 sweep (`params_mpgd_l2_snr0_100.*`)
and entropy sweep results.
