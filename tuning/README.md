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
