# LDPC simulator in the AWGN channel

## Usage
This module utilizes a parallel simulator submodule. To run the experiment, specify the `experiment` section in the experiment JSON file.
Required parameters are:
- `src_dir` source dir containing code description
- `code` JSON filename specifying code
- `modulation` in text format. In accordance with simulation submodue implementations, supported values are `BPSK `, `QPSK`, `PAM-4`, and `QAM-16`.
- `algorithm` is the decoding algorithm. Supported values are `sum_product`, `min_sum`, and `layered_min_sum`,
- `llr_scale` is applicable to the min-sum decoding algorithms.
- `n_iterations` is the number of decoding iterations

The JSON file specifying code contains the following parameters:
- `pcm` is a parity check matrix (alist format)
- `generator` is a text file containing the generator matrix (in space-separated format, see `numpy.savetxt`). If the generator matrix is not specified, then simulations will use zero-codewords 
- `punc_idx` a list of punctured indices in the string format, like `0:15`. If this parameter is absent, no puncturing is assumed.
- `inf_bits` specifies indices of information bits. If this parameter is missing, then the output bit error rate will be evaluated using a whole codeword.

See [example.sh](example.sh) for more details.

## One-command Slurm run with a standalone dashboard

Run the launcher directly on the login server (do not prefix it with `sbatch`):

```console
./run.sh experiments/experiment_pmgdbf.json
```

It starts or reuses the dashboard on port 8888 and submits the simulation to
Slurm. The command prints the fixed dashboard URL to open from the institute's
local network. The dashboard reads results from the shared `data` directory and
continues running when the Slurm job finishes.

The experiment JSON is a positional argument. To select another config or port:

```console
DASHBOARD_PORT=8890 ./run.sh experiments/experiment.json
```

## EPMGDBF hyperparameter search

Search the scalar EPMGDBF parameters at SNR 0.5 dB:

```console
python3 tune_epmgdbf.py
```

Every time a better set is found, it is printed and saved to `params.txt` in
the repository root. By default, every parameter set is evaluated on
10,000,000 frames at SNR 0.5 dB. All candidates use the same random seeds. Use
`--trials`, `--workers`, and the parameter-list options shown by `--help` to
control the search. For example, a quick smoke test is:

```console
python3 tune_epmgdbf.py --trials 10 --workers 1 --max-configs 1
```

## Tools
### 5G LDPC constructor
See [ldpc_5g.py](ldpc_5g.py) script.

## Limitations
This software was tested on Ubuntu Linux and MacOS
