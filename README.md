# LDPC simulator in the AWGN channel

## Usage
This module utilizes a parallel simulator submodule. To run the experiment, specify the `experiment` section in the experiment JSON file.
Required parameters are:
- `src_dir` source dir containing code description
- `code` JSON filename specifying code
- `modulation` in text format. In accordance with simulation submodue implementations, supported values are `BPSK `, `QPSK`, `PAM-4`, and `QAM-16`.
- `algorithm` is the decoding algorithm. Supported values are registered in `ldpc_py/decoder_factory.py`.
- `llr_scale` is applicable to the min-sum decoding algorithms.
- `n_iterations` is the number of decoding iterations

For the erasure-add probabilistic momentum gradient descent bit-flipping decoder,
use the algorithm name `erasure add probabilistic momentum gradient descent
bit-flipping`. Its decoder parameters are:

- `delta` defines the energy interval for ordinary bit flipping.
- `delta_e` defines the upper energy interval for introducing new erasures. It
	must be greater than `delta` to create a non-empty erasure interval.
- `alpha` weights the channel LLR contribution in the energy function.
- `p` is the probability of applying a selected flip or erasure update.
- `rho` and `L` define the momentum/history penalty; `len(rho)` must equal `L`.
- `zeros_in_init` controls initialization. When `false`, the initial hard
	decision uses only `-1` and `+1`. When `true`, LLR values between `-0.5` and
	`0.5` are initialized as erasures (`0`).

During each iteration, existing erasures are processed first. A bit is restored
from a degree-one check when possible. Otherwise, checks containing exactly one
erasure provide votes, and the majority value is used. If no value can be
determined, the bit is restored to its value at the beginning of the iteration.
Then ordinary candidates are flipped and a separate energy interval introduces
new erasures. New erasures are kept for the next iteration. Any erasures that
remain after the final iteration are replaced with their values from the start
of that final iteration before being written to `llr_out`.

The JSON file specifying code contains the following parameters:
- `pcm` is a parity check matrix (alist format)
- `generator` is a text file containing the generator matrix (in space-separated format, see `numpy.savetxt`). If the generator matrix is not specified, then simulations will use zero-codewords 
- `punc_idx` a list of punctured indices in the string format, like `0:15`. If this parameter is absent, no puncturing is assumed.
- `inf_bits` specifies indices of information bits. If this parameter is missing, then the output bit error rate will be evaluated using a whole codeword.

See [example.sh](example.sh) for more details.

## Interactive decision-energy explorer

The standalone Dash application in
[`visualisation/error_by_energy_visualisation`](visualisation/error_by_energy_visualisation)
provides reproducible, step-by-step FTGDBF and PMGDBF inspection. It displays
decision-energy distributions, BER/FER, immutable state and parameter history,
and supports session/export/comparison workflows. See the module
[`README.md`](visualisation/error_by_energy_visualisation/README.md) for local
and server launch commands.

Example EAFPMGDBF configuration:

```json
"algorithm": "erasure add probabilistic momentum gradient descent bit-flipping",
"decoder_params": {
	"delta": 1.0,
	"delta_e": 1.3,
	"alpha": 1.7,
	"p": 0.9,
	"zeros_in_init": false,
	"rho": [2, 2, 2, 2, 2, 1, 1],
	"L": 7
}
```

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

## E-GDBF V1 hyperparameter search

For the C++ E-GDBF V1 decoder, submit the existing tuner runner on the server:

```console
sbatch tune_egdbf.sh
```

The default staged search screens 2,048 combinations of `alpha`, `delta`, and
all seven `rho` entries on 2,000 common channel frames each. It then generates
coordinate neighbors of the best 24 sets at half-step resolution and evaluates
each on 5,000 fresh common frames. The best eight are compared on another
common sample. The final sample
grows until the best set has at least 500 frame errors or reaches the
10,000,000-frame cap. The selected parameters are then checked on a separate
sample, also aiming for 500 errors. Profiles include both monotone and
unrestricted `rho` vectors and the `L=0` case. Each stage writes results to
`params_cpp_egdbf.jsonl` and the current best to `params_cpp_egdbf.txt`.
Use `--mode grid` for the earlier Cartesian two-level-profile search.

## EPMGDBF hyperparameter search

Search the scalar EPMGDBF parameters at SNR 0.5 dB:

```console
python3 tune_epmgdbf.py
```

Every time a better set is found, it is printed and saved to `params.txt` in
the repository root. By default, every parameter set is evaluated at SNR 0.5
dB until 10 frame errors are collected. The upper limit is 10,000,000 frames.
All candidates use the same random seeds. Use `--max-errors`, `--trials`,
`--workers`, and the parameter-list options shown by `--help` to control the
search. For example, a quick smoke test is:

```console
python3 tune_epmgdbf.py --trials 10 --workers 1 --max-configs 1
```

Submit the full search to Slurm with 64 CPUs:

```console
sbatch tune_epmgdbf.sh
```

Additional search arguments are forwarded to Python, for example:

```console
sbatch tune_epmgdbf.sh --trials 10000 --max-configs 10
```

## Tools
### 5G LDPC constructor
See [ldpc_5g.py](ldpc_5g.py) script.

## Limitations
This software was tested on Ubuntu Linux and MacOS
