"""
Main function
"""
# Numpy is required for single_run function only
import numpy as np

from lbc_encoder.lbc_encoder import lib_compile as lbc_compile
from ldpc_soft_py.bin_ldpc_soft import lib_compile as ldpc_compile
from ldpc_py.cpp_bin_ldpc_soft_gdbf import lib_compile as soft_gdbf_compile
from simulator_awgn_python.channel import lib_compile as chan_compile
from ldpc_experiment import LdpcExperimentInstance, LdpcExperimentSettings, LdpcDataEntry
from simulator_awgn_python.tools import load_json

def compile_all():
    """
    Compile all libraries involved
    """
    chan_compile()
    ldpc_compile()  # LDPC codec
    soft_gdbf_compile()  # C++ soft GDBF decoder
    lbc_compile()  # Low-complexity encoder


def single_run(config_filename='experiment.json', snr_db=-8.0):
    """
    This function instantiates the experiment and performs a single test.
    When creating a new experiment, check that this run is succesful
    """
    # Works only with single experiment (do not specify lists of parameters)
    config = load_json(config_filename)
    # Create experiment settings
    exp_settings = LdpcExperimentSettings(**config['experiment'])
    # Create experiment instance
    exp_instance = LdpcExperimentInstance(exp_settings)
    # Perform single run and print the output
    data = exp_instance.run(snr_db, np.random.default_rng(seed=1))
    print(data)


if __name__ == '__main__':
    from simulator_awgn_python.simulator import run_all_experiments
    # Usage: python3 main.py --config=experiment.json
    # Default file is experiment.json
    # To create the proposed code to simulate, run the following command:
    # python3 ldpc_5g.py --k=120 --rate=0.2 --BG=2

    compile_all()
    # Debug new experiments with a single_run function
    # single_run()
    run_all_experiments(LdpcExperimentSettings, LdpcExperimentInstance, LdpcDataEntry)
