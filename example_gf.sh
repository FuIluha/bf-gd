#!/bin/bash
pip install -r requirements.txt
python3 ldpc_mackay.py --code=all
python3 main.py -c experiments/experiment_gf_n96.json
python3 main.py -c experiments/experiment_gf_n204.json
python3 main.py -c experiments/experiment_gf_n504.json
python3 main.py -c experiments/experiment_gf_n1008.json
