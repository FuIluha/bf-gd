#!/bin/bash
pip install -r requirements.txt
python3 ldpc_mackay.py --code=all
python3 main.py -c experiments/experiment_gf.json
