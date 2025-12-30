#!/bin/bash
pip install -r requirements.txt
python3 ldpc_5g.py --k=120 --rate=0.2 --BG=2
python3 main.py -c experiment.json
