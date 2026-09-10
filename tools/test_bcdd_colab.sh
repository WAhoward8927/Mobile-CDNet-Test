#!/usr/bin/env bash
set -euo pipefail
cd /content/Mobile-CDNet
export PYTHONPATH=/content/Mobile-CDNet
export MOBILE_CDNET_DATA_ROOT=/content/BCDD
python tools/test.py --file_root BCDD --savedir /content/drive/MyDrive/Mobile-CDNet/outputs/BCDD/results --batch_size 1 --lr 5e-4 --max_steps 76000 --num_workers 2
