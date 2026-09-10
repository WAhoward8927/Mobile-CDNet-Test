#!/usr/bin/env bash
set -euo pipefail
cd /content/Mobile-CDNet
export PYTHONPATH=/content/Mobile-CDNet
export MOBILE_CDNET_DATA_ROOT=/content/BCDD_256
python tools/train.py --file_root BCDD --savedir /content/drive/MyDrive/Mobile-CDNet/outputs/BCDD/results --batch_size 16 --lr 5e-4 --max_steps 76000 --lr_mode step --step_loss 100 --num_workers 2
