#!/usr/bin/env bash
set -euo pipefail
test -n "$MOBILE_CDNET_DATA_ROOT"
test -n "$MOBILE_CDNET_OUTPUT_ROOT"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"
python tools/test.py --file_root SYSU --savedir "$MOBILE_CDNET_OUTPUT_ROOT/results" --batch_size 1 --lr 5e-4 --max_steps 150000 --num_workers 2
