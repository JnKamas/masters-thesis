#!/bin/bash
set -e

#!/bin/bash
set -e

run_until_success () {
  while true; do
    python masters-thesis/train.py "$@" && break
    echo "GPU0 busy, retrying in 5s..."
    sleep 5
  done
}

run_until_success -bb resnet34 -iw 516 -ih 386 -b12 -e 500 -de 10 -lr 1e-4 -w 0.1 --gpu "0" --use_aleatoric large-data/complete/dataset.json
