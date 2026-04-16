#!/bin/bash
set -e

#!/bin/bash
set -e

run_until_success () {
  while true; do
    python masters-thesis/train.py "$@" && break
    echo "GPU busy, retrying in 5s..."
    sleep 5
  done
}

run_until_success -bb resnet34 -iw 516 -ih 386 -b 12 -e 500 -de 10 -lr 1e-3 -w 0.1 -w 0.1 -ccw 1e-5 -is 0.1 -bt 0 -sn 5 -mod bayesian -wp models/quartres_resnet34_synth.pth large-data/complete/dataset.json
run_until_success -bb resnet34 -iw 516 -ih 386 -b 12 -e 500 -de 10 -lr 1e-3 -w 0.1 -w 0.1 -ccw 1e-4 -is 0.1 -bt 0 -sn 5 -mod bayesian -wp models/quartres_resnet34_synth.pth large-data/complete/dataset.json
run_until_success -bb resnet34 -iw 516 -ih 386 -b 12 -e 500 -de 10 -lr 1e-3 -w 0.1 -w 0.1 -ccw 1e-3 -is 0.1 -bt 0 -sn 5 -mod bayesian -wp models/quartres_resnet34_synth.pth large-data/complete/dataset.json