#!/bin/bash
set -e

run_until_success () {
  GPU_ID=$1
  shift
  while true; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python masters-thesis/train.py "$@" --gpu "$GPU_ID" && break
    echo "GPU$GPU_ID busy, retrying in 5s..."
    sleep 5
  done
}

# run_until_success 0 -bb resnet34 -iw 516 -ih 386 -b 12 -e 500 -de 10 -lr 1e-3 -w 0.1 -ccw 1e-5 -is 0.1 -bt 0 -sn 5 -mod bayesian ~/thesis/large-data/larger-dataset/train_val.json &
run_until_success 1 -bb resnet34 -iw 516 -ih 386 -b 12 -e 500 -de 10 -lr 1e-3 -w 0.1 -ccw 1e-5 -is 0.1 -bt 0 -sn 5 -mod bayesian ~/thesis/large-data/larger-dataset/train_val.json &

wait

