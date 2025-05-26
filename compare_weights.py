#!/usr/bin/env python3
"""
compare_weights.py

Compare how a given checkpoint’s parameters map onto two Network variants:
  (a) baseline (no dropout), and
  (b) MC–Dropout (with dropout layers inserted).

For each key in the checkpoint, reports whether it matches a parameter
in each model variant, and the corresponding shape.

TO RUN:

python masters-thesis/compare_weights.py \
  --weights_path models/quartres_resnet34_synth.pth \
  --backbone resnet34 \
  --dropout_prob 0.5
  
"""
import argparse
import torch
from network import Network

def compare_assignments(weights_path, backbone, dropout_prob):
    # load checkpoint
    ckpt = torch.load(weights_path, map_location='cpu')
    ckpt_keys = list(ckpt.keys())

    # instantiate both model variants
    model_base = Network(backbone=backbone, modifications=None, dropout_prob=dropout_prob)
    model_mc   = Network(backbone=backbone, modifications='mc_dropout', dropout_prob=dropout_prob)

    sd_base = model_base.state_dict()
    sd_mc   = model_mc.state_dict()

    # header
    print(f"{'key':50} | {'in_base':8} | {'shape_base':15} | {'in_mc':8} | {'shape_mc':15}")
    print("-"*110)

    # compare each checkpoint key
    for k in ckpt_keys:
        in_b = k in sd_base
        in_m = k in sd_mc
        sb   = tuple(sd_base[k].shape) if in_b else '-'
        sm   = tuple(sd_mc[k].shape)   if in_m else '-'
        print(f"{k:50} | {str(in_b):8} | {str(sb):15} | {str(in_m):8} | {str(sm):15}")

    # Optionally, show missing/unexpected keys when loading
    print("\nLoading with strict=False:")
    res_base = model_base.load_state_dict(ckpt, strict=False)
    res_mc   = model_mc.load_state_dict(ckpt, strict=False)
    print("\n-- baseline (no dropout) --")
    print(" missing_keys:   ", res_base.missing_keys)
    print(" unexpected_keys:", res_base.unexpected_keys)
    print("\n-- mc_dropout model --")
    print(" missing_keys:   ", res_mc.missing_keys)
    print(" unexpected_keys:", res_mc.unexpected_keys)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compare checkpoint parameter assignment to model variants"
    )
    parser.add_argument('--weights_path', type=str, required=True,
                        help='Path to the .pth checkpoint file')
    parser.add_argument('--backbone', type=str, default='resnet34',
                        help='Backbone: resnet18, resnet34, or resnet50')
    parser.add_argument('--dropout_prob', type=float, default=0.5,
                        help='Dropout probability for MC Dropout model')
    args = parser.parse_args()

    compare_assignments(
        weights_path=args.weights_path,
        backbone=args.backbone,
        dropout_prob=args.dropout_prob
    )
