#!/usr/bin/env python3
"""
compare_weights.py

Compare how a given checkpoint’s parameters map onto two Network variants:
  (a) classic (no Bayesian modifications), and
  (b) Bayesian (with dropout/Bayesian layers inserted).
"""
import argparse
import torch
from network import Network

def compare_assignments(args):
    # load checkpoint
    ckpt = torch.load(args.weights_path, map_location='cpu')
    keys = list(ckpt.keys())

    # --- ENSURE that both rotation- and translation-dropout attrs exist ---
    for name in ('dropout_prob_rot', 'dropout_prob_trans'):
        if not hasattr(args, name):
            setattr(args, name, 0.0)

    # build two args namespaces, identical except for 'modifications'
    base_args  = argparse.Namespace(**vars(args), modifications=None)
    bayes_args = argparse.Namespace(**vars(args), modifications='bayesian')

    # instantiate
    m_base  = Network(base_args)
    m_bayes = Network(bayes_args)

    sd_base, sd_bayes = m_base.state_dict(), m_bayes.state_dict()

    # header
    print(f"{'key':50} | {'in_base':8} | {'shape_base':15} | {'in_bayes':8} | {'shape_bayes':15}")
    print("-"*120)

    # compare
    for k in keys:
        in_b   = k in sd_base
        in_ba  = k in sd_bayes
        sb     = tuple(sd_base[k].shape)  if in_b  else '-'
        sba    = tuple(sd_bayes[k].shape) if in_ba else '-'
        print(f"{k:50} | {str(in_b):8} | {str(sb):15} | {str(in_ba):8} | {str(sba):15}")

    # load with strict=False to see missing/unexpected
    print("\nLoading with strict=False:")
    rb  = m_base.load_state_dict(ckpt, strict=False)
    rba = m_bayes.load_state_dict(ckpt, strict=False)

    print("\n-- classic model --")
    print(" missing_keys:   ", rb.missing_keys)
    print(" unexpected_keys:", rb.unexpected_keys)
    print("\n-- Bayesian model --")
    print(" missing_keys:   ", rba.missing_keys)
    print(" unexpected_keys:", rba.unexpected_keys)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compare checkpoint parameter assignment to model variants"
    )
    parser.add_argument('--weights_path',       required=True,
                        help='Path to .pth checkpoint')
    parser.add_argument('--backbone',           default='resnet34',
                        help='resnet18|resnet34|resnet50')
    parser.add_argument('--dropout_prob',       type=float, default=0.5,
                        help='Dropout prob for Bayesian model')
    parser.add_argument('--dropout_prob_rot',   type=float, default=0.0,
                        help='Rotation-dropout prob (fallback if 0)')
    parser.add_argument('--dropout_prob_trans', type=float, default=0.0,
                        help='Translation-dropout prob (fallback if 0)')
    args = parser.parse_args()

    compare_assignments(args)
