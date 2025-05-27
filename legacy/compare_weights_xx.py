#!/usr/bin/env python3
"""
compare_weights.py

Compare how a given checkpoint’s parameters map onto two Network variants:
  (a) classic (no Bayesian modifications), and
  (b) Bayesian (with dropout/Bayesian layers inserted),
using your load_model utility.
"""
import argparse
import torch
from main import load_model


def remap_bayesian_state_dict(base_sd):
    """
    Remap baseline state_dict keys for Bayesian layers:
      fc_*.weight → fc_*.weight_mu
      fc_*.bias   → fc_*.bias_mu
    """
    new_sd = {}
    for k, v in base_sd.items():
        parts = k.split('.')
        name, attr = parts[0], parts[-1]
        if name in ('fc_z','fc_y','fc_t') and attr in ('weight','bias'):
            new_key = k + '_mu'
        else:
            new_key = k
        new_sd[new_key] = v
    return new_sd


def compare_assignments(args):
    # load raw checkpoint
    ckpt = torch.load(args.weights_path, map_location='cpu')
    ckpt_keys = list(ckpt.keys())

    # ensure dropout attrs exist
    for name in ('dropout_prob_rot', 'dropout_prob_trans'):
        if not hasattr(args, name):
            setattr(args, name, 0.0)

    # build args for classic vs bayesian
    base_args  = argparse.Namespace(**vars(args), modifications=None)
    bayes_args = argparse.Namespace(**vars(args), modifications='bayesian')

    # instantiate and load via load_model
    model_base  = load_model(base_args)
    model_bayes = load_model(bayes_args)

    sd_base  = model_base.state_dict()
    sd_bayes = model_bayes.state_dict()

    # header
    print(f"{'key':50} | {'in_base':8} | {'shape_base':15} | {'in_bayes':8} | {'shape_bayes':15}")"{'key':50} | {'in_base':8} | {'shape_base':15} | {'in_bayes':8} | {'shape_bayes':15}"})
    print("-"*120)

    # compare mapping of ckpt keys
    for k in ckpt_keys:
        in_b   = k in sd_base
        sb     = tuple(sd_base[k].shape)  if in_b  else '-'
        # for bayesian, checkpoint k maps to k+'_mu'
        k_mu   = k + '_mu'
        in_ba  = k_mu in sd_bayes
        sba    = tuple(sd_bayes[k_mu].shape) if in_ba else '-'
        print(f"{k:50} | {str(in_b):8} | {str(sb):15} | {str(in_ba):8} | {str(sba):15}")

    # now show missing/unexpected when loading raw into base, and remapped into bayes
    print("\nLoading with strict=False:")
    # classic load
    res_base = model_base.load_state_dict(ckpt, strict=False)
    # bayesian load uses remapped state_dict
    state_dict_bayes = remap_bayesian_state_dict(ckpt)
    res_bayes = model_bayes.load_state_dict(state_dict_bayes, strict=False)

    print("\n-- classic model --")
    print(" missing_keys:   ", res_base.missing_keys)
    print(" unexpected_keys:", res_base.unexpected_keys)
    print("\n-- Bayesian model --")
    print(" missing_keys:   ", res_bayes.missing_keys)
    print(" unexpected_keys:", res_bayes.unexpected_keys)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compare checkpoint parameter assignment to model variants"
    )
    parser.add_argument('--weights_path',       type=str, required=True,
                        help='Path to .pth checkpoint')
    parser.add_argument('--backbone',           type=str, default='resnet34',
                        help='resnet18|resnet34|resnet50')
    parser.add_argument('--dropout_prob',       type=float, default=0.5,
                        help='Dropout prob for MC Dropout / Bayesian model')
    parser.add_argument('--dropout_prob_rot',   type=float, default=0.0,
                        help='Rotation-dropout prob (fallback if 0)')
    parser.add_argument('--dropout_prob_trans', type=float, default=0.0,
                        help='Translation-dropout prob (fallback if 0)')
    parser.add_argument('--resume',             type=int,   default=None,
                        help='Which epoch checkpoint to resume (if any)')
    args = parser.parse_args()

    compare_assignments(args)
