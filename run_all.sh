#!/bin/bash
set -e

# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod mc_dropout -dpt 0.1 -dpr 0.1 quartres_resnet34_synth; done
# python ~/thesis/masters-thesis/merge_jsons.py ADD_DROP_HEADS
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod mc_dropout -dpt 0.1 -dpr 0.1 -dpb 0.1 quartres_resnet34_synth; done
# python ~/thesis/masters-thesis/merge_jsons.py ADD_DROP_BB_HEADS
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod mc_dropout -dpt 0.1 -dpr 0.1 FINAL_dropout_train_no_BB; done
# python ~/thesis/masters-thesis/merge_jsons.py TRAIN_DROP_HEADS
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod mc_dropout -dpt 0.1 -dpr 0.1 -dpb 0.1 FINAL_dropout_train_with_BB; done
# python ~/thesis/masters-thesis/merge_jsons.py TRAIN_DROP_BB_HEADS

# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod mc_dropout -dpb 0.05 quartres_resnet34_synth; done
# python ~/thesis/masters-thesis/merge_jsons.py DROP_BB_005
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod mc_dropout -dpb 0.1 quartres_resnet34_synth; done
# python ~/thesis/masters-thesis/merge_jsons.py DROP_BB_010
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod mc_dropout -dpb 0.15 quartres_resnet34_synth; done
# python ~/thesis/masters-thesis/merge_jsons.py DROP_BB_015
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod mc_dropout -dpb 0.20 quartres_resnet34_synth; done
# python ~/thesis/masters-thesis/merge_jsons.py DROP_BB_020
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod mc_dropout -dpb 0.30 quartres_resnet34_synth; done
# python ~/thesis/masters-thesis/merge_jsons.py DROP_BB_030
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod mc_dropout -dpb 0.40 quartres_resnet34_synth; done
# python ~/thesis/masters-thesis/merge_jsons.py DROP_BB_040

# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod mc_dropout -dpt 0.05 quartres_resnet34_synth; done
# python ~/thesis/masters-thesis/merge_jsons.py DROP_TRANS_005
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod mc_dropout -dpt 0.07 quartres_resnet34_synth; done
# python ~/thesis/masters-thesis/merge_jsons.py DROP_TRANS_007
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod mc_dropout -dpt 0.1 quartres_resnet34_synth; done
# python ~/thesis/masters-thesis/merge_jsons.py DROP_TRANS_010
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod mc_dropout -dpt 0.15 quartres_resnet34_synth; done
# python ~/thesis/masters-thesis/merge_jsons.py DROP_TRANS_015
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod mc_dropout -dpt 0.20 quartres_resnet34_synth; done
# python ~/thesis/masters-thesis/merge_jsons.py DROP_TRANS_020


# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod mc_dropout -dpr 0.05 quartres_resnet34_synth; done
# python ~/thesis/masters-thesis/merge_jsons.py DROP_ROT_005
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod mc_dropout -dpr 0.1 quartres_resnet34_synth; done
# python ~/thesis/masters-thesis/merge_jsons.py DROP_ROT_010
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod mc_dropout -dpr 0.20 quartres_resnet34_synth; done
# python ~/thesis/masters-thesis/merge_jsons.py DROP_ROT_020
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod mc_dropout -dpr 0.30 quartres_resnet34_synth; done
# python ~/thesis/masters-thesis/merge_jsons.py DROP_ROT_030
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod mc_dropout -dpr 0.50 quartres_resnet34_synth; done
# python ~/thesis/masters-thesis/merge_jsons.py DROP_ROT_050


# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod mc_dropout -dpb 0.1 -dpt 0.05 -dpr 0.10 quartres_resnet34_synth; done
# python ~/thesis/masters-thesis/merge_jsons.py B010T005R010
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod mc_dropout -dpb 0.1 -dpt 0.05 -dpr 0.20 quartres_resnet34_synth; done
# python ~/thesis/masters-thesis/merge_jsons.py B010T005R020
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod mc_dropout -dpb 0.15 -dpt 0.05 -dpr 0.10 quartres_resnet34_synth; done
# python ~/thesis/masters-thesis/merge_jsons.py B015T005R010
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod mc_dropout -dpb 0.1 -dpt 0.05 -dpr 0.25 quartres_resnet34_synth; done
# python ~/thesis/masters-thesis/merge_jsons.py B010T005R025
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod mc_dropout -dpb 0.12 -dpt 0.05 -dpr 0.20 quartres_resnet34_synth; done
# python ~/thesis/masters-thesis/merge_jsons.py B012T005R020


# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod ensemble -bs 3 FINAL_ensemble_mod1; done
# python ~/thesis/masters-thesis/merge_jsons.py ENS1_BS_3
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod ensemble -bs 5 FINAL_ensemble_mod1; done
# python ~/thesis/masters-thesis/merge_jsons.py ENS1_BS_5
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod ensemble -bs 10 FINAL_ensemble_mod1; done
# python ~/thesis/masters-thesis/merge_jsons.py ENS1_BS_10
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod ensemble -bs 20 FINAL_ensemble_mod1; done
# python ~/thesis/masters-thesis/merge_jsons.py ENS1_BS_20
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod ensemble -bs 30 FINAL_ensemble_mod1; done
# python ~/thesis/masters-thesis/merge_jsons.py ENS1_BS_30

# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod ensemble -bs 3 FINAL_ensemble_mod2; done
# python ~/thesis/masters-thesis/merge_jsons.py ENS2_BS_3
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod ensemble -bs 5 FINAL_ensemble_mod2; done
# python ~/thesis/masters-thesis/merge_jsons.py ENS2_BS_5
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod ensemble -bs 10 FINAL_ensemble_mod2; done
# python ~/thesis/masters-thesis/merge_jsons.py ENS2_BS_10
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod ensemble -bs 20 FINAL_ensemble_mod2; done
# python ~/thesis/masters-thesis/merge_jsons.py ENS2_BS_20
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod ensemble -bs 30 FINAL_ensemble_mod2; done
# python ~/thesis/masters-thesis/merge_jsons.py ENS2_BS_30

# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod ensemble -bs 25 FINAL_ensemble_mod3_001; done
# python ~/thesis/masters-thesis/merge_jsons.py ENS3_P_001
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod ensemble -bs 25 FINAL_ensemble_mod3_002; done
# python ~/thesis/masters-thesis/merge_jsons.py ENS3_P_002
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod ensemble -bs 25 FINAL_ensemble_mod3_005; done
# python ~/thesis/masters-thesis/merge_jsons.py ENS3_P_005
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod ensemble -bs 25 FINAL_ensemble_mod3_010; done
# python ~/thesis/masters-thesis/merge_jsons.py ENS3_P_010

for i in {1..10}; do python ~/thesis/masters-thesis/train.py -bb resnet34 -iw 516 -ih 386 -b 12 -e 500 -de 10 -lr 1e-3 -w 0.1 -ccw 1e-4 -is 0.1 -bt 0 -sn 5 -mod bayesian ~/thesis/large-data/complete/dataset.json; done
