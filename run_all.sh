# #!/bin/bash
# set -e

# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod mc_dropout -mc 10 -dpt 0.1 -dpr 0.1 -dpb 0.1 quartres_resnet34_synth; done
# python ~/thesis/masters-thesis/merge_jsons.py MC_SAMPLES_10
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod mc_dropout -mc 20 -dpt 0.1 -dpr 0.1 -dpb 0.1 quartres_resnet34_synth; done
# python ~/thesis/masters-thesis/merge_jsons.py MC_SAMPLES_20
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod mc_dropout -mc 50 -dpt 0.1 -dpr 0.1 -dpb 0.1 quartres_resnet34_synth; done
# python ~/thesis/masters-thesis/merge_jsons.py MC_SAMPLES_50
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod mc_dropout -mc 100 -dpt 0.1 -dpr 0.1 -dpb 0.1 quartres_resnet34_synth; done
# python ~/thesis/masters-thesis/merge_jsons.py MC_SAMPLES_100

# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod mc_dropout -dpt 0.1 -dpr 0.1 quartres_resnet34_synth; done
# python ~/thesis/masters-thesis/merge_jsons.py ADD_DROP_HEADS
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod mc_dropout -dpt 0.1 -dpr 0.1 -dpb 0.1 quartres_resnet34_synth; done
# python ~/thesis/masters-thesis/merge_jsons.py ADD_DROP_BB_HEADS
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod mc_dropout -dpt 0.1 -dpr 0.1 FINAL_dropout_train_no_BB; done
# python ~/thesis/masters-thesis/merge_jsons.py TRAIN_DROP_HEADS
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod mc_dropout -dpt 0.1 -dpr 0.1 -dpb 0.1 FINAL_dropout_train_with_BB; done
# python ~/thesis/masters-thesis/merge_jsons.py TRAIN_DROP_BB_HEADS

# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod mc_dropout -dpb 0.03 quartres_resnet34_synth; done
# python ~/thesis/masters-thesis/merge_jsons.py DROP_BB_003
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

# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod mc_dropout -dpt 0.01 quartres_resnet34_synth; done
# python ~/thesis/masters-thesis/merge_jsons.py DROP_TRANS_001
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod mc_dropout -dpt 0.02 quartres_resnet34_synth; done
# python ~/thesis/masters-thesis/merge_jsons.py DROP_TRANS_002
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

# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod mc_dropout -dpr 0.02 quartres_resnet34_synth; done
# python ~/thesis/masters-thesis/merge_jsons.py DROP_ROT_002
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


# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod mc_dropout -dpb 0.01 -dpt 0.02 -dpr 0.10 quartres_resnet34_synth; done
# python ~/thesis/masters-thesis/merge_jsons.py B001T002R010
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod mc_dropout -dpb 0.03 -dpt 0.02 -dpr 0.10 quartres_resnet34_synth; done
# python ~/thesis/masters-thesis/merge_jsons.py B003T002R010
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod mc_dropout -dpb 0.05 -dpt 0.02 -dpr 0.10 quartres_resnet34_synth; done
# python ~/thesis/masters-thesis/merge_jsons.py B005T002R010
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod mc_dropout -dpb 0.07 -dpt 0.01 -dpr 0.05 quartres_resnet34_synth; done
# python ~/thesis/masters-thesis/merge_jsons.py B007T001R005
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod mc_dropout -dpb 0.10 -dpt 0.01 -dpr 0.05 quartres_resnet34_synth; done
# python ~/thesis/masters-thesis/merge_jsons.py B010T001R005
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod mc_dropout -dpb 0.02 -dpt 0.02 -dpr 0.08 quartres_resnet34_synth; done
# python ~/thesis/masters-thesis/merge_jsons.py B002T002R008


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

# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod bayesian -sn 5 -ccw 1e-3 FINAL_BNN_TRAIN; done
# python ~/thesis/masters-thesis/merge_jsons.py FINAL_BNN_TRAIN
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod bayesian -sn 5 -ccw 1e-3 FINAL_BNN_FINETUNE; done
# python ~/thesis/masters-thesis/merge_jsons.py FINAL_BNN_FINETUNE

# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod bayesian -sn 3 -ccw 1e-3 FINAL_BNN_FINETUNE; done
# python ~/thesis/masters-thesis/merge_jsons.py FINAL_BNN_FINETUNE_SN_3
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod bayesian -sn 5 -ccw 1e-3 FINAL_BNN_FINETUNE; done
# python ~/thesis/masters-thesis/merge_jsons.py FINAL_BNN_FINETUNE_SN_5
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod bayesian -sn 10 -ccw 1e-3 FINAL_BNN_FINETUNE; done
# python ~/thesis/masters-thesis/merge_jsons.py FINAL_BNN_FINETUNE_SN_10
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod bayesian -sn 20 -ccw 1e-3 FINAL_BNN_FINETUNE; done
# python ~/thesis/masters-thesis/merge_jsons.py FINAL_BNN_FINETUNE_SN_20


# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod bayesian -sn 5 FINAL_CCW5_BATCH5; done
# python ~/thesis/masters-thesis/merge_jsons.py FINAL_CCW5_BATCH5
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod bayesian -sn 5 FINAL_CCW4_BATCH5; done
# python ~/thesis/masters-thesis/merge_jsons.py FINAL_CCW4_BATCH5
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod bayesian -sn 5 FINAL_CCW3_BATCH5; done
# python ~/thesis/masters-thesis/merge_jsons.py FINAL_CCW3_BATCH5

# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod bayesian -sn 5 FINAL_CCW5_BATCH12; done
# python ~/thesis/masters-thesis/merge_jsons.py FINAL_CCW5_BATCH12
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod bayesian -sn 5 FINAL_CCW4_BATCH12; done
# python ~/thesis/masters-thesis/merge_jsons.py FINAL_CCW4_BATCH12
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod bayesian -sn 5 FINAL_CCW3_BATCH12; done
# python ~/thesis/masters-thesis/merge_jsons.py FINAL_CCW3_BATCH12

# TEST

# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod mc_dropout -dpt 0.02 -dpr 0.1 quartres_resnet34_synth; done
# python ~/thesis/masters-thesis/merge_jsons.py DROPOUT_TEST
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod ensemble -bs 30 FINAL_ensemble_mod1; done
# python ~/thesis/masters-thesis/merge_jsons.py ENSEMBLE_TEST
# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod bayesian -ccw 1e-5 FINAL_CCW5_BATCH12; done
# python ~/thesis/masters-thesis/merge_jsons.py BAYESIAN_TEST

# TEST LARGE

# for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod mc_dropout -dpb 0.02 -dpr 0.1 FINAL_LARGE_BASELINE; done
# python ~/thesis/masters-thesis/merge_jsons.py LARGE_TEST_DROPOUT


for i in {1..10}; do python ~/thesis/masters-thesis/run_model.py -mod bayesian -ccw 1e-5 FINAL_LARGE_BNN; done
python ~/thesis/masters-thesis/merge_jsons.py LARGE_TEST_BNN