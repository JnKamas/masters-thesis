#!/usr/bin/env python3
import os, sys, argparse, subprocess, shutil

def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    proj_root = os.path.normpath(os.path.join(script_dir, '..'))
    parser = argparse.ArgumentParser(description="Run inference + evaluation in one command")
    parser.add_argument('model_name', help="Name of the model (without .pth), e.g. bayes1300")
    parser.add_argument('-mod','--modifications',choices=['mc_dropout','bayesian'],default=None,help="Modification type (options: mc_dropout, bayesian)")
    parser.add_argument('-bb','--backbone',default='resnet34',help="Backbone for inference (default: resnet34)")
    parser.add_argument('--no_preload',action='store_true',help="Pass --no_preload to infer.py")
    parser.add_argument('--dataset',default=os.path.expanduser('~/thesis/large-data/complete/dataset.json'),help="Path to dataset JSON")
    parser.add_argument('--models_dir',default=os.path.join(proj_root,'models'),help="Directory containing .pth files")
    parser.add_argument('--inference_dir',default=os.path.join(proj_root,'inference'),help="Base output dir")
    args = parser.parse_args()

    infer_script = os.path.join(script_dir,'infer.py')
    eval_script  = os.path.join(script_dir,'evaluate.py')
    weights_path = os.path.join(args.models_dir, args.model_name + '.pth')
    infer_out    = os.path.join(args.inference_dir, args.model_name)

    # remove previous outputs
    if os.path.exists(infer_out):
        shutil.rmtree(infer_out)
    os.makedirs(infer_out, exist_ok=True)

    # build & run infer.py
    infer_cmd = [sys.executable, infer_script, '-bb', args.backbone]
    if args.modifications: infer_cmd += ['-mod', args.modifications]
    infer_cmd += ['--weights_path', weights_path]
    if args.no_preload: infer_cmd.append('--no_preload')
    infer_cmd.append(args.dataset)
    print("▶︎ Inference:", ' '.join(infer_cmd))
    if subprocess.run(infer_cmd).returncode != 0: sys.exit(1)

    # build & run evaluate.py
    eval_cmd = [sys.executable, eval_script]
    if args.modifications: eval_cmd += ['--modifications', args.modifications]
    eval_cmd.append(infer_out)
    print("▶︎ Evaluation:", ' '.join(eval_cmd))
    sys.exit(subprocess.run(eval_cmd).returncode)

if __name__=='__main__':
    main()
