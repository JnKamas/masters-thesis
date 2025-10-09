#!/usr/bin/env python3
import os, sys, argparse, subprocess, shutil

def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    proj_root = os.path.normpath(os.path.join(script_dir, '..'))
    # I need to merge two parsers which I have, but the main issue is that I cannot include moel_name in the other one... just TODO
    parser = argparse.ArgumentParser(description="Run inference + evaluation in one command")
    parser.add_argument('model_name', help="Name of the model (without .pth), e.g. bayes1300")
    parser.add_argument('-mod','--modifications',choices=['mc_dropout','bayesian'],default=None,help="Modification type (options: mc_dropout, bayesian)")
    parser.add_argument('-bb','--backbone',default='resnet34',help="Backbone for inference (default: resnet34)")
    parser.add_argument('-mc','--mc_samples',type=int,default=30,help="Number of Monte Carlo samples (default: 30)")
    parser.add_argument('--batch_size',type=int,default=8,help="Batch size for inference (default: 8)")
    parser.add_argument('--no_preload',action='store_true',help="Pass --no_preload to infer.py")
    parser.add_argument('--dataset',default=os.path.expanduser('~/thesis/large-data/complete/dataset.json'),help="Path to dataset JSON")
    parser.add_argument('--models_dir',default=os.path.join(proj_root,'models'),help="Directory containing .pth files")
    parser.add_argument('--inference_dir',default=os.path.join(proj_root,'inference'),help="Base output dir")
    parser.add_argument('-dpt', '--dropout_prob_trans', type=float, default=0, help='Dropout probability for translation') # add JK
    parser.add_argument('-dpr', '--dropout_prob_rot', type=float, default=0, help='Dropout probability for rotation') # add JK
    parser.add_argument('-dp', '--dropout_prob', type=float, default=0, help='Dropout probability for MC Dropout') # add JK
    parser.add_argument('-sn', '--sample_nbr', type=int, default=0, help='Sample number for MC Dropout')
    parser.add_argument('-ccw', '--complexity_cost_weight', type=float, default=0.001, help='Weight for complexity cost in Bayesian layers')
    parser.add_argument('-bt', '--bayesian_type' , type=int, default=0, help='Bayesian type: 0 for full MLP Bayesian, 1 for first MLP layer Bayesian, 2 for last MLP layer Bayesian, 3 for only mid Bayesian, 4 for complete bayesian')
    parser.add_argument('-is', '--input_sigma', type=float, default=0.1, help='Input sigma for Bayesian layers')
    args = parser.parse_args()

    infer_script = os.path.join(script_dir, 'infer.py')
    eval_script  = os.path.join(script_dir, 'evaluate.py')
    weights_path = os.path.join(args.models_dir, args.model_name + '.pth')
    infer_out    = os.path.join(args.inference_dir, args.model_name)

    # clean previous outputs
    if os.path.exists(infer_out):
        shutil.rmtree(infer_out)
    os.makedirs(infer_out, exist_ok=True)

    # build & run infer.py
    infer_cmd = [sys.executable, infer_script, '-bb', args.backbone, '-dpt', str(args.dropout_prob_trans), '-dpr', str(args.dropout_prob_rot), '-b', str(args.batch_size), '-dp', str(args.dropout_prob), '-sn', str(args.sample_nbr), '-ccw', str(args.complexity_cost_weight), '-bt', str(args.bayesian_type), '-is', str(args.input_sigma)]
    if args.modifications: infer_cmd += ['-mod', args.modifications]
    infer_cmd += ['--weights_path', weights_path, '--mc_samples', str(args.mc_samples)]
    if args.no_preload: infer_cmd.append('--no_preload')
    infer_cmd.append(args.dataset)
    print("▶︎ Inference:", ' '.join(infer_cmd))
    if subprocess.run(infer_cmd).returncode != 0:
        sys.exit(1)

    # build & run evaluate.py
    eval_cmd = [sys.executable, eval_script]
    if args.modifications: eval_cmd += ['--modifications', args.modifications]
    eval_cmd += ['--mc_samples', str(args.mc_samples), infer_out]
    print("▶︎ Evaluation:", ' '.join(eval_cmd))
    # Run evaluate.py and capture stdout
    result = subprocess.run(eval_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # Extract everything from "Evaluated samples" onward
    output_lines = result.stdout.splitlines()
    start_idx = next((i for i, line in enumerate(output_lines) if line.startswith("Evaluated samples")), None)

    # If found, slice the output from that point onward
    evaluated_block = output_lines[start_idx:] if start_idx is not None else []

    # Save to results/model_name.txt
    results_dir = os.path.join(proj_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    name = args.model_name
    try: 
        name += f"_dpt{args.dropout_prob_trans}"
    except:
        pass
    try:
        name += f"_dpr{args.dropout_prob_rot}"
    except:
        pass
    try:
        name += f"_dp{args.dropout_prob}"
    except:
        pass
    try:
        name += f"_{args.modifications}"
    except:
        pass
    result_file = os.path.join(results_dir, name + '.txt')

    with open(result_file, 'w') as f:
        for line in evaluated_block:
            f.write(line + '\n')

    # Optionally print the full output (for debugging/logs)
    print(result.stdout)

    # Exit with the same return code
    sys.exit(result.returncode)


if __name__=='__main__':
    main()
