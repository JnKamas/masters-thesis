#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import shutil


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def build_parser(proj_root):
    parser = argparse.ArgumentParser(
        description="Run inference + evaluation in one command",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required
    parser.add_argument(
        "model_name",
        help="Name of the model (without .pth), e.g. bayes1300"
    )

    # Model / backbone
    parser.add_argument("-mod", "--modifications",
                        choices=["mc_dropout", "bayesian", "ensemble", "ensemble_mc_dropout"],
                        default=None,
                        help="Modification type (options: mc_dropout, bayesian, ensemble, ensemble_mc_dropout)")
    parser.add_argument("-bb", "--backbone",
                        default="resnet34",
                        help="Backbone for inference")

    # Inference settings
    parser.add_argument("-mc", "--mc_samples", type=int, default=30,
                        help="Number of Monte Carlo samples")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for inference")
    parser.add_argument("--no_preload", action="store_true",
                        help="Pass --no_preload to infer.py")

    # Paths
    parser.add_argument("--dataset",
                        default=os.path.expanduser("~/thesis/large-data/complete/dataset.json"),
                        help="Path to dataset JSON")
    parser.add_argument("--models_dir",
                        default=os.path.join(proj_root, "models"),
                        help="Directory containing .pth files")
    parser.add_argument("--inference_dir",
                        default=os.path.join(proj_root, "inference"),
                        help="Base output dir")

    # Dropout / UQ
    parser.add_argument("-dpt", "--dropout_prob_trans", type=float, default=0.0,
                        help="Dropout probability for translation")
    parser.add_argument("-dpr", "--dropout_prob_rot", type=float, default=0.0,
                        help="Dropout probability for rotation")
    parser.add_argument("-dp", "--dropout_prob", type=float, default=0.0,
                        help="Dropout probability for MC Dropout heads")
    parser.add_argument("-dpb", "--dropout_prob_backbone", type=float, default=0.0,
                        help="Dropout probability for ResNet backbone residual blocks")

    parser.add_argument("-sn", "--sample_nbr", type=int, default=200,
                        help="Sample number for MC Dropout")
    parser.add_argument("-ccw", "--complexity_cost_weight", type=float, default=0.001,
                        help="Weight for complexity cost in Bayesian layers")
    parser.add_argument("-bt", "--bayesian_type", type=int, default=0,
                        help="Bayesian type")
    parser.add_argument("-is", "--input_sigma", type=float, default=0.1,
                        help="Input sigma for Bayesian layers")

    parser.add_argument("--eval_only", type=bool, default=False,
                        help="If True, only runs evaluation on existing inference results")
    return parser


# ------------------------------------------------------------
# Command Builders
# ------------------------------------------------------------
def build_infer_cmd(args, infer_script, weights_path):
    cmd = [
        sys.executable, infer_script,
        "-bb", args.backbone,
        "-dpt", str(args.dropout_prob_trans),
        "-dpr", str(args.dropout_prob_rot),
        "-dpb", str(args.dropout_prob_backbone),
        "-dp", str(args.dropout_prob),
        "-b", str(args.batch_size),
        "-sn", str(args.sample_nbr),
        "-ccw", str(args.complexity_cost_weight),
        "-bt", str(args.bayesian_type),
        "-is", str(args.input_sigma),
        "--weights_path", weights_path,
        "--mc_samples", str(args.mc_samples),
        args.dataset,
    ]

    if args.modifications:
        cmd += ["-mod", args.modifications]

    if args.no_preload:
        cmd.append("--no_preload")

    return cmd


def build_eval_cmd(args, eval_script, inference_output_dir):
    cmd = [
        sys.executable, eval_script,
        "--mc_samples", str(args.mc_samples),
        inference_output_dir
    ]
    if args.modifications:
        cmd += ["--modifications", args.modifications]
    return cmd


# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------
def save_results(proj_root, args, evaluated_block):
    results_dir = os.path.join(proj_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    name = args.model_name
    name += f"_dpt{args.dropout_prob_trans}"
    name += f"_dpr{args.dropout_prob_rot}"
    name += f"_dp{args.dropout_prob}"
    if args.modifications:
        name += f"_{args.modifications}"

    result_file = os.path.join(results_dir, name + ".txt")

    with open(result_file, "w") as f:
        for line in evaluated_block:
            f.write(line + "\n")

    return result_file


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    proj_root = os.path.normpath(os.path.join(script_dir, ".."))

    parser = build_parser(proj_root)
    args = parser.parse_args()

    infer_script = os.path.join(script_dir, "infer.py")
    eval_script = os.path.join(script_dir, "evaluate.py")

    inference_output_dir = os.path.join(args.inference_dir, args.model_name)

    # Clean old infer output
    if not args.eval_only:
        if os.path.exists(inference_output_dir):
            shutil.rmtree(inference_output_dir)
        os.makedirs(inference_output_dir, exist_ok=True)

    # -----------------------------
    # 1) INFERENCE
    # -----------------------------
    if not args.eval_only:
        if args.modifications in ["ensemble", "ensemble_mc_dropout"]:
            ensemble_dir = os.path.join(args.models_dir, args.model_name)

            if not os.path.isdir(ensemble_dir):
                raise FileNotFoundError(f"Ensemble folder not found: {ensemble_dir}")

            ckpts = sorted(f for f in os.listdir(ensemble_dir) if f.endswith(".pth"))
            if not ckpts:
                raise RuntimeError(f"No .pth files in ensemble folder {ensemble_dir}")

            for ckpt in ckpts:
                weights_path = os.path.join(ensemble_dir, ckpt)
                infer_cmd = build_infer_cmd(args, infer_script, weights_path)

                print("▶︎ Inference (ensemble):", " ".join(infer_cmd))
                if subprocess.run(infer_cmd).returncode != 0:
                    sys.exit(1)

        else:
            weights_path = os.path.join(args.models_dir, args.model_name + ".pth")

            infer_cmd = build_infer_cmd(args, infer_script, weights_path)
            print("▶︎ Inference:", " ".join(infer_cmd))
            if subprocess.run(infer_cmd).returncode != 0:
                sys.exit(1)

    # -----------------------------
    # 2) EVALUATION
    # -----------------------------
    eval_cmd = build_eval_cmd(args, eval_script, inference_output_dir)
    print("▶︎ Evaluation:", " ".join(eval_cmd))

    result = subprocess.run(
        eval_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    # Extract block starting with "Evaluated samples"
    lines = result.stdout.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith("Evaluated samples")), None)
    evaluated_block = lines[start:] if start is not None else []

    # Save results
    out_file = save_results(proj_root, args, evaluated_block)

    print(f"✔ Results saved to: {out_file}")
    print(result.stdout)

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
