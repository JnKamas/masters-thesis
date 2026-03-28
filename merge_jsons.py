import json
import glob
import numpy as np
import os
import shutil

EXPECTED = 10

files = glob.glob("eval_results/*.json")

if len(files) > EXPECTED:
    print(f"Error: Found {len(files)} files, expected {EXPECTED}. Remove extra jsons")
    exit()
elif len(files) < EXPECTED:
    print(f"Error: Found only {len(files)} files, expected {EXPECTED}. Make sure all evaluations are completed.")
    exit()

i = 1
while os.path.exists(f"final_results/experiment_{i}"):
    i += 1

exp_dir = f"final_results/experiment_{i}"
os.makedirs(exp_dir)

moved_files = []
for f in files:
    dst = os.path.join(exp_dir, os.path.basename(f))
    shutil.move(f, dst)
    moved_files.append(dst)

def extract(d):
    return {
        "eTE": d["results"]["pose"]["eTE"]["mean"],
        "eRE": d["results"]["pose"]["eRE"]["mean"],

        "cov_t": np.mean(d["results"]["epistemic"]["translation"]["coverage"]["gt"]),
        "sharp_t": d["results"]["epistemic"]["translation"]["sharpness"]["vector"],
        "nll_t": d["results"]["epistemic"]["translation"]["nll"],
        "crps_t": d["results"]["epistemic"]["translation"]["crps"],
        "corr_t": d["results"]["epistemic"]["translation"]["corr"],

        "cov_r": d["results"]["epistemic"]["rotation"]["coverage"],
        "sharp_r": d["results"]["epistemic"]["rotation"]["sharpness"]["rad"],
        "nll_r": d["results"]["epistemic"]["rotation"]["nll_matrix_fisher"],
        "crps_r": d["results"]["epistemic"]["rotation"]["crps"],
        "corr_r": d["results"]["epistemic"]["rotation"]["corr"],
    }

all_metrics = []
for f in moved_files:
    with open(f) as fp:
        all_metrics.append(extract(json.load(fp)))

agg = {}
for k in all_metrics[0].keys():
    vals = np.array([m[k] for m in all_metrics])
    agg[k] = (vals.mean(), vals.std())


order = [
    "eTE", "eRE",
    "cov_t", "sharp_t", "nll_t", "crps_t", "corr_t",
    "cov_r", "sharp_r", "nll_r", "crps_r", "corr_r"
]

out_path = os.path.join(exp_dir, "results.txt")

with open(out_path, "w") as f:
    for k in order:
        mean, std = agg[k]
        line = f"{k:10s}: {mean:.2f} ± {std:.2f}"
        print(line)
        f.write(line + "\n")

print(f"\nSaved to {out_path}")