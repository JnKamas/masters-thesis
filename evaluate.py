import argparse
import glob
import os
from datetime import datetime
from pathlib import Path
import json
import numpy as np
from statistics import mean, median
from scipy.linalg import logm, svd
from scipy.spatial.transform import Rotation as sciR
import warnings
from collections import defaultdict
import shutil
from metrics import *

warnings.filterwarnings("ignore", category=RuntimeWarning)

# presun toto neskor  do druheho suboru
def matrix_fisher_nll(R_pred, R_gt, kappa, eps=1e-8):
    """
    Matrix–Fisher negative log-likelihood on SO(3).

    Args:
        R_pred : (3,3) predicted rotation matrix
        R_gt   : (3,3) ground-truth rotation matrix
        kappa  : (3,) concentration parameters (must be >= 0)

    Returns:
        nll : float
    """

    # rotation error
    R_err = R_pred.T @ R_gt

    # alignment term
    align = (
        kappa[0] * R_err[0, 0] +
        kappa[1] * R_err[1, 1] +
        kappa[2] * R_err[2, 2]
    )

    # normalization constant (stable isotropic approx)
    kappa_bar = np.mean(kappa)
    log_c = np.log(kappa_bar + eps) - kappa_bar

    # negative log-likelihood
    return -align + log_c

def read_kappa_from_prediction_file(txt_path):
    """
    Reads kappa_x kappa_y kappa_z from the prediction txt file.
    Assumes kappa is stored as the last numeric line after a comment.
    """
    with open(txt_path, "r") as f:
        lines = f.readlines()

    for line in reversed(lines):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) == 3:
            try:
                return np.array([float(p) for p in parts], dtype=np.float32)
            except ValueError:
                pass
    return None

def get_mc_predictions(path, number, mc_samples):
    Rs, ts, kappas = [], [], []
    for i in range(mc_samples):
        fname = f'prediction{i}_scan_{number}.txt'
        fpath = os.path.join(path, fname)
        if not os.path.isfile(fpath):
            return None, None, None

        R, t = read_transform_file(fpath)
        kappa = read_kappa_from_prediction_file(fpath)

        Rs.append(R)
        ts.append(t)
        kappas.append(kappa)

    return np.stack(Rs), np.stack(ts), np.stack(kappas)

def evaluate(args):
    print(args.path)
    gt_files = glob.iglob(args.path + '/**/*.txt', recursive=True)
    good_gt_files = [f for f in gt_files if not any(sub in f for sub in ['bad', 'catas', 'ish', 'pred', 'icp', 'refined']) and any(sub in f for sub in ['scan_', 'gt_'])]

    eTE_list = []
    eTE_list_icp = []
    eRE_list = []
    eRE_list_icp = []
    eGD_list = []
    eGD_list_icp = []

    counter_better = 0
    counter_worse = 0

    all_preds_t = []
    all_preds_R = []
    all_gts_t = []
    all_gts_R = []

    all_kappas = []        # list of (N_mc, 3)
    all_kappa_means = []  # per sample

    coverage_t = np.zeros(3)
    coverage_pred_t = np.zeros(3)

    rotation_errors = []
    rotation_std_errors = []
    spread_list = []
    orth_dev_list = []
    det_dev_list = []

    # ---- New Metrics Lists ----
    credible_region_radii = []
    credible_region_coverages = []
    eaad_list = []

    for file in good_gt_files:
        path, gt_file = os.path.split(file)
        index = gt_file.rfind("_")
        number = gt_file[index + 1:-4]

        if args.modifications in {"mc_dropout", "bayesian", "ensemble", "ensemble_mc_dropout"}:
            Rs, ts, kappas = get_mc_predictions(path, number, args.mc_samples)
            all_kappas.append(kappas)
            all_kappa_means.append(np.mean(kappas, axis=0))  # (3,)

            if Rs is None:
                print(f"Some samples missing for {number}, skipping.")
                continue
        else:
            pr_file = number + '.txt'
            if not os.path.isfile(os.path.join(path, pr_file)):
                if os.path.isfile(os.path.join(path, 'prediction_scan_' + number + '.txt')):
                    pr_file = 'prediction_scan_' + number + '.txt'
                else:
                    print("Prediction file not found for " + file)
                    continue

            pr_R, pr_t = read_transform_file(os.path.join(path, pr_file))
            kappa = read_kappa_from_prediction_file(os.path.join(path, pr_file))

            Rs = np.expand_dims(pr_R, 0)
            ts = np.expand_dims(pr_t, 0)
            kappas = np.expand_dims(kappa, 0)

            all_kappas.append(kappas)
            all_kappa_means.append(kappa)

        # ---- shared logic (already exists) ----
        mean_t = np.mean(ts, axis=0)
        std_t = np.std(ts, axis=0)
        pr_t = mean_t
        pr_R = mean_rotation_SVD(Rs)
        pts_arr = ts
        rots = Rs


        gt_R1, gt_t = read_transform_file(os.path.join(path, gt_file))
        if np.linalg.det(gt_R1) < 0:
            gt_R1[:, 1] *= -1
        if np.linalg.det(pr_R) < 0:
            pr_R[:, 1] *= -1
        gt_R2 = np.matrix.copy(gt_R1)
        gt_R2[:, :2] *= -1

        # ---- Old metrics (mean only) ----
        eTE_list.append(calculate_eTE(gt_t, mean_t))
        if np.array_equal(pr_R, np.eye(3, 3)):
            eRE_list.append(float("inf"))
        else:
            eRE_list.append(min(calculate_eRE(gt_R1, pr_R), calculate_eRE(gt_R2, pr_R)))
        eGD_list.append(min(calculate_eGD(gt_R1, pr_R), calculate_eGD(gt_R2, pr_R)))


        all_preds_t.append(pts_arr)
        all_preds_R.append(rots)
        all_gts_t.append(gt_t)
        all_gts_R.append(gt_R1)  # account for 180° symmetry later

        for j in range(3):
            lo = np.percentile(pts_arr[:, j], 2.5)
            hi = np.percentile(pts_arr[:, j], 97.5)
            if lo <= gt_t[j] <= hi:
                coverage_t[j] += 1
            if lo <= mean_t[j] <= hi:
                coverage_pred_t[j] += 1

        angs = []
        for r in rots:
            q = sciR.from_matrix(r)
            err1 = (q.inv() * sciR.from_matrix(gt_R1)).magnitude()
            err2 = (q.inv() * sciR.from_matrix(gt_R2)).magnitude()
            angs.append(min(err1, err2))

        rotation_errors.append(np.mean(angs))
        rotation_std_errors.append(np.std(angs))

        N = rots.shape[0]
        pairwise_angles = []
        for i in range(N):
            for j in range(i+1, N):
                R1 = sciR.from_matrix(rots[i])
                R2 = sciR.from_matrix(rots[j])
                relative = R1.inv() * R2
                pairwise_angles.append(relative.magnitude())
        if pairwise_angles:
            spread_list.append(np.mean(pairwise_angles))
            counts, _ = np.histogram(pairwise_angles, bins=30)
            p = counts.astype(float) / counts.sum()
            p = p[p > 0]
        else:
            spread_list.append(0.0)

        R_bar = np.mean(rots, axis=0)
        # orth_dev = np.linalg.norm(R_bar.T @ R_bar - np.eye(3), ord='fro')
        # det_dev = abs(np.linalg.det(R_bar) - 1.0)
        # orth_dev_list.append(orth_dev)
        # det_dev_list.append(det_dev)

        # ---- NEW METRICS ----
        # Compute mean rotation
        R_bar_so3 = mean_rotation_SVD(rots)

        # 1. Credible region radius (SO(3) ball, 95%)
        r_alpha, _ = credible_region_radius(rots, R_bar_so3, alpha=0.95)
        credible_region_radii.append(r_alpha)
        #  -- empirical coverage: does GT lie inside that ball? --
        # account for 180° symmetry
        d1 = geodesic_distance(gt_R1, R_bar_so3)
        d2 = geodesic_distance(gt_R2, R_bar_so3)
        covered = (min(d1, d2) <= r_alpha)
        credible_region_coverages.append(covered)

        # # 2. EAAD
        # eaad_val = eaad(rots, R_bar_so3)
        # eaad_list.append(eaad_val)

        # 3. Negative Log Likelihood (NLL) 
        eps = 1e-8
        nll_values = []
        for preds_t, gt_t in zip(all_preds_t, all_gts_t):
            mu = np.mean(preds_t, axis=0)
            var = np.var(preds_t, axis=0) + eps  # avoid log(0)
            # per-dimension Gaussian NLL
            nll = 0.5 * (np.log(var) + ((gt_t - mu) ** 2) / var)
            nll_values.append(np.mean(nll))      # average over x,y,z
        mean_nll = float(np.mean(nll_values))
        std_nll  = float(np.std(nll_values))

    # ----------- Classic metrics summary -----------
    print("Evaluated samples: " + str(len(eTE_list)))
    # print("Better" + str(counter_better))
    # print("Worse" + str(counter_worse))
    print(f'MEAN eTE {mean(eTE_list)}, eRE: {mean(eRE_list)}, eGD: {mean(eGD_list)}')
    print(f'STD eTE {np.std(eTE_list)}, eRE: {np.std(eRE_list)}, eGD: {np.std(eGD_list)}')
    print(f'MEDIAN eTE {median(eTE_list)}, eRE: {median(eRE_list)}, eGD: {median(eGD_list)}')
    print(f'MIN eTE {min(eTE_list)}, eRE: {min(eRE_list)}, eGD: {min(eGD_list)}')
    print(f'MAX eTE {max(eTE_list)}, eRE: {max(eRE_list)}, eGD: {max(eGD_list)}')

    # ----------- Uncertainty summary -----------
    all_preds_t_stack = np.concatenate(all_preds_t, axis=0)
    all_gts_t_stack = np.stack(all_gts_t)
    n_samples = len(all_gts_t)
    mean_preds_per_sample = np.stack([np.mean(p, axis=0) for p in all_preds_t])
    std_preds_per_sample = np.stack([np.std(p, axis=0) for p in all_preds_t])

    avg_loss = np.mean(np.abs(mean_preds_per_sample - all_gts_t_stack))
    avg_var_t = np.mean(std_preds_per_sample, axis=0)
    avg_mean_t = np.mean(mean_preds_per_sample, axis=0)

    print("\n=== Translation Uncertainty Summary ===")
    print("Translation Uncertainty:")
    for j in range(3):
        print(f" t[{j}]: {avg_mean_t[j]:.4f} ± {avg_var_t[j]:.4f}")
    print("Coverage (GT/Pred):")
    for j in range(3):
        print(f" t[{j}]: {coverage_t[j] / n_samples:.4f} / {coverage_pred_t[j] / n_samples:.4f}")
    print()

    print("Negative Log Likelihood Score (NLL) Lower is better")
    print(f" Translation NLL: {mean_nll:.4f} ± {std_nll:.4f}")

    print("Expected Calibration Error (ECE) Lower is better")
    ece = compute_ece_translation(all_preds_t, all_gts_t, n_bins=10)
    print(f" Translation ECE: {ece:.4f}")
    ece_macro, ece_xyz = compute_ece_translation_perdim(all_preds_t, all_gts_t, n_bins=10)
    print(f" Translation ECE (macro): {ece_macro:.4f}  | per-dim: x={ece_xyz[0]:.4f}, y={ece_xyz[1]:.4f}, z={ece_xyz[2]:.4f}")
    print(f" Average |error|: {avg_loss:.4f}")
    print("Sparpness: (lower is better)")
    sharp_vec, sharp_dims = compute_sharpness_translation(all_preds_t)
    print(f" Translation Sharpness: {sharp_vec:.4f} mm | per-axis: x={sharp_dims[0]:.4f}, y={sharp_dims[1]:.4f}, z={sharp_dims[2]:.4f}")

    print("\n=== Rotation Uncertainty Summary ===")

    print("Rotation Error Summary:")
    mean_ang_err = np.mean(rotation_errors)
    std_ang_err = np.mean(rotation_std_errors)
    print(f" Mean Angular Error: {mean_ang_err:.4f} rad / {np.degrees(mean_ang_err):.2f}°")
    print(f" Std Angular Error:  {std_ang_err:.4f} rad / {np.degrees(std_ang_err):.2f}°")
    print()


    nll_R = compute_nll_rotation(all_preds_R, all_gts_R)
    ece_R = compute_ece_rotation(all_preds_R, all_gts_R)
    sharp_R = compute_sharpness_rotation(all_preds_R, all_gts_R)

    print(f"Rotation NLL, using Gaussian on Error angles: {nll_R:.4f}")
    print(f"Rotation ECE: {ece_R:.4f} rad / {np.degrees(ece_R):.2f}°")
    print(f"Rotation Sharpness: {sharp_R:.4f} rad / {np.degrees(sharp_R):.2f}°")



    # print("Rotation Uncertainty Metrics:")
    # print(f" Mean Sample Spread:          {np.mean(spread_list):.4f} rad / {np.degrees(np.mean(spread_list)):.2f}°")
    # print(f" Mean Δ_orth: {np.mean(orth_dev_list):.6f}")
    # print(f" Mean Δ_det:   {np.mean(det_dev_list):.6f}")
    # print()

    # ---- Print new SO(3) metrics ----
    print("SO(3) Rotational Uncertainty Metrics:")
    print(f" Mean 95% Credible Region Radius: {np.mean(credible_region_radii):.4f} rad / {np.degrees(np.mean(credible_region_radii)):.2f}°")
    print(f" Mean Empirical Coverage (should be ~0.95): {np.mean(credible_region_coverages):.3f}")
    # print(f"Mean EAAD: {np.mean(eaad_list):.4f} rad / {np.degrees(np.mean(eaad_list)):.2f}°")

    # --- UCS for Translation ---
    ucs_t_macro, ucs_t_dims = compute_ucs_translation(all_preds_t, all_gts_t, p_grid=np.linspace(0.1,1.0,10))
    print("\nUCS (Translation) 0..1 ↑ (higher is better)")
    print(f" Translation UCS (macro): {ucs_t_macro:.3f} | per-axis: x={ucs_t_dims[0]:.3f}, y={ucs_t_dims[1]:.3f}, z={ucs_t_dims[2]:.3f}")
    # --- UCS for Rotation: geodesic angle (single scalar) ---
    ucs_r_angle = compute_ucs_rotation_geodesic(all_preds_R, all_gts_R, p_grid=np.linspace(0.1,1.0,10))
    print("\nUCS (Rotation, geodesic angle) 0..1 ↑")
    print(f" Rotation UCS (angle): {ucs_r_angle:.3f}")

    all_kappa_means_arr = np.stack(all_kappa_means)

    print("\n=== KAPPA PLACEHOLDER STATS ===")
    print(f"Mean κx κy κz: {np.mean(all_kappa_means_arr, axis=0)}")
    print(f"Std  κx κy κz: {np.std(all_kappa_means_arr, axis=0)}")

    nlls = []
    for r, k in zip(rots, kappas):
        nlls.append(matrix_fisher_nll(r, gt_R1, k))

    rotation_nll = float(np.mean(nlls))
    print(f"Mean Rotation NLL (Matrix-Fisher): {rotation_nll:.4f}")

    print("\n=== END ===")

        # ---------------- JSON EXPORT ----------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    metadata = {
        "timestamp": timestamp,
        "modifications": args.modifications,
        "mc_samples": args.mc_samples if args.modifications not in ["none", "ensemble"] else None,
        "model_name": Path(args.path).name,
        "dataset": Path(args.path).name,
        "num_samples": len(eTE_list),
    }

    results = {
        "pose_metrics": {
            "eTE": {
                "mean": mean(eTE_list),
                "std": float(np.std(eTE_list)),
                "median": median(eTE_list),
                "min": min(eTE_list),
                "max": max(eTE_list),
            },
            "eRE": {
                "mean": mean(eRE_list),
                "std": float(np.std(eRE_list)),
                "median": median(eRE_list),
                "min": min(eRE_list),
                "max": max(eRE_list),
            },
            "eGD": {
                "mean": mean(eGD_list),
                "std": float(np.std(eGD_list)),
                "median": median(eGD_list),
                "min": min(eGD_list),
                "max": max(eGD_list),
            },
        },

        "translation": {
            "nll": {
                "mean": mean_nll,
                "std": std_nll,
            },
            "ece": {
                "scalar": ece,
                "macro": ece_macro,
                "per_dim": list(ece_xyz),
            },
            "sharpness": {
                "vector": sharp_vec,
                "per_dim": sharp_dims.tolist(),
            },
            "coverage": {
                "gt": (coverage_t / n_samples).tolist(),
                "pred": (coverage_pred_t / n_samples).tolist(),
            },
        },

        "rotation": {
            "mean_error": {
                "rad": mean_ang_err,
                "deg": float(np.degrees(mean_ang_err)),
            },
            "nll on angle errors": nll_R,
            "ece": {
                "rad": ece_R,
                "deg": float(np.degrees(ece_R)),
            },
            "sharpness": {
                "rad": sharp_R,
                "deg": float(np.degrees(sharp_R)),
            },
            "credible_region": {
                "radius_rad": float(np.mean(credible_region_radii)),
                "coverage": float(np.mean(credible_region_coverages)),
            },
        },

        "ucs": {
            "translation": {
                "macro": ucs_t_macro,
                "per_dim": ucs_t_dims,
            },
            "rotation_geodesic": ucs_r_angle,
        },
        "aleatoric_uncertainty": {
        "kappa_mean_xyz": np.mean(all_kappa_means_arr, axis=0).tolist(),
        "kappa_std_xyz": np.std(all_kappa_means_arr, axis=0).tolist(),
        "note": "Placeholders only. Metrics not yet computed."
        }

    }

    final_json = {
        "meta": metadata,
        "results": results,
    }

    eval_dir = Path.home() / "thesis" / "eval_results"
    eval_dir.mkdir(parents=True, exist_ok=True)

    out_file = eval_dir / f"evaluation_{timestamp}.json"

    with open(out_file, "w") as f:
        json.dump(final_json, f, indent=2)

    print(f"\n✔ JSON results saved to: {out_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Path to dataset root folder.')
    parser.add_argument('-mod', '--modifications', type=str, default='none', help='Modifications to the model (e.g., mc_dropout)')
    parser.add_argument('-mc', '--mc_samples', type=int, default=30, help='Number of MC samples to average over (for MC dropout)')
    args = parser.parse_args()
    evaluate(args)
