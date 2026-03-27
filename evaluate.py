import argparse
import glob
import os
from datetime import datetime
from pathlib import Path
import json
import numpy as np
from statistics import mean, median
from scipy.spatial.transform import Rotation as sciR
from scipy.stats import spearmanr
import warnings
from metrics import *
from network_helpers import parse_command_line

warnings.filterwarnings("ignore", category=RuntimeWarning)

# symmetry 
S = sciR.from_euler('z', np.pi).as_matrix()

def estimate_isotropic_kappa_from_samples(Rs, eps=1e-8):
    R_bar = mean_rotation_SVD(Rs)
    angles = np.array([geodesic_distance(R_bar, R) for R in Rs])
    sigma2 = np.var(angles) + eps
    kappa_iso = np.clip(1.0 / sigma2, 1e-3, 50)
    return float(kappa_iso)

def read_kappa_from_prediction_file(txt_path):
    with open(txt_path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if "# kappa" in line:
            return float(lines[i+1].strip())

    return None

def read_sigma_from_prediction_file(txt_path):
    with open(txt_path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if "# sigma_tx" in line:
            vals = lines[i+1].strip().split()
            return np.array([float(v) for v in vals], dtype=np.float32)

    return None

def get_mc_predictions(path, number, mc_samples):
    Rs, ts, kappas, sigmas = [], [], [], []
    for i in range(mc_samples):
        fname = f'prediction{i}_scan_{number}.txt'
        fpath = os.path.join(path, fname)
        if not os.path.isfile(fpath):
            return None, None, None

        R, t = read_transform_file(fpath)
        kappa = read_kappa_from_prediction_file(fpath)
        sigma = read_sigma_from_prediction_file(fpath)
        if kappa is None:
            kappa = 1.0

        if sigma is None:
            sigma = np.ones(3) * 1e-3
        Rs.append(R)
        ts.append(t)
        kappas.append(kappa)
        sigmas.append(sigma)

    return np.stack(Rs), np.stack(ts), np.stack(kappas), np.stack(sigmas)

def evaluate(args):
    print(args.path)
    gt_files = glob.iglob(args.path + '/**/*.txt', recursive=True)
    good_gt_files = [f for f in gt_files if not any(sub in f for sub in ['bad', 'catas', 'ish', 'pred', 'icp', 'refined']) and any(sub in f for sub in ['scan_', 'gt_'])]

    eTE_list = []
    eRE_list = []

    all_preds_t = []
    all_preds_R = []
    all_gts_t = []
    all_gts_R = []

    all_kappas = []        # list of (N_mc, 1)
    all_kappa_means = []  # per sample

    all_sigmas = []       # list of (N_mc, 3)

    rotation_nll_aleatoric = []
    rotation_nll_epistemic = []

    coverage_t = np.zeros(3)
    coverage_pred_t = np.zeros(3)

    # ---- New Metrics Lists ----
    credible_region_radii = []
    credible_region_coverages = []

    for file in good_gt_files:
        path, gt_file = os.path.split(file)
        index = gt_file.rfind("_")
        number = gt_file[index + 1:-4]

        if args.modifications in {"mc_dropout", "bayesian", "ensemble", "ensemble_mc_dropout"}:
            Rs, ts, kappas, sigmas = get_mc_predictions(path, number, args.mc_samples)   
            if Rs is None:
                print(f"Some samples missing for {number}, skipping.")
                continue
            all_kappas.append(kappas)
            all_kappa_means.append(np.mean(kappas))
            all_sigmas.append(sigmas)

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
            sigma = read_sigma_from_prediction_file(os.path.join(path, pr_file))

            if kappa is None:
                kappa = 1.0

            if sigma is None:
                sigma = np.ones(3) * 1e-3

            Rs = np.expand_dims(pr_R, 0)
            ts = np.expand_dims(pr_t, 0)
            kappas = np.array([kappa])
            sigmas = np.expand_dims(sigma, 0)

            all_kappas.append(kappas)
            all_kappa_means.append(kappa)

            all_sigmas.append(sigmas)

        mean_t = np.mean(ts, axis=0)
        pr_t = mean_t
        pr_R = mean_rotation_SVD(Rs)
        pts_arr = ts
        rots = Rs

        gt_R1, gt_t = read_transform_file(os.path.join(path, gt_file))
        if np.linalg.det(gt_R1) < 0:
            gt_R1[:, 1] *= -1
        if np.linalg.det(pr_R) < 0:
            pr_R[:, 1] *= -1


        # ---- Old metrics (mean only) ----
        eTE_list.append(calculate_eTE(gt_t, mean_t))
        if np.array_equal(pr_R, np.eye(3, 3)):
            eRE_list.append(float("inf"))
        else:
            eRE_list.append(min(calculate_eRE(gt_R1, pr_R), calculate_eRE(gt_R1 @ S, pr_R)))

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
            angs.append(min(geodesic_distance(r, gt_R1), geodesic_distance(r, gt_R1 @ S)))


        # ---- NEW METRICS ----
        # Compute mean rotation
        R_bar_so3 = mean_rotation_SVD(rots)

        # 1. Credible region radius (SO(3) ball, 95%)
        r_alpha, _ = credible_region_radius(rots, R_bar_so3, alpha=0.95)
        credible_region_radii.append(r_alpha)
        #  -- empirical coverage: does GT lie inside that ball? --
        # account for 180° symmetry
        d1 = geodesic_distance(gt_R1, R_bar_so3)
        d2 = geodesic_distance(gt_R1 @ S, R_bar_so3)
        covered = (min(d1, d2) <= r_alpha)
        credible_region_coverages.append(covered)

        # 3. NLL rotation

        # --- Choose correct GT (handle symmetry) ---
        d1 = geodesic_distance(gt_R1, R_bar_so3)
        d2 = geodesic_distance(gt_R1 @ S, R_bar_so3)
        gt_R_best = gt_R1 if d1 <= d2 else (gt_R1 @ S)

        # --- ALEATORIC NLL ---
        if args.use_aleatoric:
            kappa_alea = np.mean(kappas, axis=0)
            nll_alea = matrix_fisher_nll(R_bar_so3, gt_R_best, kappa_alea)
            rotation_nll_aleatoric.append(nll_alea)

        # --- EPISTEMIC NLL ---
        kappa_epi = estimate_isotropic_kappa_from_samples(rots)
        nll_epi = matrix_fisher_nll(R_bar_so3, gt_R_best, kappa_epi)
        rotation_nll_epistemic.append(nll_epi)

    # ----------- Uncertainty summary -----------
    all_preds_t_stack = np.concatenate(all_preds_t, axis=0)
    all_gts_t_stack = np.stack(all_gts_t)
    n_samples = len(all_gts_t)
    mean_preds_per_sample = np.stack([np.mean(p, axis=0) for p in all_preds_t])
    std_preds_per_sample = np.stack([np.std(p, axis=0) for p in all_preds_t])

    avg_loss = np.mean(np.abs(mean_preds_per_sample - all_gts_t_stack))
    avg_var_t = np.mean(std_preds_per_sample, axis=0)
    avg_mean_t = np.mean(mean_preds_per_sample, axis=0)

    # Negative Log Likelihood (NLL) translation
    eps = 1e-8

    nll_total_list = []
    nll_epi_list = []
    nll_alea_list = []

    for preds_t, sigmas_t, gt_t in zip(all_preds_t, all_sigmas, all_gts_t):

        mu = np.mean(preds_t, axis=0)

        # epistemic variance
        var_epi = np.var(preds_t, axis=0)

        # aleatoric variance
        if args.use_aleatoric:
            var_alea = np.mean(sigmas_t**2, axis=0)
        else:
            var_alea = np.zeros_like(var_epi)

        # ---- separate NLLs ----
        nll_epi = 0.5 * (np.log(var_epi + eps) + ((gt_t - mu) ** 2) / (var_epi + eps))
        nll_alea = 0.5 * (np.log(var_alea + eps) + ((gt_t - mu) ** 2) / (var_alea + eps))

        # total variance
        var_total = var_epi + var_alea + eps
        nll_total = 0.5 * (np.log(var_total) + ((gt_t - mu) ** 2) / var_total)

        nll_epi_list.append(np.mean(nll_epi))
        nll_alea_list.append(np.mean(nll_alea))
        nll_total_list.append(np.mean(nll_total))

    mean_nll_total = float(np.mean(nll_total_list))
    mean_nll_trans_epi   = float(np.mean(nll_epi_list))
    mean_nll_trans_alea  = float(np.mean(nll_alea_list))

    crps_t_list = []
    crps_r_list = []

    for preds_t, preds_R, gt_t, gt_R in zip(all_preds_t, all_preds_R, all_gts_t, all_gts_R):

        mu_t = np.mean(preds_t, axis=0)
        sigma_t = np.std(preds_t, axis=0)

        crps_t = crps_translation(mu_t[None, :], sigma_t[None, :], gt_t[None, :])
        crps_t_list.append(crps_t)

        preds_R_arr = np.stack(preds_R)  # [T, 3, 3]
        crps_r = crps_rotation(
            preds_R_arr[:, None, :, :],   # [T,1,3,3]
            gt_R[None, :, :]              # [1,3,3]
        )
        crps_r_list.append(crps_r)

    mean_crps_t = float(np.mean(crps_t_list))
    mean_crps_r = float(np.mean(crps_r_list))


    if all_kappa_means:
        all_kappa_means_arr = np.stack(all_kappa_means)
    else:
        all_kappa_means_arr = np.zeros((1,3))

    mean_nll_rot_alea = float(np.mean(rotation_nll_aleatoric))
    mean_nll_rot_epi  = float(np.mean(rotation_nll_epistemic))

    errors_t = []
    uncertainties_t = []

    for preds_t, gt_t in zip(all_preds_t, all_gts_t):
        preds_t_arr = np.array(preds_t)

        mu = np.mean(preds_t_arr, axis=0)
        sigma = np.std(preds_t_arr, axis=0)

        err = np.linalg.norm(mu - gt_t)
        unc = np.linalg.norm(sigma)

        errors_t.append(err)
        uncertainties_t.append(unc)

    mean_corr_t = float(spearmanr(errors_t, uncertainties_t).correlation)

    errors_r = []
    uncertainties_r = []

    for preds_R, gt_R in zip(all_preds_R, all_gts_R):
        preds_R_arr = np.array(preds_R)

        R_bar = mean_rotation_SVD(preds_R_arr)

        err = min(
            geodesic_distance(R_bar, gt_R),
            geodesic_distance(R_bar, gt_R @ S)
        )

        angles = [geodesic_distance(R_bar, R) for R in preds_R_arr]
        unc = np.std(angles)

        errors_r.append(err)
        uncertainties_r.append(unc)

    mean_corr_r = float(spearmanr(errors_r, uncertainties_r).correlation)

    print("Translation Uncertainty:")
    for j in range(3):
        print(f" t[{j}]: {avg_mean_t[j]:.4f} ± {avg_var_t[j]:.4f}")
    print("Coverage (GT/Pred):")
    for j in range(3):
        print(f" t[{j}]: {coverage_t[j] / n_samples:.4f} / {coverage_pred_t[j] / n_samples:.4f}")
    print()


    print("\n================ 3.1 ACCURACY ================")

    print("\n-- Translation Error (eTE) --")
    print(f" mean:   {mean(eTE_list):.4f}")
    print(f" std:    {np.std(eTE_list):.4f}")
    print(f" median: {median(eTE_list):.4f}")
    print(f" min:    {min(eTE_list):.4f}")
    print(f" max:    {max(eTE_list):.4f}")

    print("\n-- Rotation Error (eRE) --")
    print(f" mean:   {mean(eRE_list):.4f} rad / {np.degrees(mean(eRE_list)):.2f}°")
    print(f" std:    {np.std(eRE_list):.4f}")
    print(f" median: {median(eRE_list):.4f}")
    print(f" min:    {min(eRE_list):.4f}")
    print(f" max:    {max(eRE_list):.4f}")


    print("\n================ 3.2 COVERAGE ================")

    cov_gt = coverage_t / n_samples
    cov_pred = coverage_pred_t / n_samples

    print("Translation Coverage:")
    print(f" GT (min):   {np.min(cov_gt):.4f} | per-dim: {cov_gt}")
    print(f" Pred (min): {np.min(cov_pred):.4f} | per-dim: {cov_pred}")

    print("\nRotation Coverage:")
    print(f" Mean coverage: {np.mean(credible_region_coverages):.4f}")


    print("\n================ 3.3 SHARPNESS ================")

    sharp_vec, sharp_dims = compute_sharpness_translation(all_preds_t)
    sharp_R = compute_sharpness_rotation(all_preds_R, all_gts_R)

    print("Translation:")
    print(f" {sharp_vec:.4f} mm | per-dim: {sharp_dims}")

    print("Rotation:")
    print(f" {sharp_R:.4f} rad / {np.degrees(sharp_R):.2f}°")


    print("\n================ 3.4 NLL =====================")

    print("Translation:")
    print(f" Total:      {mean_nll_total:.4f}")
    print(f" Epistemic:  {mean_nll_trans_epi:.4f}")
    if args.use_aleatoric:
        print(f" Aleatoric:  {mean_nll_trans_alea:.4f}")

    print("\nRotation:")
    print(f" Epistemic:  {mean_nll_rot_epi:.4f}")
    if args.use_aleatoric:
        print(f" Aleatoric:  {mean_nll_rot_alea:.4f}")


    print("\n================ 3.5 CRPS ====================")

    print(f"Translation: {mean_crps_t:.4f}")
    print(f"Rotation:    {mean_crps_r:.4f}")

    print("\n================ 3.6 CORRELATION =============")

    print(f"Translation (error vs uncertainty): {mean_corr_t:.4f}")
    print(f"Rotation (error vs uncertainty):    {mean_corr_r:.4f}")


    print("\n================ END ==================")

    # ---------------- JSON EXPORT ----------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    metadata = {
        "timestamp": timestamp,
        "modifications": args.modifications,
        "mc_samples": args.mc_samples or None,
        "model_name": Path(args.path).name,
        "dataset": Path(args.path).name,
        "num_samples": len(eTE_list),
    }

    def safe(val):
        return val if val is not None else "N/A"

    # ----------- POSE METRICS -----------
    pose_metrics = {
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
    }

    # ----------- EPISTEMIC -----------
    epistemic_metrics = {
        "translation": {
            "coverage": {
                "gt": (coverage_t / n_samples).tolist(),
                "pred": (coverage_pred_t / n_samples).tolist(),
            },
            "sharpness": {
                "vector": safe(sharp_vec),
                "per_dim": safe(sharp_dims.tolist()),
            },
            "nll": safe(mean_nll_trans_epi),
            "crps": safe(mean_crps_t),
            "corr": safe(mean_corr_t),
        },
        "rotation": {
            "coverage": safe(float(np.mean(credible_region_coverages))),
            "sharpness": {
                "rad": safe(sharp_R),
                "deg": safe(float(np.degrees(sharp_R))),
            },
            "nll_matrix_fisher": safe(mean_nll_rot_epi),
            "crps": safe(mean_crps_r),
            "corr": safe(mean_corr_r),
        },
    }

    # ----------- ALEATORIC (optional) -----------
    if args.use_aleatoric:
        aleatoric_metrics = {
            "translation": {
                "nll": safe(mean_nll_trans_alea),
            },
            "rotation": {
                "nll_matrix_fisher": safe(mean_nll_rot_alea),
            },
            "parameters": {
                "kappa_mean": safe(float(np.mean(all_kappa_means_arr))),
                "kappa_std": safe(float(np.std(all_kappa_means_arr))),
            },
        }
    else:
        aleatoric_metrics = "N/A"

    # ----------- FINAL JSON -----------
    final_json = {
        "meta": metadata,
        "results": {
            "pose": pose_metrics,
            "epistemic": epistemic_metrics,
            "aleatoric": aleatoric_metrics,
        },
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
    args = parse_command_line()
    evaluate(args)
