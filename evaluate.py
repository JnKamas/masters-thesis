import argparse
import glob
import os
import numpy as np
from statistics import mean, median
from scipy.linalg import logm, svd
from scipy.spatial.transform import Rotation as sciR
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def calculate_eTE(gt_t, pr_t):
    return np.linalg.norm((pr_t - gt_t), ord=2) / 10

def calculate_eRE(gt_R, pr_R):
    numerator = np.trace(np.matmul(gt_R, np.linalg.inv(pr_R))) - 1
    numerator = np.clip(numerator, -2, 2)
    return np.arccos(numerator / 2)

def calculate_eGD(gt_R, pr_R):
    argument = logm(np.matmul(gt_R, np.transpose(pr_R)))
    numerator = np.linalg.norm(argument, ord='fro')
    return numerator / (2 ** .5)

def read_transform_file(file):
    with open(file, 'r') as tfile:
        P = tfile.readline().strip().split(' ')
        R = np.array([[float(P[0]), float(P[4]), float(P[8])],
                    [float(P[1]), float(P[5]), float(P[9])],
                    [float(P[2]), float(P[6]), float(P[10])]])
        t = np.array([float(P[12]), float(P[13]), float(P[14])])
        return R, t

def read_icp_file(file):
    with open(file, 'r') as tfile:
        P = tfile.read().strip().replace('\n', ', ').split(', ')
        R = np.array([[float(P[0]), float(P[4]), float(P[8])],
                    [float(P[1]), float(P[5]), float(P[9])],
                    [float(P[2]), float(P[6]), float(P[10])]])
        t = np.array([float(P[12]), float(P[13]), float(P[14])])
        return R, t

def write_refined_file(path, number, R44):
    with open(path + '/refined_scan_' + number + '.txt', 'w') as rf:
        print(f'{R44[0][0]} {R44[1][0]} {R44[2][0]} 0.0 '
            f'{R44[0][1]} {R44[1][1]} {R44[2][1]} 0.0 '
            f'{R44[0][2]} {R44[1][2]} {R44[2][2]} 0.0 '
            f'{R44[0][3]} {R44[1][3]} {R44[2][3]} 1.0', file=rf)

def get_mc_predictions(path, number, mc_samples):
    Rs, ts = [], []
    for i in range(mc_samples):
        fname = f'prediction{i}_scan_{number}.txt'
        fpath = os.path.join(path, fname)
        if not os.path.isfile(fpath):
            return None, None
        R, t = read_transform_file(fpath)
        Rs.append(R)
        ts.append(t)
    return np.stack(Rs), np.stack(ts)

def mean_rotation_SVD(Rs):
    M = np.mean(Rs, axis=0)
    U, _, Vt = np.linalg.svd(M)
    R_mean = np.dot(U, Vt)
    if np.linalg.det(R_mean) < 0:
        U[:, -1] *= -1
        R_mean = np.dot(U, Vt)
    return R_mean

def compute_ece_translation(all_preds_t, all_gts_t, n_bins=10):
    """
    Computes Expected Calibration Error (ECE) for translation uncertainty.
    all_preds_t : list of np.arrays, each of shape [K, 3] (MC samples for one item)
    all_gts_t   : list of np.arrays, each of shape [3] (ground-truth translation)
    n_bins      : number of bins for uncertainty partitioning
    """
    sigma_list = []
    error_list = []

    for preds_t, gt_t in zip(all_preds_t, all_gts_t):
        mu = np.mean(preds_t, axis=0)
        sigma = np.mean(np.std(preds_t, axis=0))  # average std across 3 dims
        error = np.linalg.norm(mu - gt_t)         # L2 translation error (mm)
        sigma_list.append(sigma)
        error_list.append(error)

    sigma_list = np.array(sigma_list)
    error_list = np.array(error_list)

    # Bin edges between min and max predicted std
    bin_edges = np.linspace(np.min(sigma_list), np.max(sigma_list), n_bins + 1)
    ece = 0.0
    N = len(sigma_list)

    for i in range(n_bins):
        # indices of samples within this bin
        mask = (sigma_list >= bin_edges[i]) & (sigma_list < bin_edges[i + 1])
        n_b = np.sum(mask)
        if n_b == 0:
            continue

        mean_sigma = np.mean(sigma_list[mask])
        mean_error = np.mean(error_list[mask])
        ece += (n_b / N) * np.abs(mean_error - mean_sigma)

    return ece
def compute_ece_translation_perdim(all_preds_t, all_gts_t, n_bins=10):
    """
    Regression ECE for translation: per-dimension calibration.
    Binning by σ quantiles. Compares mean |error| to mean σ in each bin.
    Returns macro-average over x,y,z and also per-dim values.
    """
    N = len(all_preds_t)
    preds_mu = np.stack([p.mean(axis=0) for p in all_preds_t])   # [N,3]
    preds_std = np.stack([p.std(axis=0)  for p in all_preds_t])  # [N,3]
    gts = np.stack(all_gts_t)                                    # [N,3]
    abs_err = np.abs(preds_mu - gts)                             # [N,3]

    ece_dims = []
    for d in range(3):
        sigma_d = preds_std[:, d]
        err_d   = abs_err[:, d]

        # Quantile bins to avoid empty bins
        q = np.linspace(0, 1, n_bins+1)
        edges = np.quantile(sigma_d, q)
        # Ensure strictly increasing (handle ties)
        edges = np.unique(edges)
        if len(edges) < 2:
            # no spread at all => cannot calibrate
            ece_dims.append(float(np.mean(np.abs(err_d - sigma_d))))
            continue

        ece_d = 0.0
        Ntot = len(sigma_d)
        for i in range(len(edges)-1):
            lo, hi = edges[i], edges[i+1]
            # include right edge on last bin
            if i == len(edges)-2:
                mask = (sigma_d >= lo) & (sigma_d <= hi)
            else:
                mask = (sigma_d >= lo) & (sigma_d <  hi)
            nb = mask.sum()
            if nb == 0: 
                continue
            ece_d += (nb / Ntot) * abs(err_d[mask].mean() - sigma_d[mask].mean())
        ece_dims.append(float(ece_d))

    return float(np.mean(ece_dims)), ece_dims  # macro-average, [x,y,z]

def compute_sharpness_translation(all_preds_t):
    """
    Computes the average predicted uncertainty magnitude (Sharpness) for translation.
    all_preds_t: list of np.arrays, each [K, 3] (MC dropout samples for one item)
    Returns: scalar (mean sharpness in mm), and per-dimension values
    """
    # Standard deviation per sample, per axis
    stds = np.stack([np.std(p, axis=0) for p in all_preds_t])  # [N, 3]
    # Mean L2-norm of stds (vector sharpness)
    sharpness_vec = np.mean(np.linalg.norm(stds, axis=1))
    # Mean per-dimension sharpness (macro-style)
    sharpness_dims = np.mean(stds, axis=0)
    return float(sharpness_vec), sharpness_dims

def compute_sharpness_rotation(all_preds_R, all_gts_R):
    """
    Computes average predicted rotation uncertainty (radians).
    all_preds_R: list of np.arrays, each [K, 3x3]
    all_gts_R: list of 3x3 GT rotations
    """
    from scipy.spatial.transform import Rotation as sciR
    sharp_list = []
    for Rs, Rgt in zip(all_preds_R, all_gts_R):
        mean_R = mean_rotation_SVD(Rs)
        errs = [np.arccos(np.clip((np.trace(Rgt.T @ R) - 1) / 2, -1, 1)) for R in Rs]
        sharp_list.append(np.std(errs))
    return float(np.mean(sharp_list))
    
def compute_ece_rotation(all_preds_R, all_gts_R, n_bins=10):
    from scipy.spatial.transform import Rotation as sciR
    errs, sigmas = [], []
    for Rs, Rgt in zip(all_preds_R, all_gts_R):
        mean_R = mean_rotation_SVD(Rs)
        eR = np.arccos(np.clip((np.trace(Rgt.T @ mean_R) - 1) / 2, -1, 1))
        errs.append(eR)
        sample_errs = [np.arccos(np.clip((np.trace(Rgt.T @ R) - 1) / 2, -1, 1)) for R in Rs]
        sigmas.append(np.std(sample_errs))
    errs, sigmas = np.array(errs), np.array(sigmas)
    # binning
    edges = np.quantile(sigmas, np.linspace(0,1,n_bins+1))
    edges = np.unique(edges)
    ece, N = 0, len(sigmas)
    for i in range(len(edges)-1):
        lo, hi = edges[i], edges[i+1]
        mask = (sigmas >= lo) & (sigmas < hi if i < len(edges)-2 else sigmas <= hi)
        nb = mask.sum()
        if nb == 0: continue
        ece += (nb/N) * abs(errs[mask].mean() - sigmas[mask].mean())
    return float(ece)

def compute_nll_rotation(all_preds_R, all_gts_R):
    from scipy.spatial.transform import Rotation as sciR
    nlls = []
    for Rs, Rgt in zip(all_preds_R, all_gts_R):
        mean_R = mean_rotation_SVD(Rs)
        eR_mean = np.arccos(np.clip((np.trace(Rgt.T @ mean_R) - 1) / 2, -1, 1))
        eR_samples = [np.arccos(np.clip((np.trace(Rgt.T @ R) - 1) / 2, -1, 1)) for R in Rs]
        mu, sigma = np.mean(eR_samples), np.std(eR_samples) + 1e-8
        nll = 0.5 * (((eR_mean - mu)**2) / (sigma**2) + np.log(sigma**2) + np.log(2*np.pi))
        nlls.append(nll)
    return np.mean(nlls)


# ---- SO(3) Metrics Helper Functions ----

def geodesic_distance(Ra, Rb):
    return np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1.0, 1.0))

def credible_region_radius(Rs, R_bar, alpha=0.95):
    distances = np.array([geodesic_distance(R, R_bar) for R in Rs])
    r_alpha = np.percentile(distances, alpha * 100)
    prop_in_region = np.mean(distances <= r_alpha)
    return r_alpha, prop_in_region

def eaad(Rs, R_bar):
    deviations = np.array([geodesic_distance(R_bar, R) for R in Rs])
    return deviations.mean()

def evaluate(args):
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

        pr_file = number + '.txt'
        if not os.path.isfile(os.path.join(path, pr_file)):
            if os.path.isfile(os.path.join(path, 'prediction_scan_' + number + '.txt')):
                pr_file = 'prediction_scan_' + number + '.txt'
            else:
                print("Prediction file not found for " + file)
                continue

        if args.modifications == "mc_dropout" or args.modifications == "bayesian":
            Rs, ts = get_mc_predictions(path, number, args.mc_samples)
            if Rs is None:
                print(f"Some MC samples missing for {number}, skipping.")
                continue
            mean_t = np.mean(ts, axis=0)
            std_t = np.std(ts, axis=0)
            pr_t = mean_t
            pr_R = mean_rotation_SVD(Rs)
            pts_arr = ts
            rots = Rs
        else:
            pr_R, pr_t = read_transform_file(os.path.join(path, pr_file))
            pts_arr = np.expand_dims(pr_t, 0)
            rots = np.expand_dims(pr_R, 0)
            mean_t = pr_t
            std_t = np.zeros_like(pr_t)

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

        # ---- ICP metrics ----
        # pr_RICP = np.zeros_like(pr_R)
        # pr_tICP = np.zeros_like(pr_t)
        # icp_file = 'icp_scan_' + number + '.txt'
        # if os.path.isfile(os.path.join(path, icp_file)):
        #     pr_RICP, pr_tICP = read_icp_file(os.path.join(path, icp_file))
        #     pr_44R = [[pr_R[0][0], pr_R[0][1], pr_R[0][2], pr_t[0]],
        #               [pr_R[1][0], pr_R[1][1], pr_R[1][2], pr_t[1]],
        #               [pr_R[2][0], pr_R[2][1], pr_R[2][2], pr_t[2]],
        #               [0.0, 0.0, 0.0, 1.0]]
        #     pr_44I = [[pr_RICP[0][0], pr_RICP[0][1], pr_RICP[0][2], pr_tICP[0]],
        #               [pr_RICP[1][0], pr_RICP[1][1], pr_RICP[1][2], pr_tICP[1]],
        #               [pr_RICP[2][0], pr_RICP[2][1], pr_RICP[2][2], pr_tICP[2]],
        #               [0.0, 0.0, 0.0, 1.0]]
        #     pr_44F = np.matmul(pr_44I, pr_44R)
        #     pr_RICP = pr_44F[:3, :3]
        #     pr_tICP = pr_44F[:3, 3]
        #     write_refined_file(path, number, pr_44F)

        # if min(calculate_eRE(gt_R1, pr_R), calculate_eRE(gt_R2, pr_R)) > min(calculate_eRE(gt_R1, pr_RICP), calculate_eRE(gt_R2, pr_RICP)) and \
        #     calculate_eTE(gt_t, pr_t) > calculate_eTE(gt_t, pr_tICP):
        #     print('BETTER: ' + path + '/' + gt_file)
        #     counter_better += 1
        # else:
        #     print('WORSE: ' + path + '/' + gt_file)
        #     counter_worse += 1

        # print(f"eTE BEFORE: {calculate_eTE(gt_t, pr_t)} eTE AFTER: {calculate_eTE(gt_t, pr_tICP)}\n GT t: {gt_t}, Pred t: {pr_t}, ICP t: {pr_tICP}")

        # eRE_list_icp.append(min(calculate_eRE(gt_R1, pr_RICP), calculate_eRE(gt_R2, pr_RICP)))
        # eTE_list_icp.append(calculate_eTE(gt_t, pr_tICP))
        # eGD_list_icp.append(min(calculate_eGD(gt_R1, pr_RICP), calculate_eGD(gt_R2, pr_RICP)))

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

    # print('AFTER ICP')
    # print(f'MEAN eTE {mean(eTE_list_icp)}, eRE: {mean(eRE_list_icp)}, eGD: {mean(eGD_list_icp)}')
    # print(f'STD eTE {np.std(eTE_list_icp)}, eRE: {np.std(eRE_list_icp)}, eGD: {np.std(eGD_list_icp)}')
    # print(f'MEDIAN eTE {median(eTE_list_icp)}, eRE: {median(eRE_list_icp)}, eGD: {median(eGD_list_icp)}')
    # print(f'MIN eTE {min(eTE_list_icp)}, eRE: {min(eRE_list_icp)}, eGD: {min(eGD_list_icp)}')
    # print(f'MAX eTE {max(eTE_list_icp)}, eRE: {max(eRE_list_icp)}, eGD: {max(eGD_list_icp)}')

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

    print(f"Rotation NLL: {nll_R:.4f}")
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



    print("\n=== END ===")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Path to dataset root folder.')
    parser.add_argument('-mod', '--modifications', type=str, default='none', help='Modifications to the model (e.g., mc_dropout)')
    parser.add_argument('-mc', '--mc_samples', type=int, default=30, help='Number of MC samples to average over (for MC dropout)')
    args = parser.parse_args()
    evaluate(args)
