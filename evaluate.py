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
        pr_RICP = np.copy(pr_R)
        pr_tICP = np.copy(pr_t)
        icp_file = 'icp_scan_' + number + '.txt'
        if os.path.isfile(os.path.join(path, icp_file)):
            pr_RICP, pr_tICP = read_icp_file(os.path.join(path, icp_file))
            pr_44R = [[pr_R[0][0], pr_R[0][1], pr_R[0][2], pr_t[0]],
                      [pr_R[1][0], pr_R[1][1], pr_R[1][2], pr_t[1]],
                      [pr_R[2][0], pr_R[2][1], pr_R[2][2], pr_t[2]],
                      [0.0, 0.0, 0.0, 1.0]]
            pr_44I = [[pr_RICP[0][0], pr_RICP[0][1], pr_RICP[0][2], pr_tICP[0]],
                      [pr_RICP[1][0], pr_RICP[1][1], pr_RICP[1][2], pr_tICP[1]],
                      [pr_RICP[2][0], pr_RICP[2][1], pr_RICP[2][2], pr_tICP[2]],
                      [0.0, 0.0, 0.0, 1.0]]
            pr_44F = np.matmul(pr_44I, pr_44R)
            pr_RICP = pr_44F[:3, :3]
            pr_tICP = pr_44F[:3, 3]
            write_refined_file(path, number, pr_44F)

        if min(calculate_eRE(gt_R1, pr_R), calculate_eRE(gt_R2, pr_R)) > min(calculate_eRE(gt_R1, pr_RICP), calculate_eRE(gt_R2, pr_RICP)) and \
                calculate_eTE(gt_t, pr_t) > calculate_eTE(gt_t, pr_tICP):
            print('BETTER: ' + path + '/' + gt_file)
            counter_better += 1
        else:
            print('WORSE: ' + path + '/' + gt_file)
            counter_worse += 1

        eRE_list_icp.append(min(calculate_eRE(gt_R1, pr_RICP), calculate_eRE(gt_R2, pr_RICP)))
        eTE_list_icp.append(calculate_eTE(gt_t, pr_tICP))
        eGD_list_icp.append(min(calculate_eGD(gt_R1, pr_RICP), calculate_eGD(gt_R2, pr_RICP)))

        all_preds_t.append(pts_arr)
        all_preds_R.append(rots)
        all_gts_t.append(gt_t)

        for j in range(3):
            lo = np.percentile(pts_arr[:, j], 2.5)
            hi = np.percentile(pts_arr[:, j], 97.5)
            if lo <= gt_t[j] <= hi:
                coverage_t[j] += 1
            if lo <= mean_t[j] <= hi:
                coverage_pred_t[j] += 1

        # Rotation uncertainty metrics (classic)
        angs = [ (sciR.from_matrix(r).inv() * sciR.from_matrix(gt_R1)).magnitude() for r in rots ]
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
        orth_dev = np.linalg.norm(R_bar.T @ R_bar - np.eye(3), ord='fro')
        det_dev = abs(np.linalg.det(R_bar) - 1.0)
        orth_dev_list.append(orth_dev)
        det_dev_list.append(det_dev)

        # ---- NEW METRICS ----
        # Compute mean rotation
        R_bar_so3 = mean_rotation_SVD(rots)

        # 1. Credible region radius (SO(3) ball, 95%)
        r_alpha, prop_in_region = credible_region_radius(rots, R_bar_so3, alpha=0.95)
        credible_region_radii.append(r_alpha)
        credible_region_coverages.append(prop_in_region)

        # 2. EAAD
        eaad_val = eaad(rots, R_bar_so3)
        eaad_list.append(eaad_val)

    # ----------- Classic metrics summary -----------
    print("Evaluated samples: " + str(len(eTE_list)))
    # print("Better" + str(counter_better))
    # print("Worse" + str(counter_worse))
    print(f'MEAN eTE {mean(eTE_list)}, eRE: {mean(eRE_list)}, eGD: {mean(eGD_list)}')
    print(f'STD eTE {np.std(eTE_list)}, eRE: {np.std(eRE_list)}, eGD: {np.std(eGD_list)}')
    print(f'MEDIAN eTE {median(eTE_list)}, eRE: {median(eRE_list)}, eGD: {median(eGD_list)}')
    # print(f'MIN eTE {min(eTE_list)}, eRE: {min(eRE_list)}, eGD: {min(eGD_list)}')
    # print(f'MAX eTE {max(eTE_list)}, eRE: {max(eRE_list)}, eGD: {max(eGD_list)}')

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

    print("\n=== Uncertainty Summary ===")
    print("Translation Uncertainty:")
    for j in range(3):
        print(f" t[{j}]: {avg_mean_t[j]:.4f} ± {avg_var_t[j]:.4f}")
    print(" Coverage (GT/Pred):")
    for j in range(3):
        print(f"  t[{j}]: {coverage_t[j] / n_samples:.4f} / {coverage_pred_t[j] / n_samples:.4f}")
    print()

    print("Rotation Error Summary:")
    mean_ang_err = np.mean(rotation_errors)
    std_ang_err = np.mean(rotation_std_errors)
    print(f" Mean Angular Error: {mean_ang_err:.4f} rad / {np.degrees(mean_ang_err):.2f}°")
    print(f" Std Angular Error:  {std_ang_err:.4f} rad / {np.degrees(std_ang_err):.2f}°")
    print()

    print("Rotation Uncertainty Metrics:")
    print(f" Mean Sample Spread:          {np.mean(spread_list):.4f} rad / {np.degrees(np.mean(spread_list)):.2f}°")
    print(f" Mean Δ_orth: {np.mean(orth_dev_list):.6f}")
    print(f" Mean Δ_det:   {np.mean(det_dev_list):.6f}")

    # ---- Print new SO(3) metrics ----
    print("\n=== SO(3) Rotational Uncertainty Metrics ===")
    print(f"Mean 95% Credible Region Radius: {np.mean(credible_region_radii):.4f} rad / {np.degrees(np.mean(credible_region_radii)):.2f}°")
    print(f"Mean Empirical Coverage (should be ~0.95): {np.mean(credible_region_coverages):.3f}")
    print(f"Mean EAAD: {np.mean(eaad_list):.4f} rad / {np.degrees(np.mean(eaad_list)):.2f}°")
    print("\n=== END ===")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Path to dataset root folder.')
    parser.add_argument('-mod', '--modifications', type=str, default='none', help='Modifications to the model (e.g., mc_dropout)')
    parser.add_argument('-mc', '--mc_samples', type=int, default=10, help='Number of MC samples to average over (for MC dropout)')
    args = parser.parse_args()
    evaluate(args)
