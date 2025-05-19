import os
import itertools
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader

from dataset import Dataset
from network import parse_command_line, load_model


def enable_dropout(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def gram_schmidt_to_rotation_matrix(vz, vy):
    uz = vz / np.linalg.norm(vz)
    wy = vy - np.dot(vy, uz) * uz
    uy = wy / np.linalg.norm(wy)
    ux = np.cross(uy, uz)
    return np.stack([ux, uy, uz], axis=-1)


def compute_rotation_spread(rotation_matrices):
    N = rotation_matrices.shape[0]
    angles = []
    for i, j in itertools.combinations(range(N), 2):
        R1 = R.from_matrix(rotation_matrices[i])
        R2 = R.from_matrix(rotation_matrices[j])
        relative = R1.inv() * R2
        angles.append(relative.magnitude())
    mean_ang = np.mean(angles) if angles else 0.0
    return angles, mean_ang


def estimate_rotation_entropy(rotation_matrices, bins=30):
    angles = []
    for i, j in itertools.combinations(range(len(rotation_matrices)), 2):
        R1 = R.from_matrix(rotation_matrices[i])
        R2 = R.from_matrix(rotation_matrices[j])
        angles.append((R1.inv() * R2).magnitude())
    if not angles:
        return 0.0
    counts, _ = np.histogram(angles, bins=bins)
    p = counts.astype(float) / counts.sum()
    p = p[p > 0]
    return -np.sum(p * np.log(p))


def plot_translation_uncertainty(pred_samples, gt_values, save_path="uncertainty_translation.png"):
    pred = np.array(pred_samples)
    # gt_values may be list of tensors or ndarrays
    gt_list = []
    for gt in gt_values:
        if hasattr(gt, 'cpu'):
            gt_list.append(gt.cpu().numpy())
        else:
            gt_list.append(gt)
    gt_arr = np.array(gt_list)

    means = pred.mean(axis=0)
    lower = np.percentile(pred, 2.5, axis=0)
    upper = np.percentile(pred, 97.5, axis=0)

    comps = ['t[0]', 't[1]', 't[2]']
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        for s in range(pred.shape[0]):
            ax[i].scatter(range(pred.shape[1]), pred[s, :, i], color='blue', alpha=0.1, s=10)
        ax[i].plot(means[:, i], 'b-', label='Mean')
        ax[i].fill_between(range(means.shape[0]), lower[:, i], upper[:, i], color='blue', alpha=0.3)
        ax[i].scatter(range(gt_arr.shape[0]), gt_arr[:, i], color='red', marker='x', s=50, label='GT')
        ax[i].set_title(f'Uncertainty in {comps[i]}')
        ax[i].legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_rotation_metrics(spread_list, entropy_list, orth_dev_list, det_dev_list,
                          angles_samples=None, angles_between=None,
                          save_path="uncertainty_rotation.png"):
    num = 5
    fig, ax = plt.subplots(num, 1, figsize=(10, 4 * num))
    idx = 0

    # 1) sample spread scatter + mean
    if angles_between is not None:
        means = []
        for si, disp in enumerate(angles_between):
            arr = np.full_like(disp, si)
            ax[idx].scatter(arr, disp, color='red', alpha=0.1, s=10,
                            label='Pairwise Angles' if si == 0 else "")
            means.append(np.mean(disp))
        ax[idx].scatter(range(len(means)), means, color='darkred', marker='x', s=50, label='Mean Pairwise Angle')
    ax[idx].plot(range(len(spread_list)), spread_list, 'g.-', label='Sample Spread')
    ax[idx].set_title('Rotation Spread per Sample')
    ax[idx].set_ylabel('Angle (rad)')
    ax[idx].set_ylim(0, np.pi)
    ax[idx].legend()
    idx += 1

    # 2) entropy
    ax[idx].plot(range(len(entropy_list)), entropy_list, 'm.-', label='Entropy')
    ax[idx].set_title('Rotation Entropy-like Measure')
    ax[idx].set_ylabel('Entropy (nats)')
    ax[idx].legend()
    idx += 1

    # 3) orthogonality deviation
    ax[idx].plot(range(len(orth_dev_list)), orth_dev_list, 'c.-', label='Δ_orth')
    ax[idx].set_title('Ortogonalitná odchýlka (Δ_orth)')
    ax[idx].set_ylabel('‖RᵀR - I‖_F')
    ax[idx].legend()
    idx += 1

    # 4) determinant deviation
    ax[idx].plot(range(len(det_dev_list)), det_dev_list, 'y.-', label='Δ_det')
    ax[idx].set_title('Determinantná odchýlka (Δ_det)')
    ax[idx].set_ylabel('|det(R) - 1|')
    ax[idx].legend()
    idx += 1

    # 5) angular error to GT
    if angles_samples is not None:
        arr = np.array(angles_samples).T
        mean_ang = arr.mean(axis=0)
        low = np.percentile(arr, 2.5, axis=0)
        high = np.percentile(arr, 97.5, axis=0)
        for j in range(arr.shape[0]):
            ax[idx].scatter(range(arr.shape[1]), arr[j], color='blue', alpha=0.1, s=10)
        ax[idx].plot(range(len(mean_ang)), mean_ang, 'b-', label='Mean Angular Error')
        ax[idx].fill_between(range(len(mean_ang)), low, high, color='blue', alpha=0.3, label='95% CI')
        ax[idx].set_title('Rotation Angular Errors to GT')
        ax[idx].set_ylabel('Error (rad)')
        ax[idx].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def mc_infer(args, export_to_folder=False, mc_samples=100):
    model = load_model(args)
    enable_dropout(model)

    dir_path = os.path.dirname(args.path)
    val_dataset = Dataset(args.path, 'val', args.input_width, args.input_height,
                          preload=not args.no_preload)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers)

    total_loss = 0
    count = 0
    coverage_t = np.zeros(3)
    pred_coverage_t = np.zeros(3)
    total_variance_t = np.zeros(3)
    total_mean_t = np.zeros(3)

    rotation_errors = []
    spread_list = []
    entropy_list = []
    orth_dev_list = []
    det_dev_list = []
    angles_samples = []
    pairwise_angles_list = []

    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            xyz = sample['xyz'].cuda()
            gt_transforms = sample['orig_transform']

            # MC forward
            pts, vs, ys = [], [], []
            for _ in range(mc_samples):
                z_hat, y_hat, t_hat = model(xyz)
                pts.append(t_hat.cpu().numpy())
                vs.append(z_hat.cpu().numpy())
                ys.append(y_hat.cpu().numpy())

            pts_arr = np.stack(pts)
            vs_arr  = np.stack(vs)
            ys_arr  = np.stack(ys)

            mean_t = pts_arr.mean(axis=0)
            std_t  = pts_arr.std(axis=0)

            for i in range(mean_t.shape[0]):
                gt = gt_transforms[i].cpu().numpy()
                # translation coverage & stats
                for j in range(3):
                    lo = np.percentile(pts_arr[:, i, j], 2.5)
                    hi = np.percentile(pts_arr[:, i, j], 97.5)
                    if lo <= gt[j, 3] <= hi:
                        coverage_t[j] += 1
                    if lo <= mean_t[i, j] <= hi:
                        pred_coverage_t[j] += 1

                loss = np.mean(np.abs(gt[0:3, 3] - mean_t[i]))
                total_loss += loss
                count += 1
                total_variance_t += std_t[i]
                total_mean_t += mean_t[i]

                # build rotations
                rots = [
                    gram_schmidt_to_rotation_matrix(vs_arr[k, i], ys_arr[k, i])
                    for k in range(mc_samples)
                ]
                # angular errors
                gtR = gt[0:3, 0:3]
                angs = [(R.from_matrix(r).inv() * R.from_matrix(gtR)).magnitude() for r in rots]
                rotation_errors.append(np.mean(angs))
                angles_samples.append(angs)

                # spread & entropy
                _, spread = compute_rotation_spread(np.stack(rots))
                spread_list.append(spread)
                entropy_list.append(estimate_rotation_entropy(np.stack(rots)))
                pairwise_angles_list.append(np.array(_))

                # orth & det dev
                R_bar = np.mean(np.stack(rots), axis=0)
                orth_dev = np.linalg.norm(R_bar.T @ R_bar - np.eye(3), ord='fro')
                det_dev  = abs(np.linalg.det(R_bar) - 1.0)
                orth_dev_list.append(orth_dev)
                det_dev_list.append(det_dev)

                # ---------- printing ----------
                print(40 * "-")
                print(f"Batch {batch_idx} Sample {i} Translation:")
                for j in range(3):
                    print(f" t[{j}] Mean: {mean_t[i, j]:.4f}, Std: {std_t[i, j]:.4f}")
                print(f" Loss (L1): {loss:.6f} (lower=better)")

                mean_ang = np.mean(angs)
                std_ang  = np.std(angs)
                sp       = spread_list[-1]
                ent      = entropy_list[-1]
                o        = orth_dev_list[-1]
                d        = det_dev_list[-1]

                print(f"\nRotation Metrics:")
                print(f"  Mean Angular Error: {mean_ang:.4f} rad / {np.degrees(mean_ang):.2f}°   "
                      "(0=perfect, higher=more error)")
                print(f"  Std Angular Error:  {std_ang:.4f} rad / {np.degrees(std_ang):.2f}°   "
                      "(0=all identical, higher=more scatter)")
                print(f"  Sample Spread:      {sp:.4f} rad / {np.degrees(sp):.2f}°   "
                      "(0=no spread, higher=more uncertainty)")
                print(f"  Entropy-like Measure: {ent:.4f}   "
                      "(range [0, ~3.4]; lower=better confidence)")
                print(f"  Δ_orth:               {o:.6f}   "
                      "(0=perfect orthogonality, higher=more spread)")
                print(f"  Δ_det:                {d:.6f}   "
                      "(0=determinant=1, higher=more spread)")
                print(40 * "-")

            # plot once per batch
            plot_translation_uncertainty(pts_arr, [gt.cpu().numpy()[0:3,3] for gt in gt_transforms])
            plot_rotation_metrics(spread_list, entropy_list,
                                  orth_dev_list, det_dev_list,
                                  angles_samples=angles_samples,
                                  angles_between=pairwise_angles_list)
            #break  # remove if you want all batches

    # ---------- final summaries ----------
    avg_loss    = total_loss / count
    avg_var_t   = total_variance_t / count
    avg_mean_t  = total_mean_t / count
    mean_ang_err = np.mean(rotation_errors)
    std_ang_err  = np.std(rotation_errors)
    mean_spread = np.mean(spread_list)
    mean_entropy = np.mean(entropy_list)
    mean_orth   = np.mean(orth_dev_list)
    mean_det    = np.mean(det_dev_list)

    print(f"\n=== Final Summary ===")
    print(f"Average Loss (L1): {avg_loss:.6f} (lower=better)\n")

    print("Translation Uncertainty:")
    for j in range(3):
        print(f" t[{j}]: {avg_mean_t[j]:.4f} ± {avg_var_t[j]:.4f}  ")
    print(" Coverage (GT/Pred):")
    for j in range(3):
        print(f"  t[{j}]: {coverage_t[j]/count:.4f} / {pred_coverage_t[j]/count:.4f}")
    print()

    print("Rotation Error Summary:")
    print(f" Mean Angular Error: {mean_ang_err:.4f} rad / {np.degrees(mean_ang_err):.2f}°   "
          "(0=perfect alignment, higher=more error)")
    print(f" Std Angular Error:  {std_ang_err:.4f} rad / {np.degrees(std_ang_err):.2f}°   "
          "(0=all identical, higher=more scatter)\n")

    print("Rotation Uncertainty Metrics:")
    print(f" Mean Sample Spread:          {mean_spread:.4f} rad / {np.degrees(mean_spread):.2f}°   "
          "(0=no spread, higher=more uncertainty)")
    print(f" Mean Entropy-like Measure:   {mean_entropy:.4f}  "
          "(range [0, ~3.4]; lower=better confidence)")

    print(f" Mean Δ_orth: {mean_orth:.6f}   "
          "(range [0, ~sqrt(3)]; 0=perfect orthogonality)")
    # print(f" Mean Δ_det:   {mean_det:.6f}   "
    #       "(0=determinant=1, 2=more spread)")

    if export_to_folder:
        os.makedirs(dir_path, exist_ok=True)
        plot_translation_uncertainty(
            pts_arr,
            [gt.cpu().numpy()[0:3,3] for gt in gt_transforms],
            os.path.join(dir_path, "uncertainty_translation.png"))
        plot_rotation_metrics(
            spread_list, entropy_list,
            orth_dev_list, det_dev_list,
            angles_samples=angles_samples,
            angles_between=pairwise_angles_list,
            save_path=os.path.join(dir_path, "uncertainty_rotation.png"))


if __name__ == '__main__':
    args = parse_command_line()
    mc_infer(args)
