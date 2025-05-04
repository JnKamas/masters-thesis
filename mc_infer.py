import os

import torch
import numpy as np
from dataset import Dataset

from network import Network, parse_command_line, load_model
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R
import itertools

def compute_rotation_spread(rotation_matrices):
    """
    Computes the mean pairwise angular distance between rotation matrices.
    :param rotation_matrices: numpy array of shape (N, 3, 3)
    :return: mean angular spread in radians
    """
    N = rotation_matrices.shape[0]
    angles = []
    for i, j in itertools.combinations(range(N), 2):
        R1 = R.from_matrix(rotation_matrices[i])
        R2 = R.from_matrix(rotation_matrices[j])
        relative_rotation = R1.inv() * R2
        angle = relative_rotation.magnitude()
        angles.append(angle)
    if len(angles) == 0:
        return 0.0
    return np.mean(angles)

def estimate_rotation_entropy(rotation_matrices, bins=30):
    """
    Estimates an entropy-like measure based on pairwise angular distances.
    :param rotation_matrices: numpy array of shape (N, 3, 3)
    :return: scalar entropy proxy (higher = more spread)
    """
    N = rotation_matrices.shape[0]
    angles = []
    for i, j in itertools.combinations(range(N), 2):
        R1 = R.from_matrix(rotation_matrices[i])
        R2 = R.from_matrix(rotation_matrices[j])
        relative_rotation = R1.inv() * R2
        angle = relative_rotation.magnitude()
        angles.append(angle)
    if len(angles) == 0:
        return 0.0
    hist, _ = np.histogram(angles, bins=bins, density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log(hist))
    return entropy


def plot_uncertainty(pred_samples, gt_values, sample_spreads=None, rotation_entropies=None, save_path="uncertainty_plot.png"):
    pred_samples = np.array(pred_samples)
    gt_values = np.array([gt.cpu().numpy() for gt in gt_values])

    if pred_samples.shape[0] == 0:
        print("Error: pred_samples is empty!")
        return

    pred_means = np.mean(pred_samples, axis=0)
    pred_lower = np.percentile(pred_samples, 2.5, axis=0)
    pred_upper = np.percentile(pred_samples, 97.5, axis=0)

    components = ['t[0]', 't[1]', 't[2]']
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    for i in range(3):
        num_samples = pred_samples.shape[1]
        for j in range(pred_samples.shape[0]):
            ax[i].scatter(range(num_samples), pred_samples[j, :, i], color='blue', alpha=0.1, s=10)
        ax[i].plot(range(num_samples), pred_means[:, i], 'b-', label='Mean Prediction')
        ax[i].fill_between(range(num_samples), pred_lower[:, i], pred_upper[:, i], color='blue', alpha=0.3, label='95% CI')
        ax[i].scatter(range(num_samples), gt_values[:, i], color='red', label='GT', marker='x', s=50)
        ax[i].set_title(f'Uncertainty in {components[i]}')
        ax[i].set_xlabel('Sample Index')
        ax[i].set_ylabel('Translation Value')
        ax[i].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved at: {save_path}")

    # Now plot sample spread and entropy if provided
    if sample_spreads is not None and rotation_entropies is not None:
        fig2, ax2 = plt.subplots(2, 1, figsize=(10, 8))
        
        ax2[0].plot(sample_spreads, 'g.-')
        ax2[0].set_title('Sample Spread per Sample')
        ax2[0].set_xlabel('Sample Index')
        ax2[0].set_ylabel('Spread (radians)')
        
        ax2[1].plot(rotation_entropies, 'm.-')
        ax2[1].set_title('Rotation Entropy-like Measure per Sample')
        ax2[1].set_xlabel('Sample Index')
        ax2[1].set_ylabel('Entropy (a.u.)')

        plt.tight_layout()
        plt.savefig(save_path.replace('.png', '_rotation_metrics.png'))
        print(f"Rotation spread/entropy plot saved at: {save_path.replace('.png', '_rotation_metrics.png')}")


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

def mc_infer(args, export_to_folder=False, mc_samples=100):
    model = load_model(args)
    enable_dropout(model)

    dir_path = os.path.dirname(args.path)
    val_dataset = Dataset(args.path, 'val', args.input_width, args.input_height, preload=not args.no_preload)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    np.set_printoptions(suppress=True)

    with torch.no_grad():
        total_loss = 0
        count = 0
        coverage_t = [0, 0, 0]
        pred_coverage_t = [0, 0, 0]
        total_variance_t = [0, 0, 0]
        total_mean_t = [0, 0, 0]
        rotation_errors = []
        rotation_ci_coverage = 0

        means = []
        stds = []
        spread_list = []
        entropy_list = []

        for sample in val_loader:
            xyz = sample['xyz'].cuda()
            gt_transforms = sample['orig_transform']

            pred_ts_list, pred_vs_list, pred_ys_list = [], [], []

            for _ in range(mc_samples):
                pred_zs, pred_ys, pred_ts = model(xyz)
                pred_ts_list.append(pred_ts.cpu().numpy())
                pred_vs_list.append(pred_zs.cpu().numpy())
                pred_ys_list.append(pred_ys.cpu().numpy())

            pred_ts_arr = np.stack(pred_ts_list)
            pred_vs_arr = np.stack(pred_vs_list)
            pred_ys_arr = np.stack(pred_ys_list)

            pred_ts_mean, pred_ts_std = np.mean(pred_ts_arr, axis=0), np.std(pred_ts_arr, axis=0)

            for i in range(len(pred_ts_mean)):
                print(20 * '*')
                # TRANSLATION UNCERTAINTY EVALUATION
                gt_transform = gt_transforms[i].cpu().numpy()
                print("GT Translation Vector:")
                print(gt_transform[0:3, 3])
                print("Predicted Translation Vector Mean:")
                print(pred_ts_mean[i])
                print("Predicted Translation Vector Std:")
                print(pred_ts_std[i])

                for j in range(3):
                    ci_t_lower = np.percentile(pred_ts_arr[:, i, j], 2.5)
                    ci_t_upper = np.percentile(pred_ts_arr[:, i, j], 97.5)
                    if ci_t_lower <= gt_transform[j, 3] <= ci_t_upper:
                        coverage_t[j] += 1
                    if ci_t_lower <= pred_ts_mean[i, j] <= ci_t_upper:
                        pred_coverage_t[j] += 1

                    print(f"t[{j}] CI: [{ci_t_lower:.4f}, {ci_t_upper:.4f}], GT: {gt_transform[j, 3]:.4f}, Pred Mean: {pred_ts_mean[i, j]:.4f}")

                loss = np.mean(np.abs(gt_transform[0:3, 3] - pred_ts_mean[i]))
                print(f"Loss (L1): {loss:.6f}")
                total_loss += loss
                count += 1

                for j in range(3):
                    total_variance_t[j] += pred_ts_std[i][j]
                    total_mean_t[j] += pred_ts_mean[i][j]

                # ROTATION UNCERTAINTY EVALUATION
                gt_rot = gt_transform[0:3, 0:3]
                gt_rot_R = R.from_matrix(gt_rot)
                angles = []
                rotation_matrices = []

                for j in range(mc_samples):
                    pred_rot = gram_schmidt_to_rotation_matrix(pred_vs_arr[j, i], pred_ys_arr[j, i])
                    rotation_matrices.append(pred_rot)
                    pred_rot_R = R.from_matrix(pred_rot)
                    relative_rot = pred_rot_R.inv() * gt_rot_R
                    angles.append(relative_rot.magnitude())

                rotation_matrices = np.stack(rotation_matrices)
                angles = np.array(angles)
                rotation_errors.append(np.mean(angles))

                lower, upper = np.percentile(angles, [2.5, 97.5])
                if lower <= 0.0 <= upper:
                    rotation_ci_coverage += 1

                mean_angle = np.mean(angles)
                std_angle = np.std(angles)

                print(f"Rotation error stats for sample {i}:")
                print(f"  Mean angular error: {mean_angle:.4f} rad / {np.degrees(mean_angle):.2f}°")
                print(f"  Std angular error: {std_angle:.4f} rad / {np.degrees(std_angle):.2f}°")

                # --- NEW PART: Spread and Entropy ---
                rotation_spread = compute_rotation_spread(rotation_matrices)
                rotation_entropy = estimate_rotation_entropy(rotation_matrices)
                spread_list.append(rotation_spread)
                entropy_list.append(rotation_entropy)

                print(f"Sample Spread (radians): {rotation_spread:.4f} rad / {np.degrees(rotation_spread):.2f}°")
                print(f"Rotation Entropy-like measure: {rotation_entropy:.4f}")

                txt_path = sample['txt_path'][i]
                txt_name = 'prediction_{}'.format(os.path.basename(txt_path)).replace("\\", '/')
                txt_dir = os.path.dirname(txt_path)
                save_txt_path = os.path.join(dir_path, txt_dir, txt_name)
                np.savetxt(save_txt_path, pred_ts_mean[i].T.ravel(), fmt='%1.6f', newline=' ')

            # Plot uncertainty for this batch (once per batch)
            plot_uncertainty(pred_ts_arr, [gt_transform[0:3, 3] for gt_transform in gt_transforms],
                             sample_spreads=spread_list, rotation_entropies=entropy_list)
            break  # only first batch

        # --- Final reporting after all samples ---
        if count > 0:
            avg_loss = total_loss / count
            print(f"\nAverage Loss (L1): {avg_loss:.6f}")
            avg_variance_t = [v / count for v in total_variance_t]
            avg_mean_t = [v / count for v in total_mean_t]

            print("\nMean Uncertainty for Translation Vector (T):")
            for j in range(3):
                print(f"t[{j}]: {avg_mean_t[j]:.6f} ± {avg_variance_t[j]:.6f}")

            print("\nCoverage for GT:")
            for j in range(3):
                print(f"t[{j}] coverage: {coverage_t[j]/count:.3f}")

            print("\nCoverage for Predictions (should be ~1.0):")
            for j in range(3):
                print(f"t[{j}] coverage: {pred_coverage_t[j]/count:.3f}")

            if rotation_errors:
                rotation_errors = np.array(rotation_errors)
                mean_rot_error = np.mean(rotation_errors)
                std_rot_error = np.std(rotation_errors)

                print("\nRotation Error Summary:")
                print(f"Mean Angular Error: {mean_rot_error:.4f} rad / {np.degrees(mean_rot_error):.2f}°")
                print(f"Std Angular Error: {std_rot_error:.4f} rad / {np.degrees(std_rot_error):.2f}°")

            if spread_list:
                mean_spread = np.mean(spread_list)
                mean_spread_deg = np.degrees(mean_spread)
                print(f"\nMean Rotation Sample Spread: {mean_spread:.4f} rad / {mean_spread_deg:.2f}°")

            if entropy_list:
                mean_entropy = np.mean(entropy_list)
                print(f"Mean Rotation Entropy-like Measure: {mean_entropy:.4f}")

if __name__ == '__main__':
    args = parse_command_line()
    mc_infer(args)


if __name__ == '__main__':
    args = parse_command_line()
    mc_infer(args)
