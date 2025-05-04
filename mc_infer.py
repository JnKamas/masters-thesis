import os
import itertools
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader

from dataset import Dataset
from network import Network, parse_command_line, load_model


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
        relative_rotation = R1.inv() * R2
        angles.append(relative_rotation.magnitude())
    return (angles, np.mean(angles)) if angles else 0.0


def estimate_rotation_entropy(rotation_matrices, bins=30):
    N = rotation_matrices.shape[0]
    angles = []
    for i, j in itertools.combinations(range(N), 2):
        R1 = R.from_matrix(rotation_matrices[i])
        R2 = R.from_matrix(rotation_matrices[j])
        relative_rotation = R1.inv() * R2
        angles.append(relative_rotation.magnitude())
    if not angles:
        return 0.0
    hist, _ = np.histogram(angles, bins=bins, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log(hist))


def plot_translation_uncertainty(pred_samples, gt_values, save_path="uncertainty_translation.png"):
    pred_samples = np.array(pred_samples)
    gt_values = np.array([gt.cpu().numpy() for gt in gt_values])

    pred_means = np.mean(pred_samples, axis=0)
    pred_lower = np.percentile(pred_samples, 2.5, axis=0)
    pred_upper = np.percentile(pred_samples, 97.5, axis=0)

    components = ['t[0]', 't[1]', 't[2]']
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    for i in range(3):
        for j in range(pred_samples.shape[0]):
            ax[i].scatter(range(pred_samples.shape[1]), pred_samples[j, :, i], color='blue', alpha=0.1, s=10)
        ax[i].plot(pred_means[:, i], 'b-', label='Mean Prediction')
        ax[i].fill_between(range(pred_means.shape[0]), pred_lower[:, i], pred_upper[:, i], color='blue', alpha=0.3)
        ax[i].scatter(range(gt_values.shape[0]), gt_values[:, i], color='red', label='GT', marker='x', s=50)
        ax[i].set_title(f'Uncertainty in {components[i]}')
        ax[i].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_rotation_metrics(spread_list, entropy_list, angles_samples, angles_between, save_path="uncertainty_rotation.png"):
    """
    Plots rotation uncertainty metrics:
    - Spread and detailed dispersion
    - Entropy
    - Angular errors to GT
    """
    num_plots = 3
    fig, ax = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots))
    plot_idx = 0

    # Plot spread and detailed angles_between together
    if angles_between is not None:
        dispersion_means = []
        for sample_idx, dispersions in enumerate(angles_between):
            sample_idx_array = np.full_like(dispersions, sample_idx)
            ax[plot_idx].scatter(sample_idx_array, dispersions, color='red', alpha=0.1, s=10, label='Pairwise Angles' if sample_idx == 0 else "")
            dispersion_means.append(np.mean(dispersions))
        
        # Mean of dispersions (small red dots)
        ax[plot_idx].scatter(range(len(dispersion_means)), dispersion_means, color='darkred', marker='x', s=50, label='Mean Pairwise Angle')

    # Spread line (green)
    ax[plot_idx].plot(range(len(spread_list)), spread_list, 'g.-', label='Sample Spread')
    ax[plot_idx].set_title('Rotation Spread and Dispersion per Sample')
    ax[plot_idx].set_xlabel('Sample Index')
    ax[plot_idx].set_ylabel('Angular Distance (radians)')
    ax[plot_idx].set_xticks(range(len(spread_list)))  # Force discrete ticks at sample points
    ax[plot_idx].set_ylim(0, np.pi)
    ax[plot_idx].legend()
    plot_idx += 1

    # Plot entropy
    ax[plot_idx].plot(range(len(entropy_list)), entropy_list, 'm.-')
    ax[plot_idx].set_title('Rotation Entropy-like Measure per Sample')
    ax[plot_idx].set_xlabel('Sample Index')
    ax[plot_idx].set_ylabel('Entropy (a.u.)')
    ax[plot_idx].set_xticks(range(len(entropy_list)))
    plot_idx += 1

    # Plot angular errors to GT if available
    if angles_samples is not None:
        angles_samples = np.array(angles_samples).T
        angles_mean = np.mean(angles_samples, axis=0)
        angles_lower = np.percentile(angles_samples, 2.5, axis=0)
        angles_upper = np.percentile(angles_samples, 97.5, axis=0)

        for j in range(angles_samples.shape[0]):
            ax[plot_idx].scatter(range(angles_samples.shape[1]), angles_samples[j, :], color='blue', alpha=0.1, s=10)
        ax[plot_idx].plot(range(angles_mean.shape[0]), angles_mean, 'b-', label='Mean Angular Error')
        ax[plot_idx].fill_between(range(angles_mean.shape[0]), angles_lower, angles_upper, color='blue', alpha=0.3, label='95% CI')
        ax[plot_idx].set_title('Rotation Uncertainty (Angular Errors to GT)')
        ax[plot_idx].set_xlabel('Sample Index')
        ax[plot_idx].set_ylabel('Angular Error (radians)')
        ax[plot_idx].set_xticks(range(angles_mean.shape[0]))
        ax[plot_idx].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def mc_infer(args, export_to_folder=False, mc_samples=100):
    model = load_model(args)
    enable_dropout(model)

    dir_path = os.path.dirname(args.path)
    val_dataset = Dataset(args.path, 'val', args.input_width, args.input_height, preload=not args.no_preload)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    total_loss = 0
    count = 0
    coverage_t = np.zeros(3)
    pred_coverage_t = np.zeros(3)
    total_variance_t = np.zeros(3)
    total_mean_t = np.zeros(3)
    rotation_errors = []
    spread_list = []
    entropy_list = []
    angles_samples = [] # to visualize the spread of angles
    pairwise_angles_list = [] # to visualize the spread of angles between samples

    with torch.no_grad():
        for samle_idx, sample in enumerate(val_loader):
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

            pred_ts_mean = np.mean(pred_ts_arr, axis=0)
            pred_ts_std = np.std(pred_ts_arr, axis=0)

            for i in range(pred_ts_mean.shape[0]):
                gt_transform = gt_transforms[i].cpu().numpy()
                for j in range(3):
                    ci_lower = np.percentile(pred_ts_arr[:, i, j], 2.5)
                    ci_upper = np.percentile(pred_ts_arr[:, i, j], 97.5)
                    if ci_lower <= gt_transform[j, 3] <= ci_upper:
                        coverage_t[j] += 1
                    if ci_lower <= pred_ts_mean[i, j] <= ci_upper:
                        pred_coverage_t[j] += 1

                loss = np.mean(np.abs(gt_transform[0:3, 3] - pred_ts_mean[i]))
                total_loss += loss
                count += 1

                total_variance_t += pred_ts_std[i]
                total_mean_t += pred_ts_mean[i]

                rotation_matrices = [gram_schmidt_to_rotation_matrix(pred_vs_arr[j, i], pred_ys_arr[j, i]) for j in range(mc_samples)]
                gt_rot = gt_transform[0:3, 0:3]
                angles = [R.from_matrix(rot).inv() * R.from_matrix(gt_rot) for rot in rotation_matrices]
                angles = np.array([r.magnitude() for r in angles])
                rotation_errors.append(np.mean(angles))
                
                pairwise_angles, angles_between_mean = compute_rotation_spread(np.stack(rotation_matrices)) # angles between samples
                spread_list.append(angles_between_mean)
                entropy_list.append(estimate_rotation_entropy(np.stack(rotation_matrices)))
                angles_samples.append(angles)
                pairwise_angles_list.append(np.array(pairwise_angles))

                print(40 * "-")
                print(f"Sample {i} Translation:")
                for j in range(3):
                    print(f" t[{j}] Mean: {pred_ts_mean[i, j]:.4f}, Std: {pred_ts_std[i, j]:.4f}")

                print(f"Sample {i} Loss (L1): {loss:.6f}")

                print(f"Sample {i} Rotation:")
                print(f"  Mean Angular Error: {np.mean(angles):.4f} rad / {np.degrees(np.mean(angles)):.2f}°")
                print(f"  Std Angular Error: {np.std(angles):.4f} rad / {np.degrees(np.std(angles)):.2f}°")

                print(f"Sample {i} Spread: {spread_list[-1]:.4f} rad / {np.degrees(spread_list[-1]):.2f}°")
                print(f"Sample {i} Entropy: {entropy_list[-1]:.4f}")
                print(40 * "-")

            plot_translation_uncertainty(pred_ts_arr, [gt_transform[0:3, 3] for gt_transform in gt_transforms])
            plot_rotation_metrics(spread_list, entropy_list, angles_samples=angles_samples, angles_between=pairwise_angles_list)

    avg_loss = total_loss / count
    avg_variance_t = total_variance_t / count
    avg_mean_t = total_mean_t / count
    mean_rot_error = np.mean(rotation_errors)
    std_rot_error = np.std(rotation_errors)
    mean_spread = np.mean(spread_list)
    mean_entropy = np.mean(entropy_list)

    print(f"\nAverage Loss (L1): {avg_loss:.6f}")
    print("\nMean Uncertainty for Translation Vector (T):")
    for j in range(3):
        print(f"t[{j}]: {avg_mean_t[j]:.6f} ± {avg_variance_t[j]:.6f}")
    print("\nRotation Error Summary:")
    print(f"Mean Angular Error: {mean_rot_error:.4f} rad / {np.degrees(mean_rot_error):.2f}°")
    print(f"Std Angular Error: {std_rot_error:.4f} rad / {np.degrees(std_rot_error):.2f}°")
    print(f"\nMean Rotation Sample Spread: {mean_spread:.4f} rad / {np.degrees(mean_spread):.2f}°")
    print(f"Mean Rotation Entropy-like Measure: {mean_entropy:.4f}")

    print("\nCoverage for Translation Vector (T):")
    for j in range(3):
        print(f"t[{j}]: {coverage_t[j] / count:.4f} (GT) / {pred_coverage_t[j] / count:.4f} (Pred)")
    if export_to_folder:
        os.makedirs(dir_path, exist_ok=True)
        plot_translation_uncertainty(pred_ts_arr, [gt_transform[0:3, 3] for gt_transform in gt_transforms], os.path.join(dir_path, "uncertainty_translation.png"))
        plot_rotation_metrics(spread_list, entropy_list, os.path.join(dir_path, "uncertainty_rotation.png"))

if __name__ == '__main__':
    args = parse_command_line()
    mc_infer(args)
