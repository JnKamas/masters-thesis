import os

import torch
import numpy as np
from dataset import Dataset

from network import Network, parse_command_line, load_model
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def plot_uncertainty(pred_samples, gt_values, save_path="uncertainty_plot.png"):
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

        for sample in val_loader:
            xyz = sample['xyz'].cuda()
            gt_transforms = sample['orig_transform']

            pred_ts_list, pred_vs_list, pred_ys_list = [], [], []

            for _ in range(mc_samples):
                pred_zs, pred_ys, pred_ts = model(xyz)
                pred_ts_list.append(pred_ts.cpu().numpy())
                pred_vs_list.append(pred_zs.cpu().numpy())
                pred_ys_list.append(pred_ys.cpu().numpy())

            # vsetky predikcie
            pred_ts_arr = np.stack(pred_ts_list)
            pred_vs_arr = np.stack(pred_vs_list)
            pred_ys_arr = np.stack(pred_ys_list)

            # ich priemery a odchylky
            pred_ts_mean, pred_ts_std = np.mean(pred_ts_arr, axis=0), np.std(pred_ts_arr, axis=0)
            plot_uncertainty(pred_ts_arr, [gt_transform[0:3, 3] for gt_transform in gt_transforms])

            means.append(pred_ts_mean)
            stds.append(pred_ts_std)

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

                for j in range(mc_samples):
                    pred_rot = gram_schmidt_to_rotation_matrix(pred_vs_arr[j, i], pred_ys_arr[j, i])
                    pred_rot_R = R.from_matrix(pred_rot)
                    relative_rot = pred_rot_R.inv() * gt_rot_R
                    angles.append(relative_rot.magnitude())

                angles = np.array(angles)
                print(angles)
                rotation_errors.append(np.mean(angles))

                # Confidence Interval
                lower, upper = np.percentile(angles, [2.5, 97.5])
                if lower <= 0.0 <= upper:
                    rotation_ci_coverage += 1

                mean_angle = np.mean(angles)
                std_angle = np.std(angles)

                rotation_errors.append(mean_angle)

                print(f"Rotation error stats for sample {i}:")
                print(f"  Mean angular error: {mean_angle:.4f} rad / {np.degrees(mean_angle):.2f}°")
                print(f"  Std angular error: {std_angle:.4f} rad / {np.degrees(std_angle):.2f}°")

                txt_path = sample['txt_path'][i]
                txt_name = 'prediction_{}'.format(os.path.basename(txt_path)).replace("\\", '/')
                txt_dir = os.path.dirname(txt_path)
                save_txt_path = os.path.join(dir_path, txt_dir, txt_name)
                np.savetxt(save_txt_path, pred_ts_mean[i].T.ravel(), fmt='%1.6f', newline=' ')
            break

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

if __name__ == '__main__':
    args = parse_command_line()
    mc_infer(args)
