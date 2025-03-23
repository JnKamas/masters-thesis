import os

import torch
import cv2
import numpy as np
from dataset import Dataset

from network import Network, parse_command_line, load_model
from scipy.spatial.transform.rotation import Rotation
from torch.utils.data import DataLoader
from shutil import copyfile
import matplotlib.pyplot as plt

def plot_uncertainty(pred_samples, gt_values, save_path="uncertainty_plot.png"):
    """Plots all MC Dropout predictions, confidence intervals, and GT values."""
    
    pred_samples = np.array(pred_samples)  # Shape: (mc_samples, num_samples, 3)
    gt_values = np.array([gt.cpu().numpy() for gt in gt_values])  # Shape: (num_samples, 3)

    if pred_samples.shape[0] == 0:
        print("Error: pred_samples is empty!")
        return

    pred_means = np.mean(pred_samples, axis=0)  # Mean over MC samples
    pred_lower = np.percentile(pred_samples, 2.5, axis=0)  # 2.5th percentile
    pred_upper = np.percentile(pred_samples, 97.5, axis=0)  # 97.5th percentile

    components = ['t[0]', 't[1]', 't[2]']
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    for i in range(3):  # X, Y, Z components of translation
        num_samples = pred_samples.shape[1]
        
        # Scatter all predictions for visualization
        for j in range(pred_samples.shape[0]):  # Loop over MC samples
            ax[i].scatter(range(num_samples), pred_samples[j, :, i], color='blue', alpha=0.1, s=10)  

        # Plot mean prediction
        ax[i].plot(range(num_samples), pred_means[:, i], 'b-', label='Mean Prediction')

        # Plot confidence interval
        ax[i].fill_between(range(num_samples), pred_lower[:, i], pred_upper[:, i], color='blue', alpha=0.3, label='95% CI')

        # Plot GT values
        ax[i].scatter(range(num_samples), gt_values[:, i], color='red', label='GT', marker='x', s=50)

        ax[i].set_title(f'Uncertainty in {components[i]}')
        ax[i].set_xlabel('Sample Index')
        ax[i].set_ylabel('Translation Value')
        ax[i].legend()

    plt.tight_layout()
    plt.savefig(save_path)  # Save instead of show
    print(f"Plot saved at: {save_path}")



def enable_dropout(model):
    """ Function to enable dropout layers during inference """
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()

def mc_infer(args, export_to_folder=False, mc_samples=100):
    model = load_model(args)
    enable_dropout(model)  # Keep dropout active

    dir_path = os.path.dirname(args.path)
    val_dataset = Dataset(args.path, 'val', args.input_width, args.input_height, preload=not args.no_preload)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    np.set_printoptions(suppress=True)

    with torch.no_grad():
        total_loss = 0
        count = 0
        coverage_t = [0, 0, 0]  # To store coverage for each component of translation t (For GT)
        pred_coverage_t = [0, 0, 0] # Coverage for the preditcion 
        total_variance_t = [0, 0, 0]  # To store total variance for each component of t
        total_mean_t = [0, 0, 0]  # To store total variance for each component of t

        means = []
        stds = []

        for sample in val_loader:
            print(sample)
            xyz = sample['xyz'].cuda()
            gt_transforms = sample['orig_transform']

            # Run MC Dropout multiple times
            pred_ts_list = []
            for _ in range(mc_samples):
                pred_zs, pred_ys, pred_ts = model(xyz) # TODO also other vectors
                pred_ts_list.append(pred_ts.cpu().numpy())

            # Convert list to numpy arrays
            pred_ts_arr = np.stack(pred_ts_list)

            # Compute mean and standard deviation of the predicted translation vector
            pred_ts_mean, pred_ts_std = np.mean(pred_ts_arr, axis=0), np.std(pred_ts_arr, axis=0)

            plot_uncertainty(pred_ts_arr, [gt_transform[0:3, 3] for gt_transform in gt_transforms])

            # Collect means and stds for the final median calculation
            means.append(pred_ts_mean)
            stds.append(pred_ts_std)

            for i in range(len(pred_ts_mean)):
                print(20 * '*')
                print("GT:")
                gt_transform = gt_transforms[i].cpu().numpy()
                print("GT Translation Vector:")
                print(gt_transform[0:3, 3])  # Extract translation vector

                print("Predicted Translation Vector Mean:")
                print(pred_ts_mean[i])

                print("Predicted Translation Vector Std:")
                print(pred_ts_std[i])

                # For each component of the translation vector t (t[0], t[1], t[2])
                for j in range(3):  # j=0 for t[0], j=1 for t[1], j=2 for t[2]
                
                    # If it is normally distributed
                    # ci_t_lower = pred_ts_mean[i][j] - 1.96 * pred_ts_std[i][j]
                    # ci_t_upper = pred_ts_mean[i][j] + 1.96 * pred_ts_std[i][j]

                    # For any distribution, more difficult to cumpute
                    ci_t_lower = np.percentile(pred_ts_arr[:, i, j], 2.5)
                    ci_t_upper = np.percentile(pred_ts_arr[:, i, j], 97.5)

                    print(f"Checking coverage for t[{j}]:")
                    print(f"CI Lower: {ci_t_lower}, CI Upper: {ci_t_upper}")
                    print(f"Pred mean: {pred_ts_mean[i, j]}")
                    print(f"GT Value: {gt_transform[j, 3]}")

                    # Check if the ground truth value falls within the confidence interval
                    if ci_t_lower <= gt_transform[j, 3] <= ci_t_upper:
                        coverage_t[j] += 1

                    # Print the results for the confidence interval check
                    print(f"Coverage for GT x_{j}: {'Inside CI' if ci_t_lower <= gt_transform[j, 3] <= ci_t_upper else 'Outside CI'}")

                    # Check if the ground truth value falls within the confidence interval
                    if ci_t_lower <= pred_ts_mean[i, j] <= ci_t_upper:
                        pred_coverage_t[j] += 1

                    # Print the results for the confidence interval check
                    print(f"Coverage for Prediction x_{j}: {'Inside CI' if ci_t_lower <= pred_ts_mean[i, j] <= ci_t_upper else 'Outside CI'}")

                # L1 loss (Mean Absolute Error) between predicted and GT translation vector
                loss = np.mean(np.abs(gt_transform[0:3, 3] - pred_ts_mean[i]))  # L1 loss for translation
                print(f"Loss (L1): {loss:.6f}")
                total_loss += loss
                count += 1

                # Compute total variance (predictive uncertainty) for each component of translation t
                for j in range(3):  # For each dimension (t[0], t[1], t[2])
                    total_variance_t[j] += pred_ts_std[i][j]
                    total_mean_t[j] += pred_ts_mean[i][j]

                # Optionally save predictions to files (if needed)
                txt_path = sample['txt_path'][i]
                txt_name = 'prediction_{}'.format(os.path.basename(txt_path)).replace("\\", '/')
                txt_dir = os.path.dirname(txt_path)
                save_txt_path = os.path.join(dir_path, txt_dir, txt_name)
                np.savetxt(save_txt_path, pred_ts_mean[i].T.ravel(), fmt='%1.6f', newline=' ')

                if export_to_folder:
                    export_path = dir_path + 'Inference'
                    if not os.path.isdir(export_path):
                        os.mkdir(export_path)

                    if not os.path.isdir(os.path.join(export_path, txt_dir)):
                        os.mkdir(os.path.join(export_path, txt_dir))

                    export_txt_path = os.path.join(export_path, txt_dir, txt_name)
                    np.savetxt(export_txt_path, pred_ts_mean[i].T.ravel(), fmt='%1.6f', newline=' ')
            # break


        # Calculate and print final statistics
        if count > 0:
            avg_loss = total_loss / count
            print(f"Average Loss (L1): {avg_loss:.6f}")

            # Calculate the mean uncertainty (variance) for each dimension of translation t
            avg_variance_t = [v / count for v in total_variance_t]  # Divide by count to get the mean variance
            avg_mean_t = [v / count for v in total_mean_t]  

            print("\nMean Uncertainty (Average Predictive Stdev) for Translation Vector (T):")
            print(f"Mean Uncertainty for t[0]: {avg_mean_t[0]:.6f} +- {avg_variance_t[0]:.6f}")
            print(f"Mean Uncertainty for t[1]: {avg_mean_t[1]:.6f} +- {avg_variance_t[1]:.6f}")
            print(f"Mean Uncertainty for t[2]: {avg_mean_t[2]:.6f} +- {avg_variance_t[2]:.6f}")

            # Calculate the median of means and the median of stds
            stacked_means = np.vstack(means)
            stacked_stds = np.vstack(stds)

            median_mean_t = np.median(stacked_means, axis=0)
            median_std_t = np.median(stacked_stds, axis=0)

            print("\nMedian Uncertainty (Median of Means and Stds) for Translation Vector (T):")
            print(f"Median Uncertainty for t[0]: {median_mean_t[0]:.6f} +- {median_std_t[0]:.6f}")
            print(f"Median Uncertainty for t[1]: {median_mean_t[1]:.6f} +- {median_std_t[1]:.6f}")
            print(f"Median Uncertainty for t[2]: {median_mean_t[2]:.6f} +- {median_std_t[2]:.6f}")

            print("\nCoverage for GT: (How well fitting are our CIs)")
            print(f"t[0] coverage: {coverage_t[0]/count}")
            print(f"t[1] coverage: {coverage_t[1]/count}")
            print(f"t[2] coverage: {coverage_t[2]/count}")

            print("\nCoverage for Predictions: (Should be all 1.0)")
            print(f"t[0] coverage: {pred_coverage_t[0]/count}")
            print(f"t[1] coverage: {pred_coverage_t[1]/count}")
            print(f"t[2] coverage: {pred_coverage_t[2]/count}")


def loss_infer(args, mc_samples=100):
    model = load_model(args)
    enable_dropout(model)  # Keep dropout active

    val_dataset = Dataset(args.path, 'val', args.input_width, args.input_height, preload=not args.no_preload)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    total_loss = 0
    count = 0

    with torch.no_grad():
        for sample in val_loader:
            xyz = sample['xyz'].cuda()
            gt_transforms = sample['orig_transform']

            # Run MC Dropout multiple times
            pred_ts_list = torch.cat([model(xyz) for _ in range(mc_samples)], dim=0)
            pred_ts_mean = np.mean(np.stack(pred_ts_list), axis=0)

            for i in range(len(pred_ts_mean)):
                gt_translation = gt_transforms[i].cpu().numpy()[0:3, 3]
                loss = np.mean(np.abs(gt_translation - pred_ts_mean[i]))  # L1 loss for translation
                total_loss += loss
                count += 1

    if count > 0:
        with open("losses72.txt", "a") as f: f.write(f"{total_loss / count:.6f}\n")


if __name__ == '__main__':
    """
    Runs inference and writes prediction txt files.
    Example usage: python infer.py --no_preload -r 200 -iw 258 -ih 193 -b 32 /path/to/MLBinsDataset/EXR/dataset.json
    """
    args = parse_command_line()
    mc_infer(args)
