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

def enable_dropout(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()

def infer(args, export_to_folder=True):
    model = load_model(args).eval()

    dir_path = os.path.dirname(args.path)

    val_dataset = Dataset(args.path, 'val', args.input_width, args.input_height, preload=not args.no_preload)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    np.set_printoptions(suppress=True)

    with torch.no_grad():
        if args.modifications == "mc_dropout":
            enable_dropout(model)

        for sample in val_loader:
            pred_zs, pred_ys, pred_ts = model(sample['xyz'].cuda())
            gt_transforms = sample['orig_transform']

            for i in range(len(pred_zs)):
                print(20 * '*')
                print("GT:")
                gt_transform = gt_transforms[i].cpu().numpy()
                print("Det: ", np.linalg.det(gt_transform))
                print(gt_transform)

                z = pred_zs[i].cpu().numpy()
                z /= np.linalg.norm(z)

                y = pred_ys[i].cpu().numpy()
                y = y - np.dot(z, y) * z
                y /= np.linalg.norm(y)

                x = np.cross(y, z)

                transform = np.zeros([4, 4])
                transform[:3, 0] = x
                transform[:3, 1] = y
                transform[:3, 2] = z

                transform[:3, 3] = pred_ts[i].cpu().numpy()
                transform[3, 3] = 1
                print("Predict:")
                print("Det: ", np.linalg.det(transform))
                print(transform)

                txt_path = sample['txt_path'][i].replace("\\", "/")
                print(txt_path)

                txt_name = f'prediction_{os.path.basename(txt_path)}'
                txt_dir = os.path.dirname(txt_path)

                # Save prediction to original dataset folder
                save_txt_path = os.path.join(dir_path, txt_dir, txt_name)
                os.makedirs(os.path.dirname(save_txt_path), exist_ok=True)
                # np.savetxt(save_txt_path, transform.T.ravel(), fmt='%1.6f', newline=' ')

                if export_to_folder:
                    """
                    Copies original scan and adds prediction into 'datasetInference/' while preserving folder structure.
                    """
                    export_root = dir_path + 'Inference'  # e.g., datasetInference
                    export_subdir = os.path.join(export_root, txt_dir)  # e.g., datasetInference/dataset0
                    os.makedirs(export_subdir, exist_ok=True)

                    # 1. Copy scan_000.txt from original dataset
                    original_scan_path = os.path.join(dir_path, txt_path)
                    export_scan_path = os.path.join(export_subdir, os.path.basename(txt_path))
                    if os.path.exists(original_scan_path) and not os.path.exists(export_scan_path):
                        copyfile(original_scan_path, export_scan_path)

                    # 2. Save predicted transform
                    export_pred_path = os.path.join(export_subdir, txt_name)
                    np.savetxt(export_pred_path, transform.T.ravel(), fmt='%1.6f', newline=' ')

                    # WE DO NOT COPY COGS...

                    # # 3. Copy scan's .cogs file if it exists
                    # scan_name = txt_name[11:-4] + '.cogs'
                    # scan_cogs_src = os.path.join(dir_path, txt_dir, scan_name)
                    # scan_cogs_dst = os.path.join(export_subdir, scan_name)
                    # if os.path.exists(scan_cogs_src) and not os.path.exists(scan_cogs_dst):
                    #     copyfile(scan_cogs_src, scan_cogs_dst)

                    # # 4. Copy bin.stl if it exists
                    # bin_src = os.path.join(dir_path, txt_dir, 'bin.stl')
                    # bin_dst = os.path.join(export_subdir, 'bin.stl')
                    # if os.path.exists(bin_src) and not os.path.exists(bin_dst):
                    #     copyfile(bin_src, bin_dst)


if __name__ == '__main__':
    """
    Runs inference and writes prediction txt files.
    Example usage: python infer.py --no_preload -r 200 -iw 258 -ih 193 -b 32 /path/to/MLBinsDataset/EXR/dataset.json
    """
    args = parse_command_line()
    infer(args)
