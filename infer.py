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
    # load and set eval()
    model = load_model(args).eval()

    # figure out dataset root
    dir_path = os.path.dirname(args.path)

    # derive model name from your checkpoint arg
    weights_path = getattr(args, 'weights', None) \
                or getattr(args, 'weights_path', None) \
                or getattr(args, 'resume', None)
    if weights_path:
        model_name = os.path.splitext(os.path.basename(weights_path))[0]
    else:
        model_name = 'model'

    # root for all predictions: ~/inference/<model_name>
    export_root = os.path.expanduser(f'~/thesis/inference/{model_name}')
    os.makedirs(export_root, exist_ok=True)

    # prepare data
    val_dataset = Dataset(
        args.path,
        'val',
        args.input_width,
        args.input_height,
        preload=not args.no_preload
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers
    )

    np.set_printoptions(suppress=True)

    with torch.no_grad():
        # if args.modifications == "mc_dropout":
        #     enable_dropout(model)

        for sample in val_loader:
            # Monte Carlo / Bayesian branch
            if args.modifications in {"mc_dropout", "bayesian"}:
                for mc_idx in range(args.mc_samples):
                    z, y, t = model(sample['xyz'].cuda())
                    pred_zs = z.cpu().numpy()
                    pred_ys = y.cpu().numpy()
                    pred_ts = t.cpu().numpy()

                    for i in range(len(pred_zs)):
                        # build orthonormal basis
                        z_i = pred_zs[i];   z_i /= np.linalg.norm(z_i)
                        y_i = pred_ys[i]
                        y_i -= np.dot(z_i, y_i) * z_i
                        y_i /= np.linalg.norm(y_i)
                        x_i = np.cross(y_i, z_i)

                        transform = np.zeros((4,4), dtype=np.float32)
                        transform[:3,0] = x_i
                        transform[:3,1] = y_i
                        transform[:3,2] = z_i
                        transform[:3,3] = pred_ts[i]
                        transform[3,3]  = 1.0

                        txt_path = sample['txt_path'][i].replace("\\","/")
                        base = os.path.basename(txt_path).replace(".txt","")
                        txt_name = f'prediction{mc_idx}_{base}.txt'
                        subdir   = os.path.dirname(txt_path)

                        # save to ~/inference/<model_name>/<subdir>/
                        dst = os.path.join(export_root, subdir)
                        os.makedirs(dst, exist_ok=True)
                        out_file = os.path.join(dst, txt_name)
                        np.savetxt(out_file, transform.T.ravel(), fmt='%1.6f', newline=' ')

            else:
                # single deterministic pass
                pred_zs, pred_ys, pred_ts = model(sample['xyz'].cuda())
                pred_zs = pred_zs.cpu().numpy()
                pred_ys = pred_ys.cpu().numpy()
                pred_ts = pred_ts.cpu().numpy()

            gt_transforms = sample['orig_transform']

            for i in range(len(pred_zs)):
                # print GT vs pred
                print("*"*20)
                print("GT:")
                gt = gt_transforms[i].cpu().numpy()
                print("Det:", np.linalg.det(gt))
                print(gt)

                z = pred_zs[i];   z /= np.linalg.norm(z)
                y = pred_ys[i]
                y -= np.dot(z,y)*z
                y /= np.linalg.norm(y)
                x = np.cross(y, z)

                transform = np.zeros((4,4), dtype=np.float32)
                transform[:3,0] = x
                transform[:3,1] = y
                transform[:3,2] = z
                transform[:3,3] = pred_ts[i]
                transform[3,3]  = 1.0

                print("Predict:")
                print("Det:", np.linalg.det(transform))
                print(transform)

                txt_path = sample['txt_path'][i].replace("\\","/")
                txt_name = f'prediction_{os.path.basename(txt_path)}'
                subdir   = os.path.dirname(txt_path)

                # copy original scan if you like
                dst = os.path.join(export_root, subdir)
                os.makedirs(dst, exist_ok=True)
                orig = os.path.join(dir_path, txt_path)
                if os.path.exists(orig):
                    copyfile(orig, os.path.join(dst, os.path.basename(txt_path)))

                # save prediction
                out_file = os.path.join(dst, txt_name)
                np.savetxt(out_file, transform.T.ravel(), fmt='%1.6f', newline=' ')

if __name__ == '__main__':
    """
    Example:
      python infer.py --weights mymodel.pth --no_preload -r 200 -iw 258 -ih 193 -b 32 /path/to/dataset.json
    """
    args = parse_command_line()
    infer(args)
