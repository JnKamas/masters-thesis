import argparse
import os

from network import Network
import torch

def points_loss_function(y_pred, y, vec):
    """Unused loss function"""
    ones = torch.ones(1, 20).cuda()
    vec = torch.cat([vec, ones], dim=0)

    res_pred = torch.matmul(torch.reshape(y_pred, [-1, 4, 4]), vec)
    res = torch.matmul(torch.reshape(y, [-1, 4, 4]), vec)

    res_pred = torch.divide(res_pred[:, :3, :], res_pred[:, 3, None, :] + 1e-7)
    res = torch.divide(res[:, :3, :], res[:, 3, None, :] + 1e-7)

    diff = (res - res_pred) ** 2
    out = torch.mean(torch.sqrt(torch.sum(diff, dim=1)))
    return out


def normalized_l2_loss(pred, gt, reduce=True):
    """
    Returns normalized L2 loss: ||pred - gt||^2 / ||gt||^2
    :param pred: prediction vectors with shape (batch, n)
    :param gt: gt vectors with shape (batch, n)
    :param reduce: if True returns means of losses otherwise returns loss for each element of batch
    :return: normalized L2 loss
    """
    norm = torch.sum(gt ** 2, dim=-1) + 1e-7
    loss = torch.sum((pred - gt) ** 2, dim=-1) / norm
    if reduce:
        return torch.mean(loss)
    else:
        return loss

def parse_command_line():
    """ Parser used for training and inference returns args. Sets up GPUs."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('-r', '--resume', type=int, default=None, help='checkpoint to resume from')
    parser.add_argument('-nw', '--workers', type=int, default=0, help='workers')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('--no_preload', action='store_true', default=False)
    parser.add_argument('-iw', '--input_width', type=int, default=256, help='size of input')
    parser.add_argument('-ih', '--input_height', type=int, default=256, help='size of input')
    parser.add_argument('-e', '--epochs', type=int, default=250, help='max number of epochs')
    parser.add_argument('-g', '--gpu', type=str, default='1', help='which gpu to use')
    parser.add_argument('-bb', '--backbone', type=str, default='resnet18', help='which backbone to use: resnet18/34/50')
    parser.add_argument('-de', '--dump_every', type=int, default=0, help='save every n frames during extraction scripts')
    parser.add_argument('-w', '--weight', type=float, default=1.0, help='weight for translation component')
    parser.add_argument('-ns', '--noise_sigma', type=float, default=None)
    parser.add_argument('-ts', '--t_sigma', type=float, default=0.0)
    parser.add_argument('-rr', '--random_rot', action='store_true', default=False)
    parser.add_argument('-wp', '--weights_path', type=str, default=None, help='Path to the model weights file') # add JK
    parser.add_argument('-vis', '--visualize', action='store_true', default=False, help='Visualize the model predictions into a file') # add JK
    parser.add_argument('-mod', '--modifications', type=str, default=None, help='Modifications to the model: mc_dropout, bayesian') # add JK
    parser.add_argument('-mc', '--mc_samples', type=int, default=30, help='Number of Monte Carlo samples for uncertainty estimation') # add JK
    parser.add_argument('-dpt', '--dropout_prob_trans', type=float, default=0, help='Dropout probability for translation') # add JK
    parser.add_argument('-dpr', '--dropout_prob_rot', type=float, default=0, help='Dropout probability for rotation') # add JK
    parser.add_argument('-dp', '--dropout_prob', type=float, default=0, help='Dropout probability for MC Dropout') # add JK
    parser.add_argument('path')
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    return args

def load_model(args):
    """
    Loads model. If args.resume is None weights for the backbone are pre-trained on ImageNet,
    otherwise previous checkpoint is loaded. If using MC Dropout, remap baseline keys to dropout model.
    """

    model = Network(args).cuda()

    def remap_dropout_state_dict(base_sd):
        """
        Remap baseline state_dict keys to match mc_dropout layer indices:
        fc_*.2.* → fc_*.3.*, and fc_*.4.* → fc_*.6.*
        """
        new_sd = {}
        for k, v in base_sd.items():
            parts = k.split('.')
            name, idx = parts[0], int(parts[1])
            if args.modifications == "mc_dropout" and name in ('fc_z','fc_y','fc_t') and idx in (2, 4):
                # shift second Linear layer (idx=2) → 3, third Linear layer (idx=4) → 6
                new_idx = {2: 3, 4: 6}[idx]
                parts[1] = str(new_idx)
                new_k = '.'.join(parts)
            else:
                new_k = k
            new_sd[new_k] = v
        return new_sd

    if args.weights_path is not None:
        print("Loading weights from:", args.weights_path)
        raw_state_dict = torch.load(args.weights_path, map_location='cpu')
        # if MC Dropout variant, remap baseline keys
        if args.modifications == "mc_dropout":
            state_dict = remap_dropout_state_dict(raw_state_dict)
        else:
            state_dict = raw_state_dict
        # load with strict=False to allow missing keys (dropout vs baseline)
        model.load_state_dict(state_dict, strict=False)

    if args.resume is not None:
        sd_path = f'checkpoints/{args.resume:03d}.pth'
        print("Resuming from:", sd_path)
        model.load_state_dict(torch.load(sd_path), strict=False)

    return model
