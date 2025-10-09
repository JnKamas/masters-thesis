import argparse
import os
import math

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
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('--no_preload', action='store_true', default=False)
    parser.add_argument('-iw', '--input_width', type=int, default=256, help='size of input')
    parser.add_argument('-ih', '--input_height', type=int, default=256, help='size of input')
    parser.add_argument('-e', '--epochs', type=int, default=250, help='max number of epochs')
    parser.add_argument('-g', '--gpu', type=str, default='1', help='which gpu to use')
    parser.add_argument('-bb', '--backbone', type=str, default='resnet18', help='which backbone to use: resnet18/34/50')
    parser.add_argument('-de', '--dump_every', type=int, default=0, help='save every n frames during extraction scripts')
    parser.add_argument('-w', '--weight', type=float, default=0.1, help='weight for translation component')
    parser.add_argument('-ns', '--noise_sigma', type=float, default=None)
    parser.add_argument('-ts', '--t_sigma', type=float, default=0.0)
    parser.add_argument('-rr', '--random_rot', action='store_true', default=False)
    parser.add_argument('-wp', '--weights_path', type=str, default=None, help='Path to the model weights file')
    parser.add_argument('-vis', '--visualize', action='store_true', default=False, help='Visualize the model predictions into a file')
    parser.add_argument('-mod', '--modifications', type=str, default=None, help='Modifications to the model: mc_dropout, bayesian')
    parser.add_argument('-mc', '--mc_samples', type=int, default=30, help='Number of Monte Carlo samples for uncertainty estimation')
    parser.add_argument('-dpt', '--dropout_prob_trans', type=float, default=0, help='Dropout probability for translation')
    parser.add_argument('-dpr', '--dropout_prob_rot', type=float, default=0, help='Dropout probability for rotation')
    parser.add_argument('-dp', '--dropout_prob', type=float, default=0, help='Dropout probability for MC Dropout')
    parser.add_argument('-sn', '--sample_nbr', type=int, default=3, help='Sample number for MC Dropout')
    parser.add_argument('-ccw', '--complexity_cost_weight', type=float, default=0.001, help='Weight for complexity cost in Bayesian layers')
    parser.add_argument('-bt', '--bayesian_type' , type=int, default=0, help='Bayesian type: 0 for full MLP Bayesian, 1 for first MLP layer Bayesian, 2 for last MLP layer Bayesian, 3 for only mid Bayesian, 4 for complete bayesian')
    parser.add_argument('-is', '--input_sigma', type=float, default=0.1, help='Input sigma for Bayesian layers')
    parser.add_argument('path')
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    return args

def remap_bayesian_state_dict(raw_sd, init_sigma=0.1, bayes_type=0):
    """
    Turn a vanilla checkpoint into a full Bayesian state dict:
      - fc_*.{weight,bias} → …_mu
      - Inject matching …_rho, …_sampler.{mu,rho,eps_w|eps_b}
    """
    # 1) compute ρ₀ so σ = log(1 + e^ρ₀) ≈ init_sigma
    rho0 = math.log(math.exp(init_sigma) - 1.0)

    # 2) first pass: remap mu keys
    base_sd = {}
    for k, v in raw_sd.items():
        prefix = k.split('.')[0]
        idx = k.split('.')[1] if '.' in k else None
        # remap if the configuration says so
        if prefix in ('fc_z','fc_y','fc_t') and k.endswith(('weight','bias')) and (bayes_type in {0, 4} or (bayes_type == 1 and idx == '0') or (bayes_type == 2 and idx == '4') or (bayes_type == 3 and idx == '2')):
            base_sd[k + '_mu'] = v
        else:
            base_sd[k] = v

    # 3) second pass: expand each mu into rho + sampler
    full_sd = {}
    for k, v in base_sd.items():
        full_sd[k] = v
        if k.endswith('_mu'):
            base_key = k[:-3]  # strip "_mu"

            # a) rho → filled with ρ₀
            full_sd[base_key + '_rho'] = torch.full_like(v, rho0)

            # b) sampler.mu (copy of μ)
            full_sd[f"{base_key}_sampler.mu"] = v.clone()

            # c) sampler.rho (filled with ρ₀)
            full_sd[f"{base_key}_sampler.rho"] = torch.full_like(v, rho0)

            # d) sampler.eps_w / eps_b (random normals)
            full_sd[f"{base_key}_sampler.eps_w"] = torch.randn_like(v)


    return full_sd

def remap_dropout_state_dict(base_sd):
    """
    Remap baseline state_dict keys to match mc_dropout layer indices:
    fc_*.2.* → fc_*.3.*, and fc_*.4.* → fc_*.6.*
    """
    new_sd = {}
    for k, v in base_sd.items():
        parts = k.split('.')
        name, idx = parts[0], int(parts[1])
        if name in ('fc_z','fc_y','fc_t') and idx in (2, 4):
            new_idx = {2: 3, 4: 6}[idx]
            parts[1] = str(new_idx)
            new_k = '.'.join(parts)
        else:
            new_k = k
        new_sd[new_k] = v
    return new_sd

def load_model(args):
    """
    Loads model. If args.resume is None weights for the backbone are pre-trained on ImageNet,
    otherwise previous checkpoint is loaded. Supports both MC Dropout and Bayesian layers:
      - mc_dropout: remap baseline keys to dropout model indices
      - bayesian: remap to weight_mu/bias_mu and initialize rho for Bayesian layers
    """

    model = Network(args).cuda()


    if args.weights_path is not None:
        print("Loading weights from:", args.weights_path)
        raw_sd = torch.load(args.weights_path, map_location='cpu')
        if args.modifications == "mc_dropout":
            state_dict = remap_dropout_state_dict(raw_sd)
        elif args.modifications == "bayesian":
            raw_sd = torch.load(args.weights_path, map_location='cpu')
            state_dict = remap_bayesian_state_dict(raw_sd, init_sigma=0.1, bayes_type=args.bayesian_type)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                print("Bayesian load – missing keys:", missing)
                print("Bayesian load – unexpected keys:", unexpected)
        else:
            state_dict = raw_sd
        model.load_state_dict(state_dict, strict=True)

        # Initialize rho for Bayesian layers so sigma ≈ 0.1
        if args.modifications == "bayesian":
            init_sigma = args.input_sigma
            rho0 = math.log(math.exp(init_sigma) - 1.0)
            for name, param in model.named_parameters():
                if name.endswith("weight_rho") or name.endswith("bias_rho"):
                    param.data.fill_(rho0)

    if args.resume is not None:
        sd_path = f'checkpoints/{args.resume:03d}.pth'
        print("Resuming from:", sd_path)
        model.load_state_dict(torch.load(sd_path), strict=True)

    return model
