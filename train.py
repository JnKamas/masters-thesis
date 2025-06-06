import torch
import numpy as np
import os

from network import *
from network_helpers import normalized_l2_loss, parse_command_line, load_model
from dataset import Dataset
from torch.utils.data import DataLoader



def get_angles(pred, gt, sym_inv=False, eps=1e-7):
    """
    Calculates angle between pred and gt vectors.
    Clamping args in acos due to: https://github.com/pytorch/pytorch/issues/8069

    :param pred: tensor with shape (batch_size, 3)
    :param gt: tensor with shape (batch_size, 3)
    :param sym_inv: if True the angle is calculated w.r.t bin symmetry
    :param eps: float for NaN avoidance if pred is 0
    :return: tensor with shape (batch_size) containing angles
    """
    pred_norm = torch.norm(pred, dim=-1)
    gt_norm = torch.norm(gt, dim=-1)
    dot = torch.sum(pred * gt, dim=-1)
    if sym_inv:
        angles = torch.acos(torch.clamp(torch.abs(dot / (eps + pred_norm * gt_norm)), -1 + eps, 1 - eps))
    else:
        angles = torch.acos(torch.clamp(dot/(eps + pred_norm * gt_norm), -1 + eps, 1 - eps))
    return angles

def bayesian_combined_loss(args, preds, targets):
    pred_z, pred_y, pred_t = preds
    gt_z, gt_y, gt_t = targets

    loss_z = torch.mean(get_angles(pred_z, gt_z))
    loss_y = torch.mean(get_angles(pred_y, gt_y))
    loss_t = torch.nn.L1Loss()(pred_t, gt_t)

    total_loss = loss_z + loss_y + args.weight * loss_t
    return total_loss, loss_z.detach(), loss_y.detach(), loss_t.detach()



def train(args):
    model = load_model(args)
    train_dataset = Dataset(args.path, 'train', args.input_width, args.input_height, noise_sigma=args.noise_sigma, t_sigma=args.t_sigma, random_rot=args.random_rot, preload=not args.no_preload)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    val_dataset = Dataset(args.path, 'val', args.input_width, args.input_height, preload=not args.no_preload)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    loss_running = 0.0
    loss_t_running = 0.0
    loss_z_running = 0.0
    loss_y_running = 0.0

    l1_loss = torch.nn.L1Loss()
    is_bayesian = (args.modifications == "bayesian")

    start_epoch = 0 if args.resume is None else args.resume
    print("Starting at epoch {}".format(start_epoch))
    print("Running till epoch {}".format(args.epochs))

    train_loss_all = []
    val_loss_all = []
    for e in range(start_epoch, args.epochs):
        print("Starting epoch: ", e)
        for sample in train_loader:
            pred_z, pred_y, pred_t = model(sample['xyz'].cuda())
            optimizer.zero_grad()
            if is_bayesian:
                loss_parts = []

                def wrapped_loss(preds, targets):
                    loss_total, loss_z, loss_y, loss_t = bayesian_combined_loss(args, preds, targets)
                    loss_parts.append((loss_z, loss_y, loss_t))
                    return loss_total

                loss = model.sample_elbo(
                    sample['xyz'].cuda(),
                    (
                        sample['bin_transform'][:, :3, 2].cuda(),
                        sample['bin_transform'][:, :3, 1].cuda(),
                        sample['bin_translation'].cuda(),
                    ),
                    criterion=wrapped_loss,
                    sample_nbr=args.sample_nbr if args.sample_nbr is not None else 3,
                    complexity_cost_weight=args.complexity_cost_weight if args.complexity_cost_weight is not None else 1e-5
                )

                loss_z, loss_y, loss_t = loss_parts[-1]

            else:
                # Angle loss is used for rotational components.
                loss_z = torch.mean(get_angles(pred_z, sample['bin_transform'][:, :3, 2].cuda()))
                # loss_y = torch.mean(get_angles(pred_y, sample['bin_transform'][:, :3, 1].cuda(), sym_inv=True))
                loss_y = torch.mean(get_angles(pred_y, sample['bin_transform'][:, :3, 1].cuda()))
                # loss_t = normalized_l2_loss(pred_t, sample['bin_translation'].cuda())
                loss_t = args.weight * l1_loss(pred_t, sample['bin_translation'].cuda())
                loss = loss_z + loss_y + loss_t

            # Note running loss calc makes loss increase in the beginning of training!
            loss_z_running = 0.9 * loss_z_running + 0.1 * loss_z.item()
            loss_y_running = 0.9 * loss_y_running + 0.1 * loss_y.item()
            loss_t_running = 0.9 * loss_t_running + 0.1 * loss_t.item()
            loss_running = 0.9 * loss_running + 0.1 * loss.item()


            print("Running loss: {}, z loss: {}, y loss: {}, t loss: {}"
                  .format(loss_running,  loss_z_running, loss_y_running, loss_t_running))

            loss.backward()
            optimizer.step()

        train_loss_all.append(loss_running)
        with torch.no_grad():
            val_losses = []
            val_losses_t = []
            val_losses_z = []
            val_losses_y = []
            # val_angles = []
            # val_magnitudes = []

            for sample in val_loader:
                if is_bayesian:
                    preds = [model(sample['xyz'].cuda()) for _ in range(3)]
                    pred_z = torch.mean(torch.stack([p[0] for p in preds]), dim=0)
                    pred_y = torch.mean(torch.stack([p[1] for p in preds]), dim=0)
                    pred_t = torch.mean(torch.stack([p[2] for p in preds]), dim=0)
                else:
                    pred_z, pred_y, pred_t = model(sample['xyz'].cuda())

                optimizer.zero_grad()

                loss_z = torch.mean(get_angles(pred_z, sample['bin_transform'][:, :3, 2].cuda()))
                loss_y = torch.mean(get_angles(pred_y, sample['bin_transform'][:, :3, 1].cuda(), sym_inv=True))
                # loss_t = normalized_l2_loss(pred_t, sample['bin_translation'].cuda())
                loss_t = args.weight * l1_loss(pred_t, sample['bin_translation'].cuda())
                loss = loss_z + loss_y + loss_t

                val_losses.append(loss.item())
                val_losses_t.append(loss_t.item())
                val_losses_z.append(loss_z.item())
                val_losses_y.append(loss_y.item())

            print(20 * "*")
            print("Epoch {}/{}".format(e, args.epochs))
            print("means - \t val loss: {} \t z loss: {} \t y loss: {} \t t loss: {}"
                  .format(np.mean(val_losses), np.mean(val_losses_z), np.mean(val_losses_y), np.mean(val_losses_t)))
            print("medians - \t val loss: {} \t z loss: {} \t y loss: {} \t t loss: {}"
                  .format(np.median(val_losses), np.median(val_losses_z), np.median(val_losses_y), np.median(val_losses_t)))
            import psutil, os
            print(f"Epoch {e} RAM: {psutil.Process(os.getpid()).memory_info().rss // (1024*1024)} MB")

            val_loss_all.append(np.mean(val_losses))

        if args.dump_every != 0 and (e) % args.dump_every == 0:
            print("Saving checkpoint")
            if not os.path.isdir('checkpoints/'):
                os.mkdir('checkpoints/')
            torch.save(model.state_dict(), 'checkpoints/{:03d}.pth'.format(e))

        with open("loss_log.txt", "a") as f:
            f.write(f"{e+1}\t{loss_running:.6f}\t{np.mean(val_losses):.6f}\n")

    final_model_name = f'models/bayes_type{args.complexity_cost_weight}.pth'
    torch.save(model.state_dict(), final_model_name)

    # toto mi neslo... ale mozno to pojde ked pridam .item() ku train_loss_all akoze kde sa to pridava
    np.set_printoptions(suppress=True) 
    np.savetxt('train_err.out', train_loss_all, delimiter=',')
    np.savetxt('val_err.out', val_loss_all, delimiter=',')


if __name__ == '__main__':
    """
    Example usage: python train.py -iw 1032 -ih 772 -b 12 -e 500 -de 10 -lr 1e-3 -bb resnet34 -w 0.1 /path/to/MLBinsDataset/EXR/dataset.json
    """
    args = parse_command_line()
    train(args)
