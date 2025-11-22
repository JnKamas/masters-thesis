import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from network import Network
from network_helpers import normalized_l2_loss, parse_command_line, load_model
from dataset import Dataset

import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from network_helpers import normalized_l2_loss, parse_command_line, load_model
from dataset import Dataset


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
        cos = torch.clamp(torch.abs(dot / (eps + pred_norm * gt_norm)), -1 + eps, 1 - eps)
    else:
        cos = torch.clamp(dot / (eps + pred_norm * gt_norm), -1 + eps, 1 - eps)

    return torch.acos(cos)


def bayesian_combined_loss(args, preds, targets):
    pred_z, pred_y, pred_t = preds
    gt_z, gt_y, gt_t = targets

    loss_z = torch.mean(get_angles(pred_z, gt_z))
    loss_y = torch.mean(get_angles(pred_y, gt_y))
    loss_t = torch.nn.L1Loss()(pred_t, gt_t)

    total_loss = loss_z + loss_y + args.weight * loss_t
    return total_loss, loss_z.detach(), loss_y.detach(), loss_t.detach()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    model = load_model(args).to(device)

    train_dataset = Dataset(
        args.path, "train",
        args.input_width, args.input_height,
        noise_sigma=args.noise_sigma,
        t_sigma=args.t_sigma,
        random_rot=args.random_rot,
        preload=not args.no_preload
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    val_dataset = Dataset(
        args.path, "val",
        args.input_width, args.input_height,
        preload=not args.no_preload
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    l1_loss = torch.nn.L1Loss()
    is_bayesian = args.modifications == "bayesian"

    loss_running = loss_t_running = loss_z_running = loss_y_running = 0.0

    start_epoch = args.resume if args.resume is not None else 0
    print(f"Starting at epoch {start_epoch}")
    print(f"Running till epoch {args.epochs}")

    train_loss_all, val_loss_all = [], []

    for e in range(start_epoch, args.epochs):
        print(f"Starting epoch: {e}")
        model.train()

        for sample in train_loader:
            xyz = sample['xyz'].to(device)
            gt_z = sample['bin_transform'][:, :3, 2].to(device)
            gt_y = sample['bin_transform'][:, :3, 1].to(device)
            gt_t = sample['bin_translation'].to(device)

            optimizer.zero_grad()

            if is_bayesian:
                loss_parts = []

                def wrapped_loss(preds, targets):
                    total, lz, ly, lt = bayesian_combined_loss(args, preds, targets)
                    loss_parts.append((lz, ly, lt))
                    return total

                loss = model.sample_elbo(
                    xyz, (gt_z, gt_y, gt_t),
                    criterion=wrapped_loss,
                    sample_nbr=args.sample_nbr,
                    complexity_cost_weight=args.complexity_cost_weight
                )
                loss_z, loss_y, loss_t = loss_parts[-1]

            else:
                pred_z, pred_y, pred_t = model(xyz)

                loss_z = torch.mean(get_angles(pred_z, gt_z))
                loss_y = torch.mean(get_angles(pred_y, gt_y))
                loss_t = args.weight * l1_loss(pred_t, gt_t)
                loss = loss_z + loss_y + loss_t

            loss_z_running = 0.9 * loss_z_running + 0.1 * loss_z.item()
            loss_y_running = 0.9 * loss_y_running + 0.1 * loss_y.item()
            loss_t_running = 0.9 * loss_t_running + 0.1 * loss_t.item()
            loss_running = 0.9 * loss_running + 0.1 * loss.item()

            print(
                f"Running loss: {loss_running:.6f}, "
                f"z: {loss_z_running:.6f}, y: {loss_y_running:.6f}, t: {loss_t_running:.6f}"
            )

            loss.backward()
            optimizer.step()

        train_loss_all.append(loss_running)

        # ---------------------- Validation --------------------
        model.eval()
        with torch.no_grad():
            val_losses = []
            val_losses_z = []
            val_losses_y = []
            val_losses_t = []

            for sample in val_loader:
                xyz = sample['xyz'].to(device)
                gt_z = sample['bin_transform'][:, :3, 2].to(device)
                gt_y = sample['bin_transform'][:, :3, 1].to(device)
                gt_t = sample['bin_translation'].to(device)

                if is_bayesian:
                    preds = [model(xyz) for _ in range(3)]
                    pred_z = torch.mean(torch.stack([p[0] for p in preds]), dim=0)
                    pred_y = torch.mean(torch.stack([p[1] for p in preds]), dim=0)
                    pred_t = torch.mean(torch.stack([p[2] for p in preds]), dim=0)
                else:
                    pred_z, pred_y, pred_t = model(xyz)

                lz = torch.mean(get_angles(pred_z, gt_z))
                ly = torch.mean(get_angles(pred_y, gt_y, sym_inv=True))
                lt = args.weight * l1_loss(pred_t, gt_t)
                l = lz + ly + lt

                val_losses.append(l.item())
                val_losses_z.append(lz.item())
                val_losses_y.append(ly.item())
                val_losses_t.append(lt.item())

            print(20 * "*")
            print(f"Epoch {e}/{args.epochs}")
            print(
                "means - val: {:.6f}, z: {:.6f}, y: {:.6f}, t: {:.6f}".format(
                    np.mean(val_losses),
                    np.mean(val_losses_z),
                    np.mean(val_losses_y),
                    np.mean(val_losses_t),
                )
            )

            val_loss_all.append(np.mean(val_losses))

        if args.dump_every != 0 and e % args.dump_every == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/{e:03d}.pth")

        with open("loss_log.txt", "a") as f:
            f.write(f"{e+1}\t{loss_running:.6f}\t{np.mean(val_losses):.6f}\n")

    # Save finals
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), f"models/bayes_is{args.input_sigma}.pth")

    np.savetxt('train_err.out', np.array(train_loss_all))
    np.savetxt('val_err.out', np.array(val_loss_all))


if __name__ == "__main__":
    args = parse_command_line()
    train(args)

def get_angles(pred, gt, sym_inv: bool = False, eps: float = 1e-7) -> torch.Tensor:
    """
    Calculates angle between pred and gt vectors.
    Clamping args in acos due to: https://github.com/pytorch/pytorch/issues/8069

    Args:
        pred: (B, 3) tensor of predicted vectors.
        gt:   (B, 3) tensor of ground-truth vectors.
        sym_inv: if True, angle is calculated w.r.t. axis symmetry
                 (e.g., for symmetric bins).
        eps: small constant for numerical stability.

    Returns:
        (B,) tensor of angles in radians.
    """
    pred_norm = torch.norm(pred, dim=-1)
    gt_norm = torch.norm(gt, dim=-1)
    dot = torch.sum(pred * gt, dim=-1)

    if sym_inv:
        cos = torch.clamp(torch.abs(dot / (eps + pred_norm * gt_norm)), -1 + eps, 1 - eps)
    else:
        cos = torch.clamp(dot / (eps + pred_norm * gt_norm), -1 + eps, 1 - eps)

    return torch.acos(cos)


def bayesian_combined_loss(args, preds, targets):
    """
    Combined loss used inside blitz's sample_elbo for Bayesian heads.
    """
    pred_z, pred_y, pred_t = preds
    gt_z, gt_y, gt_t = targets

    loss_z = torch.mean(get_angles(pred_z, gt_z))
    loss_y = torch.mean(get_angles(pred_y, gt_y))
    loss_t = torch.nn.L1Loss()(pred_t, gt_t)

    total_loss = loss_z + loss_y + args.weight * loss_t
    return total_loss, loss_z.detach(), loss_y.detach(), loss_t.detach()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load_model should construct the new Network(args) and move it to device,
    # but we defensively call to(device) again (idempotent).
    model = load_model(args)
    model = model.to(device)

    train_dataset = Dataset(
        args.path,
        "train",
        args.input_width,
        args.input_height,
        noise_sigma=args.noise_sigma,
        t_sigma=args.t_sigma,
        random_rot=args.random_rot,
        preload=not args.no_preload,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )

    val_dataset = Dataset(
        args.path,
        "val",
        args.input_width,
        args.input_height,
        preload=not args.no_preload,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    loss_running = 0.0
    loss_t_running = 0.0
    loss_z_running = 0.0
    loss_y_running = 0.0

    l1_loss = torch.nn.L1Loss()
    is_bayesian = (args.modifications == "bayesian")

    start_epoch = 0 if args.resume is None else args.resume
    print(f"Starting at epoch {start_epoch}")
    print(f"Running till epoch {args.epochs}")

    train_loss_all = []
    val_loss_all = []

    for e in range(start_epoch, args.epochs):
        print(f"Starting epoch: {e}")
        model.train()

        for sample in train_loader:
            xyz = sample["xyz"].to(device)
            gt_z = sample["bin_transform"][:, :3, 2].to(device)
            gt_y = sample["bin_transform"][:, :3, 1].to(device)
            gt_t = sample["bin_translation"].to(device)

            optimizer.zero_grad()

            if is_bayesian:
                # For Bayesian mode, we let blitz handle multiple stochastic passes.
                loss_parts = []

                def wrapped_loss(preds, targets):
                    loss_total, loss_z, loss_y, loss_t = bayesian_combined_loss(args, preds, targets)
                    loss_parts.append((loss_z, loss_y, loss_t))
                    return loss_total

                loss = model.sample_elbo(
                    xyz,
                    (gt_z, gt_y, gt_t),
                    criterion=wrapped_loss,
                    sample_nbr=args.sample_nbr if args.sample_nbr is not None else 3,
                    complexity_cost_weight=args.complexity_cost_weight if args.complexity_cost_weight is not None else 1e-5,
                )

                loss_z, loss_y, loss_t = loss_parts[-1]

            else:
                # Deterministic / MC-Dropout mode
                pred_z, pred_y, pred_t = model(xyz)

                # Angle loss is used for rotational components.
                loss_z = torch.mean(get_angles(pred_z, gt_z))
                # If you want symmetry for y-axis, re-enable sym_inv=True.
                loss_y = torch.mean(get_angles(pred_y, gt_y))
                # If you want normalized L2 instead, replace with normalized_l2_loss.
                loss_t = args.weight * l1_loss(pred_t, gt_t)
                loss = loss_z + loss_y + loss_t

            # Note: running loss calc makes loss increase in the beginning of training!
            loss_z_running = 0.9 * loss_z_running + 0.1 * loss_z.item()
            loss_y_running = 0.9 * loss_y_running + 0.1 * loss_y.item()
            loss_t_running = 0.9 * loss_t_running + 0.1 * loss_t.item()
            loss_running = 0.9 * loss_running + 0.1 * loss.item()

            print(
                f"Running loss: {loss_running:.6f}, "
                f"z loss: {loss_z_running:.6f}, "
                f"y loss: {loss_y_running:.6f}, "
                f"t loss: {loss_t_running:.6f}"
            )

            loss.backward()
            optimizer.step()

        train_loss_all.append(loss_running)

        # ---------------------------------------------------------------------
        # Validation
        # ---------------------------------------------------------------------
        model.eval()
        with torch.no_grad():
            val_losses = []
            val_losses_t = []
            val_losses_z = []
            val_losses_y = []

            for sample in val_loader:
                xyz = sample["xyz"].to(device)
                gt_z = sample["bin_transform"][:, :3, 2].to(device)
                gt_y = sample["bin_transform"][:, :3, 1].to(device)
                gt_t = sample["bin_translation"].to(device)

                if is_bayesian:
                    # Simple MC averaging at validation (3 samples).
                    preds = [model(xyz) for _ in range(3)]
                    pred_z = torch.mean(torch.stack([p[0] for p in preds]), dim=0)
                    pred_y = torch.mean(torch.stack([p[1] for p in preds]), dim=0)
                    pred_t = torch.mean(torch.stack([p[2] for p in preds]), dim=0)
                else:
                    pred_z, pred_y, pred_t = model(xyz)

                loss_z = torch.mean(get_angles(pred_z, gt_z))
                # In eval you had sym_inv=True originally for y-axis; kept that here.
                loss_y = torch.mean(get_angles(pred_y, gt_y, sym_inv=True))
                # loss_t = normalized_l2_loss(pred_t, gt_t)
                loss_t = args.weight * l1_loss(pred_t, gt_t)
                loss = loss_z + loss_y + loss_t

                val_losses.append(loss.item())
                val_losses_t.append(loss_t.item())
                val_losses_z.append(loss_z.item())
                val_losses_y.append(loss_y.item())

            print(20 * "*")
            print(f"Epoch {e}/{args.epochs}")
            print(
                "means - \t val loss: {:.6f} \t z loss: {:.6f} \t y loss: {:.6f} \t t loss: {:.6f}".format(
                    np.mean(val_losses),
                    np.mean(val_losses_z),
                    np.mean(val_losses_y),
                    np.mean(val_losses_t),
                )
            )
            print(
                "medians - \t val loss: {:.6f} \t z loss: {:.6f} \t y loss: {:.6f} \t t loss: {:.6f}".format(
                    np.median(val_losses),
                    np.median(val_losses_z),
                    np.median(val_losses_y),
                    np.median(val_losses_t),
                )
            )

            import psutil
            print(f"Epoch {e} RAM: {psutil.Process(os.getpid()).memory_info().rss // (1024 * 1024)} MB")

            val_loss_all.append(np.mean(val_losses))

        # ---------------------------------------------------------------------
        # Checkpoints & logging
        # ---------------------------------------------------------------------
        if args.dump_every != 0 and e % args.dump_every == 0:
            print("Saving checkpoint")
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/{e:03d}.pth")

        with open("loss_log.txt", "a") as f:
            f.write(f"{e+1}\t{loss_running:.6f}\t{np.mean(val_losses):.6f}\n")

    # Final model save
    os.makedirs("models", exist_ok=True)
    final_model_name = f"models/bayes_is{args.input_sigma}.pth"
    torch.save(model.state_dict(), final_model_name)

    # Save loss curves
    np.set_printoptions(suppress=True)
    np.savetxt("train_err.out", np.array(train_loss_all), delimiter=",")
    np.savetxt("val_err.out", np.array(val_loss_all), delimiter=",")


if __name__ == "__main__":
    """
    Example:
        python train.py -iw 1032 -ih 772 -b 12 -e 500 -de 10 -lr 1e-3 -bb resnet34 -w 0.1 /path/to/dataset.json

    For MC-dropout with backbone dropout, something like:
        --modifications mc_dropout --dropout_prob 0.1 --dropout_prob_backbone 0.1
    (depending on how parse_command_line defines these flags)
    """
    args = parse_command_line()
    train(args)
    