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

def build_rotation_from_yz(y, z, eps=1e-8):
    z = z / (torch.norm(z, dim=1, keepdim=True) + eps)
    y = y - torch.sum(y * z, dim=1, keepdim=True) * z
    y = y / (torch.norm(y, dim=1, keepdim=True) + eps)
    x = torch.cross(y, z, dim=1)
    return torch.stack([x, y, z], dim=2)

def matrix_fisher_nll(R_pred, R_gt, kappa, eps=1e-8):
    # ---- HARD SAFETY (required with approx normalizer) ----
    kappa = torch.clamp(kappa, min=1e-3, max=50.0)

    R_err = torch.matmul(R_pred.transpose(-1, -2), R_gt)

    align = (
        kappa[:, 0] * R_err[:, 0, 0] +
        kappa[:, 1] * R_err[:, 1, 1] +
        kappa[:, 2] * R_err[:, 2, 2]
    )

    kappa_bar = torch.mean(kappa, dim=1)

    # weak but stable normalizer
    log_c = torch.log(kappa_bar + eps) - kappa_bar

    return torch.mean(-align + log_c)



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
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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
    loss_rot_running = 0.0
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
                pred_z, pred_y, pred_t, pred_kappa, pred_sigma_t = model(xyz)

                # Original Loss:
                # # Angle loss is used for rotational components.
                # loss_z = torch.mean(get_angles(pred_z, gt_z))
                # # If you want symmetry for y-axis, re-enable sym_inv=True.
                # loss_y = torch.mean(get_angles(pred_y, gt_y))
                # # If you want normalized L2 instead, replace with normalized_l2_loss.
                # loss_t = args.weight * l1_loss(pred_t, gt_t)
                # loss = loss_z + loss_y + loss_t

                # Loss that supports Matrix-Fisher rotation with concentration (translation unchanged):
                R_pred = build_rotation_from_yz(pred_y, pred_z)
                R_gt   = sample["bin_transform"][:, :3, :3].to(device)

                loss_rot = matrix_fisher_nll(R_pred, R_gt, pred_kappa)

                # ---- Gaussian NLL for translation ----
                pred_sigma_t = torch.clamp(pred_sigma_t, min=1e-3, max=1.0)
                var = pred_sigma_t**2 + 1e-6
                loss_t = 0.5 * (torch.log(var) + (gt_t - pred_t)**2 / var)
                loss_t = args.weight * loss_t.mean()

                loss = loss_rot + loss_t

            # Running averages (true training losses)
            loss_rot_running = 0.9 * loss_rot_running + 0.1 * loss_rot.item()
            loss_t_running   = 0.9 * loss_t_running   + 0.1 * loss_t.item()
            loss_running     = 0.9 * loss_running     + 0.1 * loss.item()

            # Optional diagnostics
            kappa_mean = pred_kappa.mean().item()

            print(
                f"Running loss: {loss_running:.6f}, "
                f"rot NLL: {loss_rot_running:.6f}, "
                f"t loss: {loss_t_running:.6f}, "
                f"kappa_mean: {kappa_mean:.3f}"
            )

            loss.backward()
            optimizer.step()

        train_loss_all.append(loss_running)

        # ---------------------------------------------------------------------
        # Validation
        # ---------------------------------------------------------------------
        model.eval()
        with torch.no_grad():
            val_losses = []          # total objective
            val_losses_rot = []      # Matrix Fisher NLL
            val_losses_t = []        # translation
            val_angle_z = []         # metrics only
            val_angle_y = []

            for sample in val_loader:
                xyz = sample["xyz"].to(device)
                R_gt = sample["bin_transform"][:, :3, :3].to(device)
                gt_z = R_gt[:, :, 2]
                gt_y = R_gt[:, :, 1]
                gt_t = sample["bin_translation"].to(device)

                if is_bayesian:
                    preds = [model(xyz) for _ in range(3)]
                    pred_z = torch.mean(torch.stack([p[0] for p in preds]), dim=0)
                    pred_y = torch.mean(torch.stack([p[1] for p in preds]), dim=0)
                    pred_t = torch.mean(torch.stack([p[2] for p in preds]), dim=0)
                    pred_kappa = torch.mean(torch.stack([p[3] for p in preds]), dim=0)
                    pred_sigma_t = torch.mean(torch.stack([p[4] for p in preds]), dim=0)
                else:
                    pred_z, pred_y, pred_t, pred_kappa, pred_sigma_t = model(xyz)

                # ---- objective losses ----
                R_pred = build_rotation_from_yz(pred_y, pred_z)
                loss_rot = matrix_fisher_nll(R_pred, R_gt, pred_kappa)
                # loss_t = args.weight * l1_loss(pred_t, gt_t)
                eps = 1e-6
                pred_sigma_t = torch.clamp(pred_sigma_t, min=1e-3, max=1.0)
                var = pred_sigma_t**2 + eps
                loss_t = 0.5 * (torch.log(var) + (gt_t - pred_t)**2 / var)
                loss_t = args.weight * loss_t.mean()
                loss = loss_rot + loss_t

                # ---- metrics (angles) ----
                angle_z = torch.mean(get_angles(pred_z, gt_z))
                angle_y = torch.mean(get_angles(pred_y, gt_y, sym_inv=True))

                val_losses.append(loss.item())
                val_losses_rot.append(loss_rot.item())
                val_losses_t.append(loss_t.item())
                val_angle_z.append(angle_z.item())
                val_angle_y.append(angle_y.item())

            print(20 * "*")
            print(f"Epoch {e}/{args.epochs}")
            print(
                "means - "
                f"val loss: {np.mean(val_losses):.6f}, "
                f"rot NLL: {np.mean(val_losses_rot):.6f}, "
                f"t loss: {np.mean(val_losses_t):.6f}, "
                f"angle z: {np.mean(val_angle_z):.4f}, "
                f"angle y: {np.mean(val_angle_y):.4f}"
            )
            print(
                "medians - "
                f"val loss: {np.median(val_losses):.6f}, "
                f"rot NLL: {np.median(val_losses_rot):.6f}, "
                f"t loss: {np.median(val_losses_t):.6f}, "
                f"angle z: {np.median(val_angle_z):.4f}, "
                f"angle y: {np.median(val_angle_y):.4f}"
            )

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
    