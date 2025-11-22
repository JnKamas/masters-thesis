import torch
import torch.nn.functional as F
import torchvision
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator


@variational_estimator
class Network(torch.nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()

        # Dropout probabilities
        rot_dropout = args.dropout_prob_rot if args.dropout_prob_rot != 0 else args.dropout_prob
        trans_dropout = args.dropout_prob_trans if args.dropout_prob_trans != 0 else args.dropout_prob
        self.backbone_dropout_p = getattr(args, "dropout_prob_backbone", 0.0)

        # -----------------------------
        # Backbone (EXACTLY as before)
        # -----------------------------
        if args.backbone == 'resnet18':
            pretrained_backbone_model = torchvision.models.resnet18(pretrained=True)
        elif args.backbone == 'resnet34':
            pretrained_backbone_model = torchvision.models.resnet34(pretrained=True)
        else:
            pretrained_backbone_model = torchvision.models.resnet50(pretrained=True)

        # children()[:-3] → conv1, bn1, relu, maxpool, layer1, layer2, layer3
        # This matches the original checkpoints.
        self.backbone = torch.nn.Sequential(
            *list(pretrained_backbone_model.children())[:-3]
        )

        # Same feature dim as before
        last_feat = list(pretrained_backbone_model.children())[-1].in_features // 2

        # -----------------------------
        # Heads
        # -----------------------------
        def make_dropout_branch(p: float) -> torch.nn.Sequential:
            return torch.nn.Sequential(
                torch.nn.Linear(last_feat, 128),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(p),

                torch.nn.Linear(128, 64),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(p),

                torch.nn.Linear(64, 3),
            )

        def make_bayesian_branch(btype: int) -> torch.nn.Sequential:
            if btype == 1:
                # First MLP layer Bayesian
                return torch.nn.Sequential(
                    BayesianLinear(last_feat, 128),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(128, 64),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(64, 3),
                )
            elif btype == 2:
                # Last MLP layer Bayesian
                return torch.nn.Sequential(
                    torch.nn.Linear(last_feat, 128),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(128, 64),
                    torch.nn.LeakyReLU(),
                    BayesianLinear(64, 3),
                )
            elif btype == 3:
                # Middle MLP layer Bayesian
                return torch.nn.Sequential(
                    torch.nn.Linear(last_feat, 128),
                    torch.nn.LeakyReLU(),
                    BayesianLinear(128, 64),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(64, 3),
                )

            # Full Bayesian (type 0)
            return torch.nn.Sequential(
                BayesianLinear(last_feat, 128),
                torch.nn.LeakyReLU(),
                BayesianLinear(128, 64),
                torch.nn.LeakyReLU(),
                BayesianLinear(64, 3),
            )

        def make_standard_branch() -> torch.nn.Sequential:
            return torch.nn.Sequential(
                torch.nn.Linear(last_feat, 128),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(64, 3),
            )

        if args.modifications == "mc_dropout":
            self.fc_z = make_dropout_branch(rot_dropout)
            self.fc_y = make_dropout_branch(rot_dropout)
            self.fc_t = make_dropout_branch(trans_dropout)
        elif args.modifications == "bayesian":
            self.fc_z = make_bayesian_branch(args.bayesian_type)
            self.fc_y = make_bayesian_branch(args.bayesian_type)
            self.fc_t = make_bayesian_branch(args.bayesian_type)
        else:
            self.fc_z = make_standard_branch()
            self.fc_y = make_standard_branch()
            self.fc_t = make_standard_branch()

    # -----------------------------
    # Backbone forward with optional dropout
    # -----------------------------
    def _forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs the original backbone, but applies functional dropout on activations
        after the higher-level layers (layer1, layer2, layer3) if
        self.backbone_dropout_p > 0.

        This DOES NOT change any parameter names, so checkpoints remain compatible.
        """
        # self.backbone modules:
        # 0: conv1
        # 1: bn1
        # 2: relu
        # 3: maxpool
        # 4: layer1
        # 5: layer2
        # 6: layer3
        for i, m in enumerate(self.backbone):
            x = m(x)
            # Apply dropout only after the high-level ResNet layers
            if self.backbone_dropout_p > 0.0 and i >= 4:
                x = F.dropout(x, p=self.backbone_dropout_p, training=True)
        return x

    # -----------------------------
    # Forward
    # -----------------------------
    def forward(self, x: torch.Tensor):
        x = self._forward_backbone(x)

        # Global average pooling over spatial dims (same as original)
        x = torch.mean(x, dim=-1)
        x = torch.mean(x, dim=-1)

        z = self.fc_z(x)
        y = self.fc_y(x)
        t = self.fc_t(x)

        return z, y, t
