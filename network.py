import torch
import torch.nn as nn
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet34, ResNet34_Weights,
    resnet50, ResNet50_Weights
)
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator


# ------------------------------------------------------------
# Add dropout after every ResNet block WITHOUT changing structure
# ------------------------------------------------------------
def insert_block_dropout(backbone, p):
    """
    Inserts nn.Dropout(p) after each residual block
    in layer1, layer2, layer3, layer4.
    Matches your requirement 1.
    """
    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(backbone, layer_name)
        new_seq = []
        for block in layer:
            new_seq.append(block)
            new_seq.append(nn.Dropout(p))
        setattr(backbone, layer_name, nn.Sequential(*new_seq))
    return backbone


@variational_estimator
class Network(nn.Module):
    def __init__(self, args):
        super().__init__()

        # Dropout probabilities
        self.p_backbone = getattr(args, "dropout_prob_backbone", args.dropout_prob)
        self.p_rot = getattr(args, "dropout_prob_rot", args.dropout_prob)
        self.p_trans = getattr(args, "dropout_prob_trans", args.dropout_prob)

        # ------------------------------------------------------------
        # Load ResNet exactly as paper (Figure 3)
        # ------------------------------------------------------------
        if args.backbone == 'resnet18':
            backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif args.backbone == 'resnet34':
            backbone = resnet34(weights=ResNet34_Weights.DEFAULT)
        else:
            backbone = resnet50(weights=ResNet50_Weights.DEFAULT)

        if args.modifications == "mc_dropout":
            backbone = insert_block_dropout(backbone, self.p_backbone)

        self.backbone = nn.Sequential(*list(backbone.children())[:-3])
        last_feat = list(backbone.children())[-1].in_features // 2

        # Heads 
        def make_head(p):
            return nn.Sequential(
                nn.Linear(last_feat, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 3),
            )

        def make_dropout_head(p):
            return nn.Sequential(
                nn.Linear(last_feat, 128),
                nn.LeakyReLU(),
                nn.Dropout(p),
                nn.Linear(128, 64),
                nn.LeakyReLU(),
                nn.Dropout(p),
                nn.Linear(64, 3),
            )

        def make_bayesian_head(btype):
            if btype == 1:
                return nn.Sequential(
                    BayesianLinear(last_feat, 128),
                    nn.LeakyReLU(),
                    nn.Linear(128, 64),
                    nn.LeakyReLU(),
                    nn.Linear(64, 3),
                )
            if btype == 2:
                return nn.Sequential(
                    nn.Linear(last_feat, 128),
                    nn.LeakyReLU(),
                    nn.Linear(128, 64),
                    nn.LeakyReLU(),
                    BayesianLinear(64, 3),
                )
            if btype == 3:
                return nn.Sequential(
                    nn.Linear(last_feat, 128),
                    nn.LeakyReLU(),
                    BayesianLinear(128, 64),
                    nn.LeakyReLU(),
                    nn.Linear(64, 3),
                )
            return nn.Sequential(
                BayesianLinear(last_feat, 128),
                nn.LeakyReLU(),
                BayesianLinear(128, 64),
                nn.LeakyReLU(),
                BayesianLinear(64, 3),
            )

        if args.modifications == "mc_dropout":
            self.fc_z = make_dropout_head(self.p_rot)
            self.fc_y = make_dropout_head(self.p_rot)
            self.fc_t = make_dropout_head(self.p_trans)
        elif args.modifications == "bayesian":
            self.fc_z = make_bayesian_head(args.bayesian_type)
            self.fc_y = make_bayesian_head(args.bayesian_type)
            self.fc_t = make_bayesian_head(args.bayesian_type)
        else:
            self.fc_z = make_head(0.0)
            self.fc_y = make_head(0.0)
            self.fc_t = make_head(0.0)

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def forward(self, x):
        x = self.backbone(x)

        x = torch.mean(x, dim=-1)  # mean over width
        x = torch.mean(x, dim=-1)  # mean over height

        z = self.fc_z(x)
        y = self.fc_y(x)
        t = self.fc_t(x)

        return z, y, t
