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

        self.use_aleatoric = getattr(args, "use_aleatoric", False)
        self.p = getattr(args, "dropout_prob", 0.1)

        if args.backbone == 'resnet18':
            backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif args.backbone == 'resnet34':
            backbone = resnet34(weights=ResNet34_Weights.DEFAULT)
        elif args.backbone == 'resnet50':
            backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported backbone: {args.backbone}")
        if args.modifications == "mc_dropout":
            backbone = insert_block_dropout(backbone, self.p_backbone)

        self.backbone = nn.Sequential(*list(backbone.children())[:-3])
        last_feat = list(backbone.children())[-1].in_features // 2

        # Heads 
        def make_head(input_feat, output_feat):
            return torch.nn.Sequential(
                torch.nn.Linear(input_feat, 128),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(64, output_feat)
            )

        def make_dropout_head(input_feat, output_feat, p):
            return torch.nn.Sequential(
                torch.nn.Linear(input_feat, 128),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(p),
                torch.nn.Linear(128, 64),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(p),
                torch.nn.Linear(64, output_feat)
            )

        def make_bayesian_head(input_feat, output_feat, btype):
            if btype == 1:
                return nn.Sequential(
                    BayesianLinear(input_feat, 128),
                    nn.LeakyReLU(),
                    nn.Linear(128, 64),
                    nn.LeakyReLU(),
                    nn.Linear(64, output_feat),
                )
            if btype == 2:
                return nn.Sequential(
                    nn.Linear(input_feat, 128),
                    nn.LeakyReLU(),
                    nn.Linear(128, 64),
                    nn.LeakyReLU(),
                    BayesianLinear(64, output_feat),
                )
            if btype == 3:
                return nn.Sequential(
                    nn.Linear(input_feat, 128),
                    nn.LeakyReLU(),
                    BayesianLinear(128, 64),
                    nn.LeakyReLU(),
                    nn.Linear(64, output_feat),
                )
            return nn.Sequential(
                BayesianLinear(input_feat, 128),
                nn.LeakyReLU(),
                BayesianLinear(128, 64),
                nn.LeakyReLU(),
                BayesianLinear(64, output_feat),
            )
        
        # dimension is extended to four when using aleatoric uncertainty
        output_feat_rot = 4 if self.use_aleatoric else 3 # only for one rotation vector.
        outpot_feat_trans = 6 if self.use_aleatoric else 3 # we dont predict just pose but also std
        if args.modifications == "mc_dropout":
            self.fc_z = make_dropout_head(last_feat, 3, self.p_rot)
            self.fc_y = make_dropout_head(last_feat, output_feat_rot, self.p_rot)
            self.fc_t = make_dropout_head(last_feat, outpot_feat_trans, self.p_trans)
        elif args.modifications == "bayesian":
            self.fc_z = make_bayesian_head(last_feat, 3, args.bayesian_type)
            self.fc_y = make_bayesian_head(last_feat, output_feat_rot, args.bayesian_type)
            self.fc_t = make_bayesian_head(last_feat, outpot_feat_trans, args.bayesian_type)
        else:
            self.fc_z = make_head(last_feat, 3)
            self.fc_y = make_head(last_feat, output_feat_rot)
            self.fc_t = make_head(last_feat, outpot_feat_trans)
    

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def forward(self, x):
        x = self.backbone(x)

        # Global average pooling
        x = torch.mean(x, dim=-1)
        x = torch.mean(x, dim=-1)

        z = self.fc_z(x)
        y = self.fc_y(x)
        t = self.fc_t(x)

        if self.use_aleatoric:
            y_vec = y[:, :3]
            sigma_r = y[:, 3:4]

            t_vec = t[:, :3]
            s_t = t[:, 3:]

            return z, y_vec, t_vec, sigma_r, s_t

        return z, y, t, None, None