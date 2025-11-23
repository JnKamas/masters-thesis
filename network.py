import torch
import torchvision
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator


@variational_estimator
class Network(torch.nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()

        # ------------------------------
        # Dropout probabilities
        # ------------------------------
        self.p_backbone = getattr(args, "dropout_prob_backbone", args.dropout_prob)
        self.p_rot = getattr(args, "dropout_prob_rot", args.dropout_prob)
        self.p_trans = getattr(args, "dropout_prob_trans", args.dropout_prob)
        
        # ------------------------------
        # Backbone (ResNet variants)
        # ------------------------------
        if args.backbone == 'resnet18':
            backbone = torchvision.models.resnet18(pretrained=True)
        elif args.backbone == 'resnet34':
            backbone = torchvision.models.resnet34(pretrained=True)
        else:
            backbone = torchvision.models.resnet50(pretrained=True)

        self.backbone = torch.nn.Sequential(*list(backbone.children())[:-3])
        last_feat = list(backbone.children())[-1].in_features // 2


        # -----------------------------
        # Heads
        # -----------------------------
        def make_head(p_dropout):
            return torch.nn.Sequential(
                torch.nn.Linear(last_feat, 128),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(p_dropout),
                torch.nn.Linear(128, 64),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(p_dropout),
                torch.nn.Linear(64, 3),
            )

        def make_bayesian_head(btype: int) -> torch.nn.Sequential:
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


        if args.modifications == "mc_dropout":
            self.fc_z = make_head(self.p_rot)
            self.fc_y = make_head(self.p_rot)
            self.fc_t = make_head(self.p_trans)
        elif args.modifications == "bayesian":
            self.fc_z = make_bayesian_head(args.bayesian_type)
            self.fc_y = make_bayesian_head(args.bayesian_type)
            self.fc_t = make_bayesian_head(args.bayesian_type)
        else:
            self.fc_z = make_head(0.0)
            self.fc_y = make_head(0.0)
            self.fc_t = make_head(0.0)

        self.backbone_dropout = torch.nn.Dropout(self.p_backbone)

    # -----------------------------------------------------------
    # Forward
    # -----------------------------------------------------------
    def forward(self, x):

        # 1) Backbone
        x = self.backbone(x)

        # 2) nn.Dropout after backbone features
        x = self.backbone_dropout(x)

        # 3) Global average pooling (paper)
        x = torch.mean(x, dim=-1)
        x = torch.mean(x, dim=-1)

        # 4) Heads
        z = self.fc_z(x)
        y = self.fc_y(x)
        t = self.fc_t(x)

        return z, y, t