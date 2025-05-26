import torch
import torchvision
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator

@variational_estimator
class Network(torch.nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()

        # Set dropout probabilities with fallback
        rot_dropout = args.dropout_prob_rot if args.dropout_prob_rot != 0 else args.dropout_prob
        trans_dropout = args.dropout_prob_trans if args.dropout_prob_trans != 0 else args.dropout_prob

        if args.backbone == 'resnet18':
            pretrained_backbone_model = torchvision.models.resnet18(pretrained=True)
        elif args.backbone == 'resnet34':
            pretrained_backbone_model = torchvision.models.resnet34(pretrained=True)
        else:
            pretrained_backbone_model = torchvision.models.resnet50(pretrained=True)

        last_feat = list(pretrained_backbone_model.children())[-1].in_features // 2
        self.backbone = torch.nn.Sequential(*list(pretrained_backbone_model.children())[:-3])

        def make_dropout_branch(dropout):
            return torch.nn.Sequential(
                torch.nn.Linear(last_feat, 128),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(128, 64),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(64, 3)
            )

        def make_bayesian_branch():
            return torch.nn.Sequential(
                BayesianLinear(last_feat, 128),
                torch.nn.LeakyReLU(),
                BayesianLinear(128, 64),
                torch.nn.LeakyReLU(),
                BayesianLinear(64, 3)
            )

        def make_standard_branch():
            return torch.nn.Sequential(
                torch.nn.Linear(last_feat, 128),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(64, 3)
            )

        if args.modifications == "mc_dropout":
            self.fc_z = make_dropout_branch(rot_dropout)
            self.fc_y = make_dropout_branch(rot_dropout)
            self.fc_t = make_dropout_branch(trans_dropout)
        elif args.modifications == "bayesian":
            self.fc_z = make_bayesian_branch()
            self.fc_y = make_bayesian_branch()
            self.fc_t = make_bayesian_branch()
        else:
            self.fc_z = make_standard_branch()
            self.fc_y = make_standard_branch()
            self.fc_t = make_standard_branch()

    def forward(self, x):
        x = self.backbone(x)
        x = torch.mean(x, -1)
        x = torch.mean(x, -1)

        z = self.fc_z(x)
        y = self.fc_y(x)
        t = self.fc_t(x)

        return z, y, t
