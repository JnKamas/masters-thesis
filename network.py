import torch
import torchvision

from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator

'''
This file defines the Network class, which is a PyTorch neural network model.
It uses a ResNet backbone and can be modified for MC Dropout or Bayesian inference.
'''


@variational_estimator
class Network(torch.nn.Module):
    def __init__(self, backbone='resnet18', modifications=None, dropout_prob=0.5):
        super(Network, self).__init__()

        if backbone == 'resnet18':
            pretrained_backbone_model = torchvision.models.resnet18(pretrained=True)
        elif backbone == 'resnet34':
            pretrained_backbone_model = torchvision.models.resnet34(pretrained=True)
        else:
            pretrained_backbone_model = torchvision.models.resnet50(pretrained=True)

        last_feat = list(pretrained_backbone_model.children())[-1].in_features // 2

        self.backbone = torch.nn.Sequential(*list(pretrained_backbone_model.children())[:-3])
        # print(list(pretrained_backbone_model.children())[0])
        # self.init_conv = torch.nn.Conv2d(3, 64, (11, 11), (5, 5))
        if modifications == "mc_dropout":
            self.fc_z = torch.nn.Sequential(
                torch.nn.Linear(last_feat, 128),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(dropout_prob),
                torch.nn.Linear(128, 64),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(dropout_prob),
                torch.nn.Linear(64, 3)
            )

            self.fc_y = torch.nn.Sequential(
                torch.nn.Linear(last_feat, 128),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(dropout_prob),
                torch.nn.Linear(128, 64),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(dropout_prob),
                torch.nn.Linear(64, 3)
                )

            self.fc_t = torch.nn.Sequential(
                torch.nn.Linear(last_feat, 128),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(dropout_prob),
                torch.nn.Linear(128, 64),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(dropout_prob),
                torch.nn.Linear(64, 3)
            )
        elif modifications == "bayesian":
            self.fc_z = torch.nn.Sequential(
                BayesianLinear(last_feat, 128),
                torch.nn.LeakyReLU(),
                BayesianLinear(128, 64),
                torch.nn.LeakyReLU(),
                BayesianLinear(64, 3)
            )
            self.fc_y = torch.nn.Sequential(
                BayesianLinear(last_feat, 128),
                torch.nn.LeakyReLU(),
                BayesianLinear(128, 64),
                torch.nn.LeakyReLU(),
                BayesianLinear(64, 3)
            )
            self.fc_t = torch.nn.Sequential(
                BayesianLinear(last_feat, 128),
                torch.nn.LeakyReLU(),
                BayesianLinear(128, 64),
                torch.nn.LeakyReLU(),
                BayesianLinear(64, 3)
            )
        else:
            self.fc_z = torch.nn.Sequential(
                torch.nn.Linear(last_feat, 128),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(64, 3)
            )

            self.fc_y = torch.nn.Sequential(
                torch.nn.Linear(last_feat, 128),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(64, 3)
            )

            self.fc_t = torch.nn.Sequential(
                torch.nn.Linear(last_feat, 128),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(64, 3)
            )

    def forward(self, x):
        # x = self.init_conv(x)
        x = self.backbone(x)

        # Global Avg Pool
        x = torch.mean(x, -1)
        x = torch.mean(x, -1)

        # Max pooling
        # x = torch.max(x, -1)[0]
        # x = torch.max(x, -1)[0]

        z = self.fc_z(x)
        y = self.fc_y(x)
        t = self.fc_t(x)

        return z, y, t
