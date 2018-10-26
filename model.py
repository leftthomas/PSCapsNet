import torch
from capsule_layer import CapsuleLinear
from torch import nn

from resnet import resnet26


class MixNet(nn.Module):
    def __init__(self, data_type='MNIST', net_mode='Capsule', capsule_type='ps', routing_type='k_means',
                 num_iterations=3, **kwargs):
        super(MixNet, self).__init__()

        self.net_mode = net_mode
        if data_type == 'MNIST' or data_type == 'FashionMNIST':
            self.conv1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False))
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False))
        layers = []
        for name, module in resnet26().named_children():
            if name == 'conv1' or isinstance(module, nn.Linear):
                continue
            if capsule_type == 'ps' and isinstance(module, nn.AdaptiveAvgPool2d):
                continue
            layers.append(module)
        self.features = nn.Sequential(*layers)
        if self.net_mode == 'Capsule':
            if capsule_type == 'ps':
                self.classifier = CapsuleLinear(out_capsules=10, in_length=8, out_length=16, routing_type=routing_type,
                                                num_iterations=num_iterations, **kwargs)
            else:
                self.classifier = CapsuleLinear(out_capsules=10, in_length=8, out_length=16, in_capsules=32,
                                                share_weight=False, routing_type=routing_type,
                                                num_iterations=num_iterations, **kwargs)
        else:
            self.classifier = nn.Sequential(nn.Linear(in_features=256, out_features=128), nn.ReLU(),
                                            nn.Linear(in_features=128, out_features=10))

    def forward(self, x):
        out = self.conv1(x)
        out = self.features(out)

        if self.net_mode == 'Capsule':
            out = out.permute(0, 2, 3, 1)
            out = out.contiguous().view(out.size(0), -1, 8)
            out = self.classifier(out)
            classes = out.norm(dim=-1)
        else:
            out = out.view(out.size(0), -1)
            out = self.classifier(out)
            classes = torch.sigmoid(out)
        return classes
