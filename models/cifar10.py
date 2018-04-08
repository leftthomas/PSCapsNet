import torch.nn.functional as F
from capsule_layer import CapsuleLinear
from torch import nn

from resnet import resnet20


class CIFAR10Net(nn.Module):
    def __init__(self, net_mode='Capsule', routing_type='k_means', num_iterations=3):
        super(CIFAR10Net, self).__init__()

        self.net_mode = net_mode
        self.conv1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False))

        layers = []
        for name, module in resnet20().named_children():
            if name == 'conv1' or isinstance(module, nn.AvgPool2d) or isinstance(module, nn.Linear):
                continue
            layers.append(module)
        self.features = nn.Sequential(*layers)

        if self.net_mode == 'Capsule':
            self.classifier = nn.Sequential(
                CapsuleLinear(out_capsules=32, in_length=32, out_length=8, routing_type=routing_type,
                              num_iterations=num_iterations, squash=False),
                CapsuleLinear(out_capsules=10, in_length=8, out_length=16, routing_type=routing_type,
                              num_iterations=num_iterations))
        else:
            self.pool = nn.AdaptiveAvgPool2d(output_size=1)
            self.classifier = nn.Sequential(nn.Linear(in_features=64, out_features=64), nn.ReLU(),
                                            nn.Linear(in_features=64, out_features=10))

    def forward(self, x):
        out = self.conv1(x)
        out = self.features(out)

        if self.net_mode == 'Capsule':
            out = out.permute(0, 2, 3, 1)
            out = out.contiguous().view(out.size(0), -1, 32)
            out = self.classifier(out)
            classes = out.norm(dim=-1)
        else:
            out = self.pool(out)
            out = out.view(out.size(0), -1)
            out = self.classifier(out)
            classes = F.sigmoid(out)
        return classes
