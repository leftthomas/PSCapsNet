import torch.nn.functional as F
from capsule_layer import CapsuleLinear
from torch import nn


class CIFAR10Net(nn.Module):
    def __init__(self, num_iterations=3, net_mode='Capsule'):
        super(CIFAR10Net, self).__init__()

        self.net_mode = net_mode
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU())
        self.features = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
                                      nn.AvgPool2d(kernel_size=2),
                                      nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
                                      nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU())
        if self.net_mode == 'Capsule':
            self.classifier = CapsuleLinear(out_capsules=10, in_length=128, out_length=16, in_capsules=None,
                                            routing_type='contract', share_weight=True, num_iterations=num_iterations)
        else:
            self.pool = nn.AdaptiveAvgPool2d(output_size=1)
            self.classifier = nn.Sequential(nn.Linear(in_features=128, out_features=128), nn.ReLU(),
                                            nn.Linear(in_features=128, out_features=10))

    def forward(self, x):
        out = self.conv1(x)
        out = self.features(out)

        if self.net_mode == 'Capsule':
            out = out.permute(0, 2, 3, 1)
            out = out.contiguous().view(out.size(0), -1, 128)
            out = self.classifier(out)
            classes = out.norm(dim=-1)
        else:
            out = self.pool(out)
            out = out.view(out.size(0), -1)
            out = self.classifier(out)
            classes = F.sigmoid(out)
        return classes
