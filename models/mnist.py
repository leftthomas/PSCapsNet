import torch.nn.functional as F
from capsule_layer import CapsuleLinear
from torch import nn


class MNISTNet(nn.Module):
    def __init__(self, num_iterations=3, net_mode='Capsule'):
        super(MNISTNet, self).__init__()

        self.net_mode = net_mode
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.features = nn.Sequential(nn.ReLU(), nn.Conv2d(16, 16, kernel_size=3, padding=1),
                                      nn.ReLU(), nn.AvgPool2d(kernel_size=2),
                                      nn.Conv2d(16, 32, kernel_size=3, padding=1),
                                      nn.ReLU(), nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU())
        if self.net_mode == 'Capsule':
            self.classifier = CapsuleLinear(out_capsules=10, in_length=32, out_length=16, in_capsules=None,
                                            routing_type='contract', share_weight=True, num_iterations=num_iterations)
        else:
            self.pool = nn.AdaptiveAvgPool2d(output_size=1)
            self.classifier = nn.Linear(in_features=32, out_features=10)

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


if __name__ == '__main__':
    model = MNISTNet()
    for m in model.named_children():
        print(m)
