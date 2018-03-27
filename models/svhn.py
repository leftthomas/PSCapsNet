import torch.nn.functional as F
from capsule_layer import CapsuleLinear
from torch import nn

from resnet import resnet20


class SVHNNet(nn.Module):
    def __init__(self, num_iterations=3, net_mode='Capsule'):
        super(SVHNNet, self).__init__()

        self.net_mode = net_mode
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        layers = []
        for name, module in resnet20().named_children():
            if name == 'conv1' or isinstance(module, nn.AvgPool2d) or isinstance(module, nn.Linear):
                continue
            layers.append(module)
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        if self.net_mode == 'Capsule':
            self.classifier = CapsuleLinear(in_capsules=32, out_capsules=10, in_length=2, out_length=4,
                                            routing_type='contract', share_weight=False, num_iterations=num_iterations)
        else:
            self.classifier = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.features(out)
        out = self.pool(out)

        if self.net_mode == 'Capsule':
            out = out.view(*out.size()[:2], -1)
            out = out.transpose(-1, -2)
            out = out.contiguous().view(out.size(0), -1, 2)
            out = self.classifier(out)
            classes = out.norm(dim=-1)
        else:
            out = out.view(out.size(0), -1)
            classes = F.sigmoid(self.classifier(out))
        return classes


if __name__ == '__main__':
    model = SVHNNet()
    for m in model.named_children():
        print(m)
