import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, last_planes, in_planes, out_planes, dense_depth, stride, first_layer):
        super(Bottleneck, self).__init__()
        self.out_planes = out_planes
        self.dense_depth = dense_depth

        self.conv1 = nn.Conv2d(last_planes, in_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=8, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        self.conv3 = nn.Conv2d(in_planes, out_planes + dense_depth, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes + dense_depth)

        self.shortcut = nn.Sequential()
        if first_layer:
            self.shortcut = nn.Sequential(
                nn.Conv2d(last_planes, out_planes + dense_depth, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes + dense_depth)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        x = self.shortcut(x)
        d = self.out_planes
        out = torch.cat([x[:, :d, :, :] + out[:, :d, :, :], x[:, d:, :, :], out[:, d:, :, :]], 1)
        out = F.relu(out)
        return out


class DPN(nn.Module):
    def __init__(self, cfg, stl10=False, num_classes=10):
        super(DPN, self).__init__()
        in_planes, out_planes = cfg['in_planes'], cfg['out_planes']
        num_blocks, dense_depth = cfg['num_blocks'], cfg['dense_depth']

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.last_planes = 16
        self.layer1 = self._make_layer(in_planes[0], out_planes[0], num_blocks[0], dense_depth[0], stride=1)
        self.layer2 = self._make_layer(in_planes[1], out_planes[1], num_blocks[1], dense_depth[1], stride=2)
        self.layer3 = self._make_layer(in_planes[2], out_planes[2], num_blocks[2], dense_depth[2], stride=2)
        self.stl10 = stl10
        if self.stl10:
            self.layer4 = self._make_layer(in_planes[3], out_planes[3], num_blocks[3], dense_depth[3], stride=2)
            self.final_features = out_planes[3] + (num_blocks[3] + 1) * dense_depth[3]
            self.avgpool = nn.AvgPool2d(12, stride=1)
        else:
            self.final_features = out_planes[2] + (num_blocks[2] + 1) * dense_depth[2]
            self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(self.final_features, num_classes)

    def _make_layer(self, in_planes, out_planes, num_blocks, dense_depth, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, stride in enumerate(strides):
            layers.append(Bottleneck(self.last_planes, in_planes, out_planes, dense_depth, stride, i == 0))
            self.last_planes = out_planes + (i + 2) * dense_depth
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        if self.stl10:
            out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def dpn_26(**kwargs):
    cfg = {
        'in_planes': (24, 48, 96, 192),
        'out_planes': (64, 128, 256, 512),
        'num_blocks': (2, 2, 2, 2),
        'dense_depth': (4, 8, 6, 32)
    }
    return DPN(cfg, **kwargs)


def dpn_92(**kwargs):
    cfg = {
        'in_planes': (24, 48, 96, 192),
        'out_planes': (64, 128, 256, 512),
        'num_blocks': (3, 4, 20, 3),
        'dense_depth': (4, 8, 6, 32)
    }
    return DPN(cfg, **kwargs)


def dpn_26_stl10(**kwargs):
    cfg = {
        'in_planes': (24, 48, 96, 192),
        'out_planes': (64, 128, 256, 512),
        'num_blocks': (2, 2, 2, 2),
        'dense_depth': (4, 8, 6, 32)
    }
    return DPN(cfg, stl10=True, **kwargs)


def dpn_92_stl10(**kwargs):
    cfg = {
        'in_planes': (24, 48, 96, 192),
        'out_planes': (64, 128, 256, 512),
        'num_blocks': (3, 4, 20, 3),
        'dense_depth': (4, 8, 6, 32)
    }
    return DPN(cfg, stl10=True, **kwargs)
