from capsule_layer import CapsuleLinear
from torch import nn
from torchvision.models import resnet18


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        res_net = resnet18(pretrained=True)
        for param in res_net.parameters():
            param.requires_grad = False
        layers = []
        for name, module in res_net.named_children():
            if name == 'fc':
                break
            else:
                layers.append(module)
        self.features = nn.Sequential(*layers)

        num_features = res_net.fc.in_features
        self.classifier = CapsuleLinear(in_capsules=num_features // 8, out_capsules=2, in_length=8, out_length=16,
                                        routing_type='contract', share_weight=False, num_iterations=3)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1, 8)
        out = self.classifier(out)
        classes = out.norm(dim=-1)
        return classes
