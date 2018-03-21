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
        self.classifier = nn.Linear(in_features=num_features, out_features=2)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        classes = self.classifier(out)
        return classes
