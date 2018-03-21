from torch import nn
from torchvision.models import resnet18


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        res_net = resnet18(pretrained=True)
        for param in res_net.parameters():
            param.requires_grad = False
        num_features = res_net.fc.in_features
        print(res_net.children())
        print(res_net.modules())
        print(res_net.named_children())
        print(res_net.named_modules())
        # for module in res_net.children()
        self.features = None
        self.classifier = nn.Linear(in_features=num_features, out_features=2)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        classes = self.classifier(out)
        return classes


if __name__ == '__main__':
    model = Model()
