import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchnet.meter.meter import Meter

from datasets import CIFAR100, CIFAR10, MNIST, FashionMNIST, STL10, SVHN
from models import CIFAR10Net, CIFAR100Net, FashionMNISTNet, MNISTNet, STL10Net, SVHNNet

CLASS_NAME = {'MNIST': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
              'FashionMNIST': ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker',
                               'Bag', 'Ankle boot'],
              'SVHN': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
              'CIFAR10': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
              'CIFAR100': ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle',
                           'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle',
                           'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch',
                           'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox',
                           'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
                           'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom',
                           'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
                           'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road',
                           'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
                           'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone',
                           'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale',
                           'willow_tree', 'wolf', 'woman', 'worm'],
              'STL10': ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']}

data_set = {'MNIST': MNIST, 'FashionMNIST': FashionMNIST, 'SVHN': SVHN, 'CIFAR10': CIFAR10, 'CIFAR100': CIFAR100,
            'STL10': STL10}
models = {'MNIST': MNISTNet, 'FashionMNIST': FashionMNISTNet, 'SVHN': SVHNNet, 'CIFAR10': CIFAR10Net,
          'CIFAR100': CIFAR100Net, 'STL10': STL10Net}


class MarginLoss(nn.Module):
    def __init__(self):
        super(MarginLoss, self).__init__()

    def forward(self, classes, labels):
        left = F.relu(0.95 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.05, inplace=True) ** 2
        loss = labels * left + (2 / (classes.size(-1) - 1)) * (1 - labels) * right
        return loss.mean()


class GradCam:
    def __init__(self, model):
        self.model = model.eval()
        self.feature = None
        self.gradient = None

    def save_gradient(self, grad):
        self.gradient = grad

    def __call__(self, x):
        image_size = (x.size(-1), x.size(-2))
        datas = Variable(x)
        heat_maps = []
        for i in range(datas.size(0)):
            img = datas[i].data.cpu().numpy()
            feature = datas[i].unsqueeze(0)
            for name, module in self.model.named_children():
                if name == 'classifier':
                    if self.model.net_mode == 'Capsule':
                        feature = feature.permute(0, 2, 3, 1)
                        feature = feature.contiguous().view(feature.size(0), -1, module.weight.size(-1))
                    else:
                        feature = feature.view(feature.size(0), -1)
                feature = module(feature)
                if name == 'features':
                    feature.register_hook(self.save_gradient)
                    self.feature = feature
            if self.model.net_mode == 'Capsule':
                classes = feature.norm(dim=-1)
            else:
                # don't apply sigmoid, just got the score
                classes = feature
            one_hot, _ = classes.max(dim=-1)
            self.model.zero_grad()
            one_hot.backward()

            weight = self.gradient.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
            mask = F.relu(1 + (weight * self.feature).sum(dim=1)).squeeze(0)
            mask = cv2.resize(mask.data.cpu().numpy(), image_size)
            mask = mask - np.min(mask)
            if np.max(mask) != 0:
                mask = mask / np.max(mask)
            heat_map = np.float32(cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET))

            cam = heat_map + np.float32(np.uint8(img.transpose((1, 2, 0)) * 255))
            cam = cam - np.min(cam)
            if np.max(cam) != 0:
                cam = cam / np.max(cam)
            heat_maps.append(transforms.ToTensor()(cv2.cvtColor(np.uint8(255 * cam), cv2.COLOR_BGR2RGB)))
        heat_maps = torch.stack(heat_maps)
        return heat_maps


def get_iterator(data_type, mode, batch_size=50):
    data = data_set[data_type](root='data/' + data_type, mode=mode, transform=transforms.ToTensor(), download=True)
    return DataLoader(dataset=data, batch_size=batch_size, shuffle=True, num_workers=4)


class MultiClassAccuracyMeter(Meter):
    def __init__(self):
        super(MultiClassAccuracyMeter, self).__init__()
        self.reset()

    def reset(self):
        self.sum = 0
        self.confidence_sum = 0
        self.n = 0

    def add(self, output, target):
        self.n += output.size(0)
        if torch.is_tensor(output):
            output = output.cpu().numpy()
        if torch.is_tensor(target):
            target = target.cpu().numpy()
        greater = np.sort(output, axis=1)[:, -2] > 0.5
        output = output.argsort()[:, -2:]
        output.sort(axis=1)
        self.sum += 1. * (np.prod(output == target, axis=1)).sum()
        self.confidence_sum += 1. * (np.prod(output == target, axis=1) * greater).sum()

    def value(self):
        return (float(self.sum) / self.n) * 100.0, (float(self.confidence_sum) / self.n) * 100.0
