import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from torchnet.meter.meter import Meter

from datasets import CIFAR10, MNIST, FashionMNIST, STL10, SVHN
from models import CIFAR10Net, FashionMNISTNet, MNISTNet, STL10Net, SVHNNet

CLASS_NAME = {
    'MNIST': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    'FashionMNIST': ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                     'Ankle boot'],
    'SVHN': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    'CIFAR10': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    'STL10': ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
}

data_set = {'MNIST': MNIST, 'FashionMNIST': FashionMNIST, 'SVHN': SVHN, 'CIFAR10': CIFAR10, 'STL10': STL10}
models = {'MNIST': MNISTNet, 'FashionMNIST': FashionMNISTNet, 'SVHN': SVHNNet, 'CIFAR10': CIFAR10Net, 'STL10': STL10Net}

transform_value = {
    'MNIST': transforms.Normalize((0.1307,), (0.3081,)),
    'FashionMNIST': transforms.Normalize((0.2860,), (0.3530,)),
    'SVHN': transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    'CIFAR10': transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    'STL10': transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713))
}
transform_trains = {
    'MNIST': transforms.Compose(
        [transforms.RandomCrop(28, padding=2), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
    'FashionMNIST': transforms.Compose(
        [transforms.RandomCrop(28, padding=2), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
         transforms.Normalize((0.2860,), (0.3530,))]),
    'SVHN': transforms.Compose(
        [transforms.RandomCrop(32, padding=2), transforms.ToTensor(),
         transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))]),
    'CIFAR10': transforms.Compose(
        [transforms.RandomCrop(32, padding=2), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]),
    'STL10': transforms.Compose(
        [transforms.RandomCrop(96, padding=6), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
         transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713))])
}


class MarginLoss(nn.Module):
    def __init__(self):
        super(MarginLoss, self).__init__()

    def forward(self, classes, labels):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2
        loss = labels * left + 0.5 * (1 - labels) * right
        return loss.sum(dim=-1).mean()


def get_iterator(data_type, mode, batch_size=50, use_data_augmentation=False):
    if use_data_augmentation:
        transform_train = transform_trains[data_type]
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transform_value[data_type]
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor()
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])
    data = data_set[data_type](root='data/' + data_type, mode=mode,
                               transform=transform_train if mode == 'train' else transform_test, download=True)
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
