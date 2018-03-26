import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader

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


class GradCam:
    def __init__(self, model):
        self.model = model.eval()
        self.feature = None
        self.gradient = None

    def save_gradient(self, grad):
        self.gradient = grad

    def __call__(self, x):
        image_size = (x.size(-2), x.size(-1))
        image_channel = x.size(1)
        datas = Variable(x)
        heat_maps = []
        for i in range(datas.size(0)):
            img = datas[i].data.cpu().numpy()
            feature = datas[i].unsqueeze(0)
            for name, module in self.model.named_children():
                if name == 'classifier':
                    if self.model.net_mode == 'Capsule':
                        feature = feature.view(*feature.size()[:2], -1)
                        feature = feature.transpose(-1, -2)
                        feature = feature.contiguous().view(feature.size(0), -1, module.weight.size(-1))
                    else:
                        feature = feature.view(feature.size(0), -1)
                feature = module(feature)
                if name == 'features':
                    feature.register_hook(self.save_gradient)
                    self.feature = feature
            if self.model.net_mode == 'Capsule':
                classes = feature.sum(dim=-1)
            else:
                classes = feature
            one_hot, _ = classes.max(dim=-1)
            self.model.zero_grad()
            one_hot.backward()

            weight = self.gradient.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
            mask = F.relu((weight * self.feature).sum(dim=1)).squeeze(0)
            mask = cv2.resize(mask.cpu().data.numpy(), image_size)
            mask = mask - np.min(mask)
            if np.max(mask) != 0:
                mask = mask / np.max(mask)
            heat_map = np.float32(cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET))
            cam = heat_map + np.float32(cv2.cvtColor(np.uint8(img.transpose((1, 2, 0)) * 255),
                                                     cv2.COLOR_RGB2BGR if image_channel == 3 else cv2.COLOR_GRAY2BGR))
            cam = cam - np.min(cam)
            if np.max(cam) != 0:
                cam = cam / np.max(cam)
            heat_maps.append(transforms.ToTensor()(cv2.cvtColor(np.uint8(255 * cam), cv2.COLOR_BGR2RGB)))
        heat_maps = torch.stack(heat_maps)
        return heat_maps


def get_iterator(data_type, mode, batch_size=100):
    data = data_set[data_type](root='data/' + data_type, mode=mode, transform=transforms.ToTensor(), download=True)
    return DataLoader(dataset=data, batch_size=batch_size, shuffle=mode, num_workers=4)


if __name__ == '__main__':
    mnist = get_iterator(data_type='MNIST', mode='test_multi')
    fashion_mnist = get_iterator(data_type='FashionMNIST', mode='test_multi')
    svhn = get_iterator(data_type='SVHN', mode='test_multi')
    cifar10 = get_iterator(data_type='CIFAR10', mode='test_multi')
    cifar100 = get_iterator(data_type='CIFAR100', mode='test_multi')
    stl10 = get_iterator(data_type='STL10', mode='test_multi')
