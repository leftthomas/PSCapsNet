import argparse

import torch
from torch.autograd import Variable
from torchvision.utils import save_image

from utils import models, get_iterator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize SP Capsule Network')
    parser.add_argument('--data_type', default='CIFAR10', type=str,
                        choices=['MNIST', 'FashionMNIST', 'SVHN', 'CIFAR10', 'STL10'], help='dataset type')
    parser.add_argument('--model_name', default='CIFAR10_Capsule_95.pth', type=str, help='model epoch name')
    opt = parser.parse_args()

    DATA_TYPE = opt.data_type
    MODEL_NAME = opt.model_name
    model = models[DATA_TYPE]().eval()

    if torch.cuda.is_available():
        model.cuda()
        model.load_state_dict(torch.load('epochs/' + MODEL_NAME))
    else:
        model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location='cpu'))

    images, labels = next(iter(get_iterator(DATA_TYPE, 'test_multi', 16, True)))
    save_image(images, filename='vis_%s_original.png' % DATA_TYPE, nrow=4, normalize=True)
    if torch.cuda.is_available():
        images = images.cuda()
    images = Variable(images)

    for name, module in model.named_children():
        if name == 'conv1':
            images = module(images)
            save_image(images.mean(dim=1, keepdim=True).data, filename='vis_%s_conv1.png' % DATA_TYPE, nrow=4,
                       normalize=True)
        elif name == 'features':
            images = module(images)
            save_image(images.mean(dim=1, keepdim=True).data, filename='vis_%s_features.png' % DATA_TYPE, nrow=4,
                       normalize=True)
        elif name == 'classifier':
            images = images.permute(0, 2, 3, 1)
            images = images.contiguous().view(images.size(0), -1, module.weight.size(-1))
            images = module(images)
            classes = images.norm(dim=-1)
