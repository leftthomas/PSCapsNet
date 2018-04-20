import argparse

import torch
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
    model = models[DATA_TYPE]()

    if torch.cuda.is_available():
        model.cuda()
        model.load_state_dict(torch.load('epochs/' + MODEL_NAME))
    else:
        model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location='cpu'))

    images, labels = next(iter(get_iterator(DATA_TYPE, 'test_multi', 16, True)))
    save_image(images, filename='vis_result.png', nrow=4, normalize=True)
    if torch.cuda.is_available():
        images = images.cuda()
