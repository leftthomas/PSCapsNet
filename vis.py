import argparse

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image

from model import MixNet
from utils import get_iterator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Capsule Network and CNN Focused Parts')
    parser.add_argument('--data_type', default='STL10', type=str,
                        choices=['MNIST', 'FashionMNIST', 'SVHN', 'CIFAR10', 'STL10'], help='dataset type')
    parser.add_argument('--data_mode', default='test_single', type=str,
                        choices=['test_single', 'test_multi'], help='visualized data mode')
    parser.add_argument('--num_iterations', default=3, type=int, help='routing iterations number')
    opt = parser.parse_args()

    DATA_TYPE = opt.data_type
    DATA_MODE = opt.data_mode
    NUM_ITERATIONS = opt.num_iterations
    batch_size = 16 if DATA_MODE == 'test_single' else 8
    nrow = 4 if DATA_MODE == 'test_single' else 2

    images, labels = next(iter(get_iterator(DATA_TYPE, DATA_MODE, batch_size, False)))
    save_image(images, filename='vis_%s_%s_original.png' % (DATA_TYPE, DATA_MODE), nrow=nrow, normalize=True, padding=4,
               pad_value=255)

    for NET_MODE in ['Capsule_ps', 'Capsule_fc', 'CNN']:
        if NET_MODE == 'Capsule_ps':
            model = MixNet(data_type=DATA_TYPE, capsule_type='ps', num_iterations=NUM_ITERATIONS, return_prob=True)
        elif NET_MODE == 'Capsule_fc':
            model = MixNet(data_type=DATA_TYPE, capsule_type='fc', routing_type='dynamic',
                           num_iterations=NUM_ITERATIONS, return_prob=True)
        else:
            model = MixNet(data_type=DATA_TYPE, net_mode='CNN')
        if torch.cuda.is_available():
            model = model.to('cuda')
            model.load_state_dict(torch.load('epochs/' + DATA_TYPE + '_' + NET_MODE + '.pth'))
        else:
            model.load_state_dict(torch.load('epochs/' + DATA_TYPE + '_' + NET_MODE + '.pth', map_location='cpu'))

        if torch.cuda.is_available():
            images = images.to('cuda')
        image_size = (images.size(-1), images.size(-2))

        for name, module in model.named_children():
            if name == 'conv1':
                out = module(images)
                save_image(out.mean(dim=1, keepdim=True),
                           filename='vis_%s_%s_%s_conv1.png' % (DATA_TYPE, DATA_MODE, NET_MODE), nrow=nrow,
                           normalize=True, padding=4, pad_value=255)
            elif name == 'features':
                out = module(out)
                features = out
            elif name == 'classifier':
                out = out.permute(0, 2, 3, 1)
                out = out.contiguous().view(out.size(0), -1, module.weight.size(-1))
                out, probs = module(out)
                classes = out.norm(dim=-1)
                prob = (probs * classes.unsqueeze(dim=-1)).sum(dim=1)
                prob = prob.view(prob.size(0), *features.size()[-2:], -1)
                prob = prob.permute(0, 3, 1, 2).sum(dim=1)

                heat_maps = []
                for i in range(prob.size(0)):
                    img = images[i].detach().cpu().numpy()
                    img = img - np.min(img)
                    if np.max(img) != 0:
                        img = img / np.max(img)
                    mask = cv2.resize(prob[i].detach().cpu().numpy(), image_size)
                    mask = mask - np.min(mask)
                    if np.max(mask) != 0:
                        mask = mask / np.max(mask)
                    heat_map = np.float32(cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET))
                    cam = heat_map + np.float32((np.uint8(img.transpose((1, 2, 0)) * 255)))
                    cam = cam - np.min(cam)
                    if np.max(cam) != 0:
                        cam = cam / np.max(cam)
                    heat_maps.append(transforms.ToTensor()(cv2.cvtColor(np.uint8(255 * cam), cv2.COLOR_BGR2RGB)))
                heat_maps = torch.stack(heat_maps)
                save_image(heat_maps, filename='vis_%s_%s_%s_features.png' % (DATA_TYPE, DATA_MODE, NET_MODE),
                           nrow=nrow, padding=4, pad_value=255)
