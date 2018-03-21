import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


def get_iterator(mode, batch_size=64, use_data_augmentation=True):
    if use_data_augmentation:
        transform_train = transforms.Compose([transforms.RandomCrop(28, padding=2), transforms.ToTensor(),
                                              transforms.Normalize((0.1306604762738429,), (0.30810780717887876,))])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1306604762738429,), (0.30810780717887876,))
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor()
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])

    data = MNIST(root='data/', train=mode, transform=transform_train if mode else transform_test, download=True)
    return DataLoader(dataset=data, batch_size=batch_size, shuffle=mode, num_workers=4)
