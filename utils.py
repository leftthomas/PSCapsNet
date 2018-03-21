import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_iterator(mode, batch_size=64, use_data_augmentation=True):
    if use_data_augmentation:
        transform_train = transforms.Compose(
            [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        transform_test = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor()
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])

    data = datasets.ImageFolder(root='data/train' if mode else 'data/test',
                                transform=transform_train if mode else transform_test)
    return DataLoader(dataset=data, batch_size=batch_size, shuffle=mode, num_workers=4)
