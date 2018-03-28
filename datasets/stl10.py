import os
import os.path

import numpy as np
from PIL import Image

from .cifar import CIFAR10


class STL10(CIFAR10):
    base_folder = 'stl10_binary'
    url = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
    filename = "stl10_binary.tar.gz"
    tgz_md5 = '91f7769df0f17e558f3565bffb0c7dfb'
    class_names_file = 'class_names.txt'
    train_list = [
        ['train_X.bin', '918c2871b30a85fa023e0c44e0bee87f'],
        ['train_y.bin', '5a34089d4802c674881badbb80307741'],
        ['unlabeled_X.bin', '5242ba1fed5e4be9e1e742405eb56ca4']
    ]

    test_list = [
        ['test_X.bin', '7f263ba9f9e0b06b93213547f721ac82'],
        ['test_y.bin', '36f9794fa4beb8a2c72628de14fa638e']
    ]

    def __init__(self, root, mode='train', transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted. ' + 'You can use download=True to download it')

        # now load the picked numpy arrays
        if self.mode == 'train':
            self.data, self.labels = self.__loadfile(self.train_list[0][0], self.train_list[1][0])
        elif self.mode == 'test_single':
            self.data, self.labels = self.__loadfile(self.test_list[0][0], self.test_list[1][0])
        else:
            # test_multi
            self.data, self.labels = self.__loadfile(self.test_list[0][0], self.test_list[1][0])

            x_test, y_test = self.data, self.labels
            idx = list(range(len(x_test)))
            np.random.shuffle(idx)
            X_test = np.concatenate([x_test, x_test[idx]], 3)
            Y_test = np.vstack([y_test, y_test[idx]]).T
            # make sure the two number is different
            X_test = X_test[Y_test[:, 0] != Y_test[:, 1]]
            Y_test = Y_test[Y_test[:, 0] != Y_test[:, 1]]
            # just compare the labels, don't compare the order
            Y_test.sort(axis=1)
            self.data, self.labels = X_test, Y_test

        class_file = os.path.join(self.root, self.base_folder, self.class_names_file)
        if os.path.isfile(class_file):
            with open(class_file) as f:
                self.classes = f.read().splitlines()

    def __getitem__(self, index):
        if self.mode != 'test_multi':
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], self.labels[index]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.data.shape[0]

    def __loadfile(self, data_file, labels_file=None):
        labels = None
        if labels_file:
            path_to_labels = os.path.join(
                self.root, self.base_folder, labels_file)
            with open(path_to_labels, 'rb') as f:
                labels = np.fromfile(f, dtype=np.uint8) - 1  # 0-based

        path_to_data = os.path.join(self.root, self.base_folder, data_file)
        with open(path_to_data, 'rb') as f:
            # read whole file in uint8 chunks
            everything = np.fromfile(f, dtype=np.uint8)
            images = np.reshape(everything, (-1, 3, 96, 96))
            images = np.transpose(images, (0, 1, 3, 2))

        return images, labels
