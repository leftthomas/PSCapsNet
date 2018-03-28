import os
import os.path

import numpy as np
import torch.utils.data as data
from PIL import Image

from .utils import download_url, check_integrity


class SVHN(data.Dataset):
    url = ""
    filename = ""
    file_md5 = ""

    split_list = {
        'train': ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                  "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],
        'test': ["http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                 "test_32x32.mat", "eb5a983be6a315427106f1b164d9cef3"]}

    def __init__(self, root, mode='train', transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode

        if mode != 'train':
            split = 'test'
        else:
            split = mode

        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' + ' You can use download=True to download it')

        import scipy.io as sio

        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        self.data = loaded_mat['X']
        self.labels = loaded_mat['y'].astype(np.int64).squeeze()
        np.place(self.labels, self.labels == 10, 0)
        self.data = np.transpose(self.data, (3, 2, 0, 1))

        if mode == 'test_multi':
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

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        if self.mode != 'train':
            split = 'test'
        else:
            split = self.mode
        root = self.root
        md5 = self.split_list[split][2]
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def download(self):
        if self.mode != 'train':
            split = 'test'
        else:
            split = self.mode
        md5 = self.split_list[split][2]
        download_url(self.url, self.root, self.filename, md5)
