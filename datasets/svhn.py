import os
import os.path

import numpy as np

from .cifar import CIFAR10


class SVHN(CIFAR10):
    urls = [
        'http://ufldl.stanford.edu/housenumbers/train_32x32.mat',
        'http://ufldl.stanford.edu/housenumbers/test_32x32.mat',
    ]

    train_list = ['train_32x32.mat']

    test_list = ['test_32x32.mat']

    base_folder = ''

    def __loadfile(self, data_file):
        import scipy.io as sio

        loaded_mat = sio.loadmat(os.path.join(self.root, self.raw_folder, data_file[0]))

        data = loaded_mat['X']
        labels = loaded_mat['y'].astype(np.int64).squeeze()
        np.place(labels, labels == 10, 0)
        data = np.transpose(data, (3, 2, 0, 1))
        # convert to HWC
        data = np.transpose(data, (0, 2, 3, 1))
        return data, labels
