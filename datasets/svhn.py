import os
import os.path

import numpy as np
import torch.utils.data as data


class SVHN(data.Dataset):
    urls = [
        'http://ufldl.stanford.edu/housenumbers/train_32x32.mat',
        'http://ufldl.stanford.edu/housenumbers/test_32x32.mat',
    ]

    train_list = ['train_32x32.mat']

    test_list = ['test_32x32.mat']

    filename = "cifar-10-python.tar.gz"

    base_folder = 'cifar-10-batches-py'

    def __init__(self, root, mode='train', transform=None, target_transform=None, download=False):
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
