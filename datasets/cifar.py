import os
import os.path
import sys

import numpy as np

from .mnist import MNIST

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


class CIFAR10(MNIST):
    urls = ['https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz']

    train_list = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']

    test_list = ['test_batch']

    base_folder = 'cifar-10-batches-py'

    def loadfile(self, data_file):
        data = []
        labels = []
        for f in data_file:
            file = os.path.join(self.root, self.raw_folder, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            data.append(entry['data'])
            if 'labels' in entry:
                labels += entry['labels']
            else:
                labels += entry['fine_labels']
            fo.close()

        data = np.concatenate(data)
        data = data.reshape((-1, 3, 32, 32))
        # convert to HWC
        data = data.transpose((0, 2, 3, 1))
        labels = np.asarray(labels)
        return data, labels


class CIFAR100(CIFAR10):
    urls = ['https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz']

    train_list = ['train']

    test_list = ['test']

    base_folder = 'cifar-100-python'
