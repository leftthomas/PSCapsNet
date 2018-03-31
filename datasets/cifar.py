import os
import os.path
import sys

import numpy as np

from .mnist import MNIST

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import errno
import torch
from PIL import Image


class CIFAR10(MNIST):
    urls = ['https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz']

    train_list = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']

    test_list = ['test_batch']

    base_folder = 'cifar-10-batches-py'

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img.numpy())

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def download(self):
        from six.moves import urllib
        import tarfile

        if self._check_exists():
            return

        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
        # download files
        for url in self.urls:
            print('Downloading ' + url)
            filename = url.split('/')[-1]
            urllib.request.urlretrieve(url, os.path.join(self.root, self.raw_folder, filename))
            if filename.endswith('.gz'):
                # extract file
                tar = tarfile.open(os.path.join(self.root, self.raw_folder, filename), "r:gz")
                tar.extractall(os.path.join(self.root, self.raw_folder))
                tar.close()

        train_data, train_labels = self.__loadfile(self.train_list)
        test_data, test_labels = self.__loadfile(self.test_list)

        # process and save as torch files
        training_set = (torch.from_numpy(train_data), torch.from_numpy(train_labels))
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        test_single_set = (torch.from_numpy(test_data), torch.from_numpy(test_labels))
        with open(os.path.join(self.root, self.processed_folder, self.test_single_file), 'wb') as f:
            torch.save(test_single_set, f)

        # generate multi dataset
        x_test, y_test = test_data, test_labels
        idx = list(range(len(x_test)))
        np.random.shuffle(idx)
        x_test, y_test = np.concatenate([x_test, x_test[idx]], 2), np.vstack([y_test, y_test[idx]]).T
        # make sure the two number is different
        x_test, y_test = x_test[y_test[:, 0] != y_test[:, 1]], y_test[y_test[:, 0] != y_test[:, 1]]
        # just compare the labels, don't compare the order
        y_test.sort(axis=1)
        test_multi_set = (torch.from_numpy(x_test), torch.from_numpy(y_test))
        with open(os.path.join(self.root, self.processed_folder, self.test_multi_file), 'wb') as f:
            torch.save(test_multi_set, f)

    def __loadfile(self, data_file):
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
