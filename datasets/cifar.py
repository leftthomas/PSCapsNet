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
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

    train_list = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']

    test_list = ['test_batch']

    filename = "cifar-10-python.tar.gz"

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

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        print('Downloading ' + self.url)
        urllib.request.urlretrieve(self.url, os.path.join(self.root, self.raw_folder, self.filename))
        # extract file
        tar = tarfile.open(os.path.join(self.root, self.raw_folder, self.filename), "r:gz")
        tar.extractall(os.path.join(self.root, self.raw_folder))
        tar.close()

        train_data = []
        train_labels = []
        for f in self.train_list:
            file = os.path.join(self.root, self.raw_folder, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            train_data.append(entry['data'])
            if 'labels' in entry:
                train_labels += entry['labels']
            else:
                train_labels += entry['fine_labels']
            fo.close()

        train_data = np.concatenate(train_data)
        train_data = train_data.reshape((50000, 3, 32, 32))
        # convert to HWC
        train_data = train_data.transpose((0, 2, 3, 1))
        train_labels = np.asarray(train_labels)

        # process and save as torch files
        training_set = (torch.from_numpy(train_data), torch.from_numpy(train_labels))
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)

        f = self.test_list[0]
        file = os.path.join(self.root, self.raw_folder, self.base_folder, f)
        fo = open(file, 'rb')
        if sys.version_info[0] == 2:
            entry = pickle.load(fo)
        else:
            entry = pickle.load(fo, encoding='latin1')
        test_data = entry['data']
        if 'labels' in entry:
            test_labels = entry['labels']
        else:
            test_labels = entry['fine_labels']
        fo.close()
        test_data = test_data.reshape((10000, 3, 32, 32))
        # convert to HWC
        test_data = test_data.transpose((0, 2, 3, 1))
        test_labels = np.asarray(test_labels)

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


class CIFAR100(CIFAR10):
    url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'

    train_list = ['train']

    test_list = ['test']

    filename = "cifar-100-python.tar.gz"

    base_folder = 'cifar-100-python'
