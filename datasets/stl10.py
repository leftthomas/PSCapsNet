import os
import os.path
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import errno
import torch
import numpy as np

from .cifar import CIFAR10


class STL10(CIFAR10):
    url = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"

    train_list = ['train_X.bin', 'train_y.bin']

    test_list = ['test_X.bin', 'test_y.bin']

    filename = "stl10_binary.tar.gz"

    base_folder = 'stl10_binary'

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

        train_data, train_labels = self.__loadfile(self.train_list[0], self.train_list[1])
        test_data, test_labels = self.__loadfile(self.test_list[0], self.test_list[1])

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
        # convert to HWC
        images = images.transpose((0, 2, 3, 1))
        labels = np.asarray(labels)
        return images, labels
