import os
import os.path
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import numpy as np

from .cifar import CIFAR10


class STL10(CIFAR10):
    url = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"

    train_list = ['train_X.bin', 'train_y.bin']

    test_list = ['test_X.bin', 'test_y.bin']

    filename = "stl10_binary.tar.gz"

    base_folder = 'stl10_binary'

    def __loadfile(self, data_file):
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
