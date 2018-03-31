import codecs
import errno
import os
import os.path

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image


class MNIST(data.Dataset):
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_single_file = 'test_single.pt'
    test_multi_file = 'test_multi.pt'
    train_list = ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte']
    test_list = ['t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte']

    def __init__(self, root, mode='train', transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.mode = mode
        self.transform = transform

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('dataset not found, you can use download=True to download it')

        if self.mode == 'train':
            self.data, self.labels = torch.load(os.path.join(self.root, self.processed_folder, self.training_file))
        elif self.mode == 'test_single':
            self.data, self.labels = torch.load(os.path.join(self.root, self.processed_folder, self.test_single_file))
        elif self.mode == 'test_multi':
            self.data, self.labels = torch.load(os.path.join(self.root, self.processed_folder, self.test_multi_file))
        else:
            raise RuntimeError('mode parameter must between train, test_single and test_multi')

        print('a')

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img.numpy())

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and os.path.exists(
            os.path.join(self.root, self.processed_folder, self.test_single_file)) and os.path.exists(
            os.path.join(self.root, self.processed_folder, self.test_multi_file))

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
        data = read_image_file(os.path.join(self.root, self.raw_folder, data_file[0], data_file[0])).numpy()
        labels = read_label_file(os.path.join(self.root, self.raw_folder, data_file[1], data_file[0])).numpy()
        return data, labels


class FashionMNIST(MNIST):
    urls = [
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz'
    ]


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def parse_byte(b):
    if isinstance(b, str):
        return ord(b)
    return b


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        labels = [parse_byte(b) for b in data[8:]]
        assert len(labels) == length
        return torch.LongTensor(labels)


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        idx = 16
        for l in range(length):
            img = []
            images.append(img)
            for r in range(num_rows):
                row = []
                img.append(row)
                for c in range(num_cols):
                    row.append(parse_byte(data[idx]))
                    idx += 1
        assert len(images) == length
        return torch.ByteTensor(images).view(-1, 28, 28)
