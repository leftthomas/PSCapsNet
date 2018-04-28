import argparse

import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate figures')
    parser.add_argument('--csv_name', default='experiment_1_iter_3.csv', type=str,
                        choices=['experiment_1_iter_3.csv', 'experiment_1_iter_10.csv', 'experiment_2_CIFAR10.csv',
                                 'experiment_2_FashionMNIST.csv', 'experiment_2_MNIST.csv', 'experiment_2_STL10.csv',
                                 'experiment_2_SVHN.csv', 'experiment_3_CIFAR10.csv', 'experiment_3_FashionMNIST.csv',
                                 'experiment_3_MNIST.csv', 'experiment_3_STL10.csv', 'experiment_3_SVHN.csv'],
                        help='csv file name')
    opt = parser.parse_args()
    CSV_NAME = opt.csv_name
    data = pd.read_csv('statistics/' + CSV_NAME)
    ax = data.plot(x='Epoch')
    ax.set_ylabel('Accuracy')
    plt.show()
