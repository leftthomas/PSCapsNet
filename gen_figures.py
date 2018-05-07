import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    csv_names = ['experiment_1_iter_3.csv', 'experiment_1_iter_10.csv', 'experiment_2_CIFAR10.csv',
                 'experiment_2_FashionMNIST.csv', 'experiment_2_MNIST.csv', 'experiment_2_STL10.csv',
                 'experiment_2_SVHN.csv', 'experiment_3_CIFAR10.csv', 'experiment_3_FashionMNIST.csv',
                 'experiment_3_MNIST.csv', 'experiment_3_STL10.csv', 'experiment_3_SVHN.csv']

    for csv_name in csv_names:
        data = pd.read_csv('statistics/' + csv_name)
        ax = data.plot(x='Epoch')
        ax.set_ylabel('Accuracy')
        plt.savefig(csv_name.split('.')[0] + '.pdf')
