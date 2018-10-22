import os

import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':

    for file_name in os.listdir('statistics/'):
        if os.path.splitext(file_name)[1] == '.csv':
            data = pd.read_csv('statistics/' + file_name)
            ax = data.plot(x='Epoch')
            ax.set_ylabel('Accuracy')
            plt.savefig(file_name.split('.')[0] + '.pdf')
