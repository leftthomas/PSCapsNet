import argparse

import pandas as pd
import torch
import torchnet as tnt
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger, VisdomLogger
from torchvision.utils import make_grid
from tqdm import tqdm

from utils import get_iterator, CLASS_NAME, models, GradCam


def processor(sample):
    data, labels, training = sample

    if torch.cuda.is_available():
        data = data.cuda()
        labels = labels.cuda()
    data = Variable(data)
    labels = Variable(labels)

    model.train(training)

    classes = model(data)
    loss = loss_criterion(classes, labels)
    return loss, classes


def on_sample(state):
    state['sample'].append(state['train'])


def reset_meters():
    meter_accuracy.reset()
    meter_loss.reset()
    confusion_meter.reset()


def on_forward(state):
    meter_accuracy.add(state['output'].data, state['sample'][1])
    confusion_meter.add(state['output'].data, state['sample'][1])
    meter_loss.add(state['loss'].data[0])


def on_start_epoch(state):
    reset_meters()
    state['iterator'] = tqdm(state['iterator'])


def on_end_epoch(state):
    print('[Epoch %d] Training Loss: %.4f Training Accuracy: %.2f%%' % (
        state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

    train_loss_logger.log(state['epoch'], meter_loss.value()[0])
    train_accuracy_logger.log(state['epoch'], meter_accuracy.value()[0])
    results['train_loss'].append(meter_loss.value()[0])
    results['train_accuracy'].append(meter_accuracy.value()[0])

    reset_meters()

    engine.test(processor, get_iterator(False, DATA_TYPE, BATCH_SIZE))

    test_loss_logger.log(state['epoch'], meter_loss.value()[0])
    test_accuracy_logger.log(state['epoch'], meter_accuracy.value()[0])
    confusion_logger.log(confusion_meter.value())
    results['test_loss'].append(meter_loss.value()[0])
    results['test_accuracy'].append(meter_accuracy.value()[0])

    print('[Epoch %d] Testing Loss: %.4f Testing Accuracy: %.2f%%' % (
        state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

    torch.save(model.state_dict(), 'epochs/%s_%d.pth' % (DATA_TYPE, state['epoch']))

    # save statistics at every 10 epochs
    if state['epoch'] % 10 == 0:
        out_path = 'statistics/'
        data_frame = pd.DataFrame(
            data={'train_loss': results['train_loss'], 'test_loss': results['test_loss'],
                  'train_accuracy': results['train_accuracy'], 'test_accuracy': results['test_accuracy']},
            index=range(1, state['epoch'] + 1))
        data_frame.to_csv(out_path + DATA_TYPE + '_results.csv', index_label='epoch')

    # features visualization
    test_image, _ = next(iter(get_iterator(False, DATA_TYPE, 25)))
    test_image_logger.log(make_grid(test_image, nrow=5, normalize=True).numpy())
    if torch.cuda.is_available():
        test_image = test_image.cuda()
    feature_image = grad_cam(test_image)
    feature_image_logger.log(make_grid(feature_image, nrow=5, normalize=True).numpy())


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train Capsule Classfication')
    parser.add_argument('--data_type', default='MNIST', type=str,
                        choices=['MNIST', 'FashionMNIST', 'SVHN', 'CIFAR10', 'CIFAR100', 'STL10'],
                        help='dataset type')
    parser.add_argument('--num_iterations', default=3, type=int, help='routing iterations number')
    parser.add_argument('--batch_size', default=100, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=100, type=int, help='train epochs number')

    opt = parser.parse_args()

    DATA_TYPE = opt.data_type
    NUM_ITERATIONS = opt.num_iterations
    BATCH_SIZE = opt.batch_size
    NUM_EPOCHS = opt.num_epochs

    results = {'train_loss': [], 'test_loss': [], 'train_accuracy': [], 'test_accuracy': []}

    class_name = CLASS_NAME[DATA_TYPE]
    CLASSES = 10
    if DATA_TYPE == 'CIFAR100':
        CLASSES = 100

    model = models[DATA_TYPE](NUM_ITERATIONS)
    loss_criterion = nn.CrossEntropyLoss()
    grad_cam = GradCam(model)
    if torch.cuda.is_available():
        model.cuda()
        loss_criterion.cuda()

    print("# parameters:", sum(param.numel() for param in model.parameters()))

    optimizer = Adam(model.parameters())

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
    confusion_meter = tnt.meter.ConfusionMeter(CLASSES, normalized=True)

    train_loss_logger = VisdomPlotLogger('line', env=DATA_TYPE, opts={'title': 'Train Loss'})
    train_accuracy_logger = VisdomPlotLogger('line', env=DATA_TYPE, opts={'title': 'Train Accuracy'})
    test_loss_logger = VisdomPlotLogger('line', env=DATA_TYPE, opts={'title': 'Test Loss'})
    test_accuracy_logger = VisdomPlotLogger('line', env=DATA_TYPE, opts={'title': 'Test Accuracy'})
    confusion_logger = VisdomLogger('heatmap', env=DATA_TYPE,
                                    opts={'title': 'Confusion Matrix', 'columnnames': class_name,
                                          'rownames': class_name})
    test_image_logger = VisdomLogger('image', env=DATA_TYPE,
                                     opts={'title': 'Test Image', 'width': 371, 'height': 335})
    feature_image_logger = VisdomLogger('image', env=DATA_TYPE,
                                        opts={'title': 'Feature Image', 'width': 371, 'height': 335})

    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    engine.train(processor, get_iterator(True, DATA_TYPE, BATCH_SIZE), maxepoch=NUM_EPOCHS, optimizer=optimizer)
