import argparse

import pandas as pd
import torch
import torchnet as tnt
from torch.optim import Adam
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger, VisdomLogger
from tqdm import tqdm

from model import MixNet
from utils import get_iterator, CLASS_NAME, MultiClassAccuracyMeter, MarginLoss


def processor(sample):
    data, labels, training = sample
    old_labels = labels
    if labels.dim() != 2:
        labels = torch.eye(CLASSES).index_select(dim=0, index=torch.tensor(labels, dtype=torch.long))

    if torch.cuda.is_available():
        data, labels = data.to('cuda'), labels.to('cuda')

    model.train(training)

    classes = model(data)
    # test multi, don't compute loss
    if old_labels.dim() == 2:
        loss = 0
    else:
        loss = loss_criterion(classes, labels)
    return loss, classes


def on_sample(state):
    state['sample'].append(state['train'])


def reset_meters():
    meter_accuracy.reset()
    meter_multi_accuracy.reset()
    meter_loss.reset()
    meter_confusion.reset()


def on_forward(state):
    # test multi
    if state['sample'][1].dim() == 2:
        meter_multi_accuracy.add(state['output'].detach(), state['sample'][1])
    else:
        meter_accuracy.add(state['output'].detach(), state['sample'][1])
        meter_confusion.add(state['output'].detach(), state['sample'][1])
        meter_loss.add(state['loss'].item())


def on_start_epoch(state):
    reset_meters()
    state['iterator'] = tqdm(state['iterator'])


def on_end_epoch(state):
    print('[Epoch %d] Training Loss: %.4f Training Accuracy: %.2f%%' % (
        state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

    train_loss_logger.log(state['epoch'], meter_loss.value()[0])
    train_accuracy_logger.log(state['epoch'], meter_accuracy.value()[0])
    train_confusion_logger.log(meter_confusion.value())
    results['train_loss'].append(meter_loss.value()[0])
    results['train_accuracy'].append(meter_accuracy.value()[0])

    # test single
    reset_meters()
    engine.test(processor, get_iterator(DATA_TYPE, 'test_single', BATCH_SIZE, USE_DA))
    test_single_loss_logger.log(state['epoch'], meter_loss.value()[0])
    test_single_accuracy_logger.log(state['epoch'], meter_accuracy.value()[0])
    test_confusion_logger.log(meter_confusion.value())
    results['test_single_loss'].append(meter_loss.value()[0])
    results['test_single_accuracy'].append(meter_accuracy.value()[0])
    print('[Epoch %d] Testing Single Loss: %.4f Testing Single Accuracy: %.2f%%' % (
        state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

    # test multi
    engine.test(processor, get_iterator(DATA_TYPE, 'test_multi', BATCH_SIZE, USE_DA))
    test_multi_accuracy_logger.log(state['epoch'], meter_multi_accuracy.value()[0])
    test_multi_confidence_accuracy_logger.log(state['epoch'], meter_multi_accuracy.value()[1])
    results['test_multi_accuracy'].append(meter_multi_accuracy.value()[0])
    results['test_multi_confidence_accuracy'].append(meter_multi_accuracy.value()[1])
    print('[Epoch %d] Testing Multi Accuracy: %.2f%% Testing Multi Confidence Accuracy: %.2f%%' % (
        state['epoch'], meter_multi_accuracy.value()[0], meter_multi_accuracy.value()[1]))

    # save best model
    global best_acc
    if meter_accuracy.value()[0] > best_acc:
        if NET_MODE == 'Capsule':
            torch.save(model.state_dict(), 'epochs/%s_%s_%s.pth' % (DATA_TYPE, NET_MODE, CAPSULE_TYPE))
        else:
            torch.save(model.state_dict(), 'epochs/%s_%s.pth' % (DATA_TYPE, NET_MODE))
        best_acc = meter_accuracy.value()[0]
    # save statistics at every 10 epochs
    if state['epoch'] % 10 == 0:
        out_path = 'statistics/'
        data_frame = pd.DataFrame(
            data={'train_loss': results['train_loss'], 'train_accuracy': results['train_accuracy'],
                  'test_single_loss': results['test_single_loss'],
                  'test_single_accuracy': results['test_single_accuracy'],
                  'test_multi_accuracy': results['test_multi_accuracy'],
                  'test_multi_confidence_accuracy': results['test_multi_confidence_accuracy']},
            index=range(1, state['epoch'] + 1))
        if NET_MODE == 'Capsule':
            data_frame.to_csv(out_path + DATA_TYPE + '_' + NET_MODE + '_' + CAPSULE_TYPE + '_results.csv',
                              index_label='epoch')
        else:
            data_frame.to_csv(out_path + DATA_TYPE + '_' + NET_MODE + '_results.csv', index_label='epoch')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train Classification')
    parser.add_argument('--data_type', default='MNIST', type=str,
                        choices=['MNIST', 'FashionMNIST', 'SVHN', 'CIFAR10', 'STL10'], help='dataset type')
    parser.add_argument('--net_mode', default='Capsule', type=str, choices=['Capsule', 'CNN'], help='network mode')
    parser.add_argument('--capsule_type', default='ps', type=str, choices=['ps', 'fc'],
                        help='capsule network type')
    parser.add_argument('--routing_type', default='k_means', type=str, choices=['k_means', 'dynamic'],
                        help='routing type')
    parser.add_argument('--num_iterations', default=3, type=int, help='routing iterations number')
    parser.add_argument('--batch_size', default=64, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=100, type=int, help='train epochs number')
    parser.add_argument('--use_da', action='store_true', help='use data augmentation or not')

    opt = parser.parse_args()

    DATA_TYPE = opt.data_type
    NET_MODE = opt.net_mode
    CAPSULE_TYPE = opt.capsule_type
    ROUTING_TYPE = opt.routing_type
    NUM_ITERATIONS = opt.num_iterations
    BATCH_SIZE = opt.batch_size
    NUM_EPOCHS = opt.num_epochs
    USE_DA = opt.use_da

    results = {'train_loss': [], 'train_accuracy': [], 'test_single_loss': [], 'test_single_accuracy': [],
               'test_multi_accuracy': [], 'test_multi_confidence_accuracy': []}

    class_name = CLASS_NAME[DATA_TYPE]
    CLASSES = 10

    model = MixNet(DATA_TYPE, NET_MODE, CAPSULE_TYPE, ROUTING_TYPE, NUM_ITERATIONS)
    loss_criterion = MarginLoss()
    if torch.cuda.is_available():
        model = model.to('cuda')

    print("# model parameters:", sum(param.numel() for param in model.parameters()))

    optimizer = Adam(model.parameters())

    # record current best test accuracy
    best_acc = 0

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
    meter_multi_accuracy = MultiClassAccuracyMeter()
    meter_confusion = tnt.meter.ConfusionMeter(CLASSES, normalized=True)

    train_loss_logger = VisdomPlotLogger('line', env=DATA_TYPE, opts={'title': 'Train Loss'})
    train_accuracy_logger = VisdomPlotLogger('line', env=DATA_TYPE, opts={'title': 'Train Accuracy'})
    test_single_loss_logger = VisdomPlotLogger('line', env=DATA_TYPE, opts={'title': 'Test Single Loss'})
    test_single_accuracy_logger = VisdomPlotLogger('line', env=DATA_TYPE, opts={'title': 'Test Single Accuracy'})
    test_multi_accuracy_logger = VisdomPlotLogger('line', env=DATA_TYPE, opts={'title': 'Test Multi Accuracy'})
    test_multi_confidence_accuracy_logger = VisdomPlotLogger('line', env=DATA_TYPE,
                                                             opts={'title': 'Test Multi Confidence Accuracy'})
    train_confusion_logger = VisdomLogger('heatmap', env=DATA_TYPE,
                                          opts={'title': 'Train Confusion Matrix', 'columnnames': class_name,
                                                'rownames': class_name})
    test_confusion_logger = VisdomLogger('heatmap', env=DATA_TYPE,
                                         opts={'title': 'Test Confusion Matrix', 'columnnames': class_name,
                                               'rownames': class_name})

    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    engine.train(processor, get_iterator(DATA_TYPE, 'train', BATCH_SIZE, USE_DA), maxepoch=NUM_EPOCHS,
                 optimizer=optimizer)
