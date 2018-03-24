import numpy as np
import torch
import torch.nn as nn
import tqdm

from dataset import CIFAR10
from logger import Logger
from model import BasicNetwork
from utils import maybe_load, save, make_variable, modify_learning_rate
from parameters import DefaultParameters


def train_or_test(net, opt, crit, log, data, params, is_first):
    is_train = opt is not None

    if is_train:
        net.train()
    else:
        net.eval()

    losses = list()
    correct = 0
    total_samples = 0

    for inputs, targets in tqdm.tqdm(
            data, total=len(data), desc='Batch', disable=not params.use_tqdm):

        inputs = make_variable(inputs, params.use_cuda)
        targets = make_variable(targets, params.use_cuda)

        logits = net(inputs)
        loss = crit(logits, targets)

        if is_train and not is_first:
            opt.zero_grad()
            loss.backward()
            opt.step()

        _, predicted = torch.max(logits.data, 1)

        correct += predicted.eq(targets.data).cpu().sum()
        total_samples += targets.size(0)

        losses.append(loss.cpu().data[0])

    loss = np.mean(losses)
    accuracy = correct / total_samples

    if is_train:
        log.update(loss_train=loss, accuracy_train=accuracy)
    else:
        log.update(loss_test=loss, accuracy_test=accuracy)


def main(params):
    if params.dataset_name == 'CIFAR10':
        data_params = CIFAR10

    net = BasicNetwork(data_params.in_channels, data_params.num_classes)
    crit = nn.CrossEntropyLoss()
    log = Logger(params.use_visdom)
    opt = torch.optim.Adam(
            net.parameters(), lr=params.lr, weight_decay=params.weight_decay)

    # All the things to be saved and loaded.
    state = {'net': net,
            'log': log,
            'opt': opt,
            'params': params}

    maybe_load(state, params.checkpoint_path)

    if params.use_cuda:
        net.cuda()

    data_train = data_params.get_data(params.data_dir, True, params.batch_size)
    data_test = data_params.get_data(params.data_dir, False, params.batch_size)

    print(params)

    log.draw()

    for epoch in tqdm.trange(
            log.epoch, params.max_epoch,
            desc='Epoch', disable=not params.use_tqdm):

        if not params.use_tqdm:
            print('Epoch: %d' % epoch)

        modify_learning_rate(epoch, opt, params)

        train_or_test(net, opt, crit, log, data_train, params, epoch == 0)
        train_or_test(net, None, crit, log, data_test, params, epoch == 0)

        log.set_epoch(epoch+1)

        if log.should_save():
            save(state, params.checkpoint_path)


if __name__ == '__main__':
    params = DefaultParameters()
    params.parse()

    main(params)
