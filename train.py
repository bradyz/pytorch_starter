import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable

from dataset import CIFAR10
from logger import Logger
from model import BasicNetwork
from utils import maybe_load, save
from parameters import Parameters


def train_or_test(net, opt, crit, log, data, params, is_train, is_first):
    if is_train:
        net.train()
    else:
        net.eval()

    # Metrics.
    losses = list()

    correct = 0
    total_samples = 0

    for inputs, targets in tqdm.tqdm(data, total=len(data), desc='Batch'):
        if params.use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets = Variable(inputs), Variable(targets)

        if is_train:
            opt.zero_grad()

        logits = net(inputs)
        loss = crit(logits, targets)

        if is_train and not is_first:
            loss.backward()

            opt.step()

        losses.append(loss.cpu().data[0])

        _, predicted = torch.max(logits.data, 1)
        correct += predicted.eq(targets.data).cpu().sum()

        total_samples += targets.size(0)

    loss = np.mean(losses)
    accuracy = correct / total_samples

    log.update(loss, accuracy, is_train)


def main(params):
    net = BasicNetwork(CIFAR10.in_channels, CIFAR10.num_classes)
    log = Logger()
    opt = optim.Adam(
            net.parameters(), lr=params.lr, weight_decay=params.weight_decay)

    state = {'net': net,
            'log': log,
            'opt': opt,
            'params': params}

    maybe_load(state, params.checkpoint_path)

    print(params)

    log.visible = params.use_vizdom

    crit = nn.CrossEntropyLoss()

    if params.use_cuda:
        net.cuda()

    data_train = CIFAR10.get_data(params.data_dir, True, params.batch_size)
    data_test = CIFAR10.get_data(params.data_dir, False, params.batch_size)

    for epoch in tqdm.trange(log.epoch, params.max_epoch, desc='Epoch'):
        is_first = epoch == 0

        train_or_test(

                net, opt, crit, log, data_train, params, True, is_first)

        train_or_test(
                net, opt, crit, log, data_test, params, False, is_first)

        log.set_epoch(epoch+1)

        if log.should_save():
            save(state, params.checkpoint_path)


if __name__ == '__main__':
    params = Parameters()
    params.parse()

    main(params)
