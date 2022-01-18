import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from shutil import copyfile

from instrument_classifier.utils.paths import *
from instrument_classifier.utils.utils import get_params
from instrument_classifier.utils.logging_utils import Logger
from instrument_classifier.utils.avgpool_cnn import AveragePoolCNN
from instrument_classifier.data.data_helper import compute_std_mean, get_single_label_files
from instrument_classifier.training.train_utils import save_checkpoint, get_train_loader, get_valid_loader, \
    check_and_create_training_dir


@torch.no_grad()
def _get_accuracy(preds: torch.Tensor, reals: torch.Tensor):
    return torch.mean((preds == reals).float())


def do_train_epoch(net, train_loader, criterion, optimiser, device):
    """ Performs one training epoch. """
    net.train()
    t_loss, t_acc = 0., 0.

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        logits = net(x)

        loss = criterion(logits, y)
        t_loss += loss.item()
        t_acc += _get_accuracy(net.predict(x), y).item()

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    return t_loss / len(train_loader), t_acc / len(train_loader)


def do_valid_epoch(net, valid_loader, criterion, device):
    """ Performs one validation epoch. """
    if valid_loader is None:
        return 0., 0.
    net.eval()
    v_loss, v_acc = 0., 0.

    for x, y in valid_loader:
        x, y = x.to(device), y.to(device)
        logits = net(x)

        loss = criterion(logits, y)
        v_loss += loss.item()
        v_acc += _get_accuracy(net.predict(x), y).item()

    return v_loss / len(valid_loader), v_acc / len(valid_loader)


def run_training(train_loader, valid_loader, logger, params, save_path):
    """ Runs training incl logging and model-saving for defined epochs. """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = AveragePoolCNN(1, 12)
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(net.parameters(), lr=params.lr)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimiser, milestones=[params.drop_ep], gamma=params.drop_rate)

    for ep in range(1, params.epochs + 1):
        # do training
        train_loss, train_acc = do_train_epoch(net, train_loader, criterion, optimiser, device)
        if valid_loader is not None:
            # do validation, write to log
            valid_loss, valid_acc = do_valid_epoch(net, valid_loader, criterion, device)
            logger.append([ep, train_loss, train_acc, valid_loss, valid_acc])
        else:
            # write to log
            logger.append([ep, train_loss, train_acc])

        if ep % params.save_interval == 0:
            save_checkpoint(ep, net, optimiser, path=save_path)

        lr_scheduler.step()


def _preps(params, train_files, valid_files, logging_path):
    """ Prepares train- and valid-loader as well as logger. """
    # first compute and save mean / std for normalisation
    if params.sample_wise_norm:
        norm_file_path = None
    else:
        mean, std = compute_std_mean(train_files, training_data_path)
        np.savetxt(os.path.join(logging_path, 'mean.csv'), mean, delimiter=',')
        np.savetxt(os.path.join(logging_path, 'std.csv'), std, delimiter=',')
        norm_file_path = os.path.join(logging_path, '{}.csv')

    if params.validation_set:
        # prep data-loaders
        train_loader = get_train_loader(train_files, params, d_path, norm_file_path=norm_file_path)
        valid_loader = get_valid_loader(valid_files, params, d_path, norm_file_path=norm_file_path)
        print('using {} files for training, {} files for validation'.format(len(train_loader.dataset),
                                                                            len(valid_loader.dataset)))
        # prep logging-columns
        logger_columns = ['epoch', 'training loss', 'training accuracy',
                          'validation loss', 'validation accuracy']
    else:
        # prep data-loader
        train_loader = get_train_loader(train_files, params, d_path, norm_file_path=norm_file_path)
        valid_loader = None
        print('using {} files for training'.format(len(train_loader.dataset)))
        # prep logging-columns
        logger_columns = ['epoch', 'training loss', 'training accuracy']

    # prepare logger
    logger = Logger(os.path.join(logging_path, params.log_file), columns=logger_columns)

    return train_loader, valid_loader, logger


def _prep_files(validation_set: bool):
    """ Prepare training / validation set if necessary. """
    tot_files = sorted(get_single_label_files())
    if not validation_set:
        return tot_files, []
    rng = np.random.RandomState(21)
    rng.shuffle(tot_files)
    split_idx = int(len(tot_files) * 0.75)
    return tot_files[:split_idx], tot_files[split_idx:]


def main():
    # get params
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    param_file = os.path.join(cur_dir, 'params.txt')
    params = get_params(param_file)

    # prepare training dirs
    save_path, logging_path = check_and_create_training_dir(params.experiment, train_path)
    copyfile(param_file, os.path.join(logging_path, 'params.txt'))

    # prep data loaders, logger
    train_files, valid_files = _prep_files(params.validation_set)
    train_loader, valid_loader, logger = _preps(params, train_files, valid_files, logging_path)

    # run training
    run_training(train_loader, valid_loader, logger, params, save_path)


if __name__ == '__main__':
    main()
