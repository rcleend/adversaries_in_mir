import os
import attrdict as attr

from torch.utils.data import DataLoader
from instrument_classifier.data.datasets import AudioDataset
import torch

def save_checkpoint(epoch, net, optimizer, path):
    """ Saves CNN and important parameters of current epoch. """
    if '~' in path:
        path = os.path.expanduser(path)
    if not os.path.exists(path):
        print('Please define valid save path for model checkpoint.\nAbort saving...')
        return
    torch.save({'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
               os.path.join(path, 'model_ep{}.tar').format(epoch))


def get_train_loader(files, params, d_path, norm_file_path):
    """ Returns shuffled data loader for training network. """
    if params.pre_computed:
        d_path = os.path.join(d_path, params.pre_computed)
    feature_dict = attr.AttrDict({'feature': params.feature, 'feature_length': params.feature_length,
                                  'pre_computed': params.pre_computed, 'sample_wise_norm': params.sample_wise_norm})
    ads = AudioDataset(files, d_path, feature_dict=feature_dict, norm_file_path=norm_file_path)
    return DataLoader(ads, batch_size=params.batch_size, shuffle=True)


def get_valid_loader(files, params, d_path, norm_file_path):
    """ Returns un-shuffled data loader for validating network during training. """
    if params.pre_computed:
        d_path = os.path.join(d_path, params.pre_computed)
    feature_dict = attr.AttrDict({'feature': params.feature, 'feature_length': None,
                                  'pre_computed': params.pre_computed, 'sample_wise_norm': params.sample_wise_norm})
    ads = AudioDataset(files, d_path, feature_dict=feature_dict, norm_file_path=norm_file_path)
    return DataLoader(ads, batch_size=1, shuffle=False)


def check_and_create_training_dir(experiment, training_path):
    """ Utils method that creates all directories necessary for training a model. """
    if '~' in training_path:
        training_path = os.path.expanduser(training_path)
    if not os.path.exists(training_path):
        os.mkdir(training_path)

    model_path = os.path.join(training_path, 'models')
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    model_save_path = os.path.join(model_path, experiment)
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    log_path = os.path.join(training_path, 'logs')
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    log_save_path = os.path.join(log_path, experiment)
    if not os.path.exists(log_save_path):
        os.mkdir(log_save_path)

    return model_save_path, log_save_path