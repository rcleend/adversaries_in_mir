import os
import attrdict as attr

from torch.utils.data import DataLoader
from instrument_classifier.utils.utils import load_net
from instrument_classifier.data.datasets import AudioDataset
from instrument_classifier.utils.attack_utils import get_files
from instrument_classifier.utils.avgpool_cnn import AveragePoolCNN
from instrument_classifier.utils.paths import d_path, adversary_path, log_path


def get_data(model_name, adversary, valid_set, backup_path=True):
    params = attr.AttrDict({'feature': 'torch', 'feature_length': None,
                            'pre_computed': False, 'sample_wise_norm': False})
    files = get_files(valid_set)
    if adversary is None:
        path = d_path if valid_set else os.path.join(d_path, 'test')
        backup_path = None
    else:
        path = os.path.join(adversary_path, adversary)
        backup_path = d_path if backup_path else None
        if not backup_path:
            files = list(set(files).intersection(set(os.listdir(path))))

    ads = AudioDataset(files, data_path=path, feature_dict=params,
                       norm_file_path=os.path.join(log_path, model_name, '{}.csv'), d_path_backup=backup_path)
    return DataLoader(ads, batch_size=1, shuffle=False)


def get_network(model_name, epoch):
    model = AveragePoolCNN(1, 12)
    return load_net(model_name, model, epoch)
