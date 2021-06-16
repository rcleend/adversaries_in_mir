import json
import torch
import librosa
import warnings

from scipy.io.wavfile import write


def load_file(file_name, sample_rate, device):
    """ Loads file with librosa, resamples and converts it to mono. """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data, sr = librosa.load(file_name, sr=sample_rate)
    data = torch.tensor(data)
    return data.to(device).detach()


def save_adversary(perturbation, file_path):
    """ Saves adversarial data. """
    write(str(file_path.with_suffix('.wav')), 22050, perturbation.view(-1, 1).numpy())


def dump_json(file_path, attr_dict):
    """ Dumps a given AttrDict to a file with json. """
    json.dump(dict(attr_dict), open(file_path, 'w'))


def load_json(file_path):
    """ Loads json file. """
    res_dict = json.load(open(file_path))
    return res_dict
