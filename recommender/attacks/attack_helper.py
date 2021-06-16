import torch
import numpy as np

from pathlib import Path
from torch import distributions as dists

from recommender.data.data_utils import get_single_kl, snr_ker
from recommender.utils.hubness import get_k_occurrence
from recommender.utils.io import dump_json, save_adversary
from recommender.attacks.soundpark_model import SoundparkModel
from recommender.utils.paths import adversary_path, data_path, save_path


def prep_paths(config):
    """ Prepares paths and files. """
    # prep paths
    adv_path = adversary_path / config.experiment_name
    if not adv_path.exists():
        adv_path.mkdir()
    # prep files
    clean_files = sorted(list(data_path.rglob('*.mp3')))
    # save parameters
    dump_json(adv_path / 'config.json', config)
    return adv_path, clean_files


def get_gaussians(data, clean_files, device):
    """ Computes Gaussians or loads them if already stored. """
    gaussian_path = save_path / 'gaussians.pt' if save_path is not None else None
    if gaussian_path is not None and gaussian_path.exists():
        print('Loading existing gaussians...')
        str_dict = torch.load(gaussian_path, map_location=device)
        gaussians = {Path(k): str_dict[k] for k in str_dict.keys()}
    else:
        gaussians = {}
        soundpark_model = SoundparkModel(device)
        for b, (indices, middles) in enumerate(data):
            indices, middles = indices.to(device), middles.to(device)
            print('{}/{} batches processed...\r'.format(b, len(data)), flush=True, end='')
            res = soundpark_model(middles)
            gaussians.update({clean_files[i]: m for i, m in zip(indices, res)})
        if gaussian_path is not None:
            torch.save({str(k): gaussians[k] for k in gaussians.keys()}, gaussian_path)
    return gaussians


def get_kl_divs(gaussians, device):
    """ Compute KL-divergences or loads them if available. """
    kl_path = save_path / 'kls.pt' if save_path is not None else None
    if kl_path is not None and kl_path.exists():
        print('Loading existing KL divergences...')
        kl_divs = torch.load(kl_path, map_location=device)
    else:
        file_names = sorted(list(gaussians.keys()))
        kl_divs = torch.zeros(len(gaussians), len(gaussians))
        for i, f in enumerate(file_names):
            print('Working on file {}/{}\r'.format(i + 1, len(file_names)), flush=True, end='')
            for j in range(i, len(file_names)):
                kl_divs[i, j] = kl_divs[j, i] = get_single_kl(gaussians[f], gaussians[file_names[j]])
        if kl_path is not None:
            torch.save(kl_divs, kl_path)

    return kl_divs.detach().to(device)


def get_target_hubs(indices, target_hubs, kl_divs, config):
    """ Determines target hubs based on different strategies. """
    if config.choose_hub == 'closest':
        hub_indices = torch.argmin(torch.stack([kl_divs[indices, h] for h in target_hubs]), dim=0)
    elif config.choose_hub == 'biggest':
        k_occs = get_k_occurrence(kl_divs, target_hubs, config.target_hub_size)
        hub_indices = torch.repeat_interleave(torch.argmax(k_occs), len(indices))
    elif config.choose_hub == 'random':
        hub_indices = torch.randint(len(target_hubs), (len(indices),))
    else:
        raise ValueError('Please define valid target-hub-method (closest, biggest or random)!')
    return hub_indices


def init_delta(clean_data, config):
    """ Initialises delta randomly, or with zeros. """
    if not config.rand_init:
        delta = torch.zeros_like(clean_data).to(clean_data.device)
        delta.requires_grad = True
    else:
        dist = dists.Uniform(-config.lr, config.lr)
        delta = dist.sample(clean_data.shape).to(clean_data.device)
        delta.requires_grad = True
    return delta


def check_convergence(convs, perts, k_occs, indices, middles, init_hubs, files, adv_path, logger):
    """ Checks whether adversary was found / file was hub to begin with, and stores some adversaries. """
    assert len(convs) == len(perts) == len(k_occs) == len(indices) == len(middles)
    middles = middles.cpu()
    perts = perts.cpu()
    for i in range(len(convs)):
        # check whether we found adversary
        idx = indices[i]
        file_name = files[idx].name
        if idx in init_hubs:
            print('File was already hub ({})'.format(file_name))
            logger.append([file_name, snr_ker(middles[i].numpy(), middles[i].numpy()).item(), 'hub', k_occs[i].item()])
        elif convs[i]:
            # randomly save 10 % of files
            if np.random.choice([True, False], p=[0.1, 0.9]):
                save_adversary((middles[i] + perts[i]), adv_path / files[idx].with_suffix('.wav').name)
            logger.append([file_name, snr_ker(middles[i].numpy(), (middles[i] + perts[i]).numpy()).item(),
                           'yes', k_occs[i].item()])
        else:
            print('Could not find adversary for this file ({})'.format(file_name))
            logger.append([file_name, snr_ker(middles[i].numpy(), (middles[i] + perts[i]).numpy()).item(),
                           'no', k_occs[i].item()])
