import torch
import torch.optim as optim
import recommender.attacks.kl_losses as kl_losses

from attrdict import AttrDict
from argparse import ArgumentParser

from recommender.utils.paths import cache_path
from recommender.utils.logging_utils import Logger
from recommender.data.data_utils import get_single_kl
from recommender.attacks.soundpark_model import SoundparkModel
from recommender.utils.hubness import get_hubs, get_k_occurrence
from recommender.data.soundpark_set import get_cached_subset_dataloader
from recommender.attacks.attack_helper import prep_paths, get_gaussians, get_kl_divs, get_target_hubs, init_delta, \
    check_convergence


config = AttrDict({
    'experiment_name': 'test',      # name to use for saving adversaries
    'start_file': 0,                # first file to compute adversary for
    'nr_files': 15750,              # how many files to use (15750 for all)
    'target_hub_size': 5,           # hub size for potential target hubs
    'hub_size': 5,                  # size of hubs (hub_size * 5 => k-occurrence)
    'max_epochs': 500,              # maximum number of epochs in which an adversary is searched
    'choose_hub': 'closest',        # method to pick target hub; one of 'closest', 'biggest', 'random'
    'rand_init': False,             # how to initialise adv. perturbation
    'lr': 1e-03,                    # lr for optimiser
    'loss_func': 'kl_delta_loss',   # optimisation function; one of 'kl_loss', 'kl_delta_loss', 'kl_multi_scale_loss'
    'alpha': 25.,                   # factor which focuses on min. kl div instead of perceptibility
    'clip_eps': 0.1                 # clipping factor for restricting perturbation
})


def argument_parsing():
    """ Parses command-line arguments, namely start file index and nr of files to be processed. """
    # prepare argument parser (for now: process start/end files)
    parser = ArgumentParser(description='Program to compute hubness adversaries')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--nr_files', type=int)
    args = parser.parse_args()
    if args.start:
        config['start_file'] = args.start
    if args.nr_files:
        config['nr_files'] = args.nr_files


def check_for_hubs(new_gaussians, indices, gaussians, kl_divs, not_convs, old_k_occs):
    """ Checks whether adversaries are already recognised as hubs. """
    # prepare all file-names for KL-div computation
    all_files = sorted(list(gaussians.keys()))
    new_update_mask = []
    k_occs = []
    for cur in range(len(new_gaussians)):
        if not not_convs[cur]:
            new_update_mask.append(0.), k_occs.append(old_k_occs[cur])
            continue

        idx = indices[cur]
        # compute new KL divergences
        new_kl = torch.stack([get_single_kl(new_gaussians[cur], gaussians[file]) for file in all_files])
        # store old KL divergence
        old_kl = kl_divs[idx].clone()
        # insert new divergences in distance matrix
        kl_divs[idx, :] = kl_divs[:, idx] = new_kl
        # compute updated hubs, restore KL divergences
        new_hubs = get_hubs(kl_divs.detach(), config.hub_size).to(indices.device)
        k_occ = get_k_occurrence(kl_divs.detach(), idx, config.hub_size)
        kl_divs[idx, :] = kl_divs[:, idx] = old_kl
        del old_kl
        new_update_mask.append(0. if idx in new_hubs else 1.), k_occs.append(k_occ)
    return torch.tensor(new_update_mask), torch.tensor(k_occs)


def compute_adversaries_batch(target_data, indices, middles, gaussians, kl_divs, init_hubs, pot_target_hubs, sp_model):
    """ Compute adversarial perturbation for a batch. """
    cpu_model = SoundparkModel(torch.device('cpu'))
    update_mask = torch.tensor([1. if idx not in init_hubs else 0. for idx in indices]).to(middles.device)
    k_occs = get_k_occurrence(kl_divs.detach(), indices, config.hub_size)
    if torch.sum(update_mask) == 0:
        return [0] * len(indices), middles, k_occs

    # get target hub indices / data
    target_hub_indices = get_target_hubs(indices, pot_target_hubs, kl_divs, config)
    target_hub_data = torch.stack([target_data[t.item()][-1] for t in target_hub_indices]).to(middles.device)

    # prepare delta, optimiser and loss function
    delta = init_delta(middles, config)
    optimiser = optim.Adam([delta], lr=config.lr)
    loss_func = getattr(kl_losses, config.loss_func)

    for e in range(config.max_epochs):
        optimiser.zero_grad()
        loss = loss_func(middles, delta, target_hub_data, sp_model, config.alpha)
        print('Epoch {}/{}, loss: {}, updates: {}, k_occs: {}\r'.format(e + 1, config.max_epochs, loss.item(),
                torch.sum(update_mask).item(), k_occs), flush=True, end='')
        loss.backward()
        with torch.no_grad():
            delta.grad = torch.sign(delta.grad) * update_mask.view(len(delta), 1, 1)
        optimiser.step()
        with torch.no_grad():
            delta.clamp_(min=-config.clip_eps, max=config.clip_eps)

        # check whether attack was successful
        if e % 10 == 0:
            update_mask, k_occs = check_for_hubs(cpu_model((middles + delta).detach().cpu()), indices,
                                                 gaussians, kl_divs, update_mask, k_occs)
            update_mask = update_mask.to(middles.device)
        if torch.sum(update_mask) == 0:
            # all adversaries found, return
            return [not u for u in update_mask], delta.detach(), k_occs
    return [not u for u in update_mask], delta.detach(), k_occs


def compute_adversaries(files, adv_path):
    """ Compute adversaries for prepared data. """
    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda:0')
    sp_model = SoundparkModel(device)

    # prepare data
    first_file = config.start_file if config.start_file >= 0 else 0
    last_file = first_file + config.nr_files if first_file + config.nr_files <= len(files) else len(files)
    data = get_cached_subset_dataloader(cache_path, torch.arange(first_file, last_file), 1)
    # prepare logger
    logger = Logger(str(adv_path / 'log_{}_{}.txt').format(first_file, last_file), columns=['file', 'db', 'conv', 'kocc'])

    # get initial gaussians, KL divergences and hubs
    gaussians = get_gaussians(data, files, torch.device('cpu'))
    kl_divs = get_kl_divs(gaussians, torch.device('cpu'))
    init_hubs = get_hubs(kl_divs, config.hub_size).to(device)
    pot_target_hubs = get_hubs(kl_divs, config.target_hub_size)
    target_data = get_cached_subset_dataloader(cache_path, pot_target_hubs, 1).dataset

    for b, (indices, middles) in enumerate(data):
        print('\nBatch {}/{}'.format(b + 1, len(data)))
        indices, middles = indices.to(device), middles.to(device)
        # compute adversaries batch-wise, store it if converged
        convs, perts, k_occs = compute_adversaries_batch(target_data, indices, middles, gaussians, kl_divs,
                                                         init_hubs, pot_target_hubs, sp_model)
        check_convergence(convs, perts, k_occs, indices, middles, init_hubs, files, adv_path, logger)


def main():
    # parse arguments, get paths, list of files
    argument_parsing()
    adv_path, clean_files = prep_paths(config)
    # start attack
    compute_adversaries(clean_files, adv_path)


if __name__ == '__main__':
    main()
