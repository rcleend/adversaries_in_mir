import torch
import librosa
import numpy as np

from recommender.utils.io import load_file
from recommender.utils.paths import data_path
from recommender.utils.hubness import get_hubs
from recommender.attacks.soundpark_model import SoundparkModel
from recommender.data.data_utils import get_single_kl, get_middle
from recommender.attacks.kl_min_approx import get_kl_divs, get_gaussians


def get_white_noise(signal, snr):
    # see https://github.com/sleekEagle/audio_processing/blob/master/mix_noise.py
    """ Creates additive white noise for a signal with given SNR. """
    rms_s = np.sqrt(np.mean(signal.detach().cpu().numpy() ** 2))
    rms_n = np.sqrt(rms_s ** 2 / (10 ** (snr / 10)))
    noise = np.random.normal(0, rms_n, signal.shape)
    return torch.tensor(noise).float().to(signal.device)


def get_noisy_gaussian(file, snr, sr, nr_samples, sp_model, device):
    """ Adds noise to a file and computes and returns its gaussian. """
    # load data
    data = load_file(file, sr, device)
    # add additive white noise
    data = data + get_white_noise(data, snr)
    # get middle 2 minutes of data
    data = get_middle(nr_samples, data).to(device)
    # get gaussian
    gaussian = sp_model(data)

    return gaussian[0]


def test_noisy_hubs(files, snr, sr, nr_samples, device, hub_size):
    """ Runs test whether adding white noise changes non-hub to hub song. """
    # get initial gaussians / distances
    sp_model = SoundparkModel(device)
    gaussians = get_gaussians(None, None, device)
    kl_dists = get_kl_divs(gaussians, device)
    initial_hubs = get_hubs(kl_dists.detach(), hub_size)
    print('Initial hubs are: {}'.format(initial_hubs))

    print('=============================================================================')
    successful = 0
    for i, file in enumerate(files):
        # copy distances for now
        cur_dists = kl_dists.clone()

        # get gaussian for noisy version
        noisy_gaussian = get_noisy_gaussian(file, snr, sr, nr_samples, sp_model, device)

        print('Current file: {}'.format(file.stem))

        # update dists
        for j, j_file in enumerate(files):
            cur_dists[i, j] = cur_dists[j, i] = get_single_kl(noisy_gaussian, gaussians[j_file])

        current_hubs = get_hubs(cur_dists.detach(), hub_size)
        if i in current_hubs:
            print('Noise successful, file is now hub!')
            successful += 1
        else:
            print('Noise unsuccessful')
        print('---------------------------------------------------------------------------------------------')
    print('{}/{} files were successful!'.format(successful, len(files)))


def main():
    # define parameters
    snr = 40
    hub_size = 5

    # prepare paths, files
    files = sorted(list(data_path.rglob('*.mp3')))

    # prepare constants
    sr = 22050
    nr_samples = librosa.core.time_to_samples(60, sr=sr)
    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda:0')

    test_noisy_hubs(files, snr, sr, nr_samples, device, hub_size)


if __name__ == '__main__':
    main()
