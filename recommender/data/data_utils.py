import numpy as np
from torch.distributions import kl_divergence


def get_single_kl(p, q):
    """ Computes KL distance between two Gaussians. """
    return kl_divergence(p, q)


def get_middle(nr_samples, data):
    """ Returns middle 2*nr_samples of given data. """
    data_middle = data.shape[0] // 2
    data = data[data_middle - nr_samples:data_middle + nr_samples]
    return data


def snr_ker(orig, adv):
    """ Computes SNR according to https://github.com/coreyker/dnn-mgr/blob/master/utils/comp_ave_snr.py. """
    ign = 2048
    l = min(orig.shape[-1], adv.shape[-1])
    snr = 20 * np.log10(np.linalg.norm(orig[..., ign:l - ign - 1])
                        / np.linalg.norm(np.abs(orig[..., ign:l - ign - 1] - adv[..., ign:l - ign - 1]) + 1e-12))
    return snr
