# hubness computation based on https://github.com/VarIr/scikit-hubness/blob/master/skhubness/analysis/estimation.py
import torch
import numpy as np


def get_hubs(dists, hub_size=5):
    """ Performs hubness analysis and returns list of hubs. """
    hub = Hubness(k=5, hub_size=hub_size)
    hub_res = hub.score(dists)
    hub_samples = hub_res['hubs']
    return hub_samples


def get_k_occurrence(dists, idx, hub_size=5):
    """ Computes k-occurrence for a particular file. """
    hub = Hubness(k=5, hub_size=hub_size)
    hub_res = hub.score(dists)
    k_occurrence = hub_res['k_occurrence'][idx]
    return k_occurrence


class Hubness:
    """ Examine hubness characteristics of data. """
    def __init__(self, k=10, hub_size=2):
        self.k = k
        self.hub_size = hub_size
        self.hubness_measures = {}

    def _k_neighbours(self, dists, start, end):
        """ Return indices of nearest neighbors in precomputed distance matrix. """
        n_test, m_test = dists.shape
        rows = torch.arange(n_test).unsqueeze(-1)
        ds = dists.clone()
        ds[~torch.isfinite(ds)] = np.inf
        rp = torch.stack([torch.randperm(m_test) for _ in range(n_test)])
        d2s = ds[rows, rp]
        _, indices = torch.topk(d2s, k=self.k + 1, dim=-1, largest=False)
        return rp[rows, indices[:, start:end]].to(dists.device)

    def score(self, dists):
        """ Estimate hubness in a data set. """
        n_test, n_train = dists.shape
        k_neighbors = self._k_neighbours(dists, start=1, end=self.k + 1)

        # get rid of negative indices (when ANN does not find enough neighbors), compute k-occurrence
        mask = k_neighbors < 0
        if torch.any(mask):
            k_neighbors = k_neighbors[~mask]
        k_occurrence = torch.bincount(k_neighbors.view(-1), minlength=n_train)

        # get hubs and their occurrence
        hubs, = torch.nonzero(k_occurrence >= self.hub_size * self.k, as_tuple=True)
        hub_occurrence = k_occurrence[hubs].sum() / float(self.k * n_test)

        # store hubness measures in dictionary
        self.hubness_measures = {'hubs': hubs, 'hub_occurrence': hub_occurrence, 'k_occurrence': k_occurrence}

        # return results
        return self.hubness_measures
