import math
import torch
import numpy as np

from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib import cm as cm, pyplot as plt

from recommender.utils.hubness import get_k_occurrence
from recommender.utils.paths import save_path, data_path
from recommender.evaluation.log_processing import read_log_file


def plot_kocc_before_after(k_occs_before, k_occs_after):
    """ Plots result of `get_kocc_before_after`. """
    cmap = cm.get_cmap('viridis')
    plt.rcParams.update({'font.size': 20})
    custom_lines = [Line2D([0], [0], color=cmap(0.), lw=4), Line2D([0], [0], color=cmap(0.8), lw=4)]

    divs = k_occs_after[k_occs_before == 0] - k_occs_before[k_occs_before == 0]
    bins, counts = np.unique(divs, return_counts=True)
    plt.figure()
    plt.bar(bins, counts, color=[cmap(0.8) if d <= 0 else cmap(0.) for d in bins], width=1)
    plt.legend(custom_lines, ['increased', 'decreased / unchanged'], prop={'size': 20})
    plt.xlabel('Differences after - before')
    plt.ylabel('Difference counts')
    plt.xticks(np.arange(math.ceil(min(bins) / 15) * 15, math.ceil(max(bins) / 15) * 15 + 15, 15), fontsize=12)
    nr_increased = round(np.sum(divs > 0) / len(divs) * 100, 2)
    plt.title('Changes in k-occurrence after attack (for anti-hubs; {} % inc.)'.format(nr_increased))
    plt.show()

    divs = k_occs_after[k_occs_before != 0] - k_occs_before[k_occs_before != 0]
    bins, counts = np.unique(divs, return_counts=True)
    plt.figure()
    plt.bar(bins, counts, color=[cmap(0.8) if d <= 0 else cmap(0.) for d in bins], width=1)
    plt.legend(custom_lines, ['increased', 'decreased / unchanged'], prop={'size': 20})
    plt.xlabel('Differences after - before')
    plt.ylabel('Difference counts')
    plt.xticks(np.arange(math.ceil(min(bins) / 15) * 15, math.ceil(max(bins) / 15) * 15 + 15, 15), fontsize=12)
    nr_increased = round(np.sum(divs > 0) / len(divs) * 100, 2)
    plt.title('Changes in k-occurrence after attack (for non-hubs; {} % inc.)'.format(nr_increased))
    plt.show()


def get_kocc_before_after(log_file_path, hub_size=5, plot=True):
    """ Compares k-occurrence before and after an attack. """
    cols, log_entries = read_log_file(log_file_path)
    conv_idx, = np.asarray(cols == 'conv').nonzero()
    file_idx, = np.asarray(cols == 'file').nonzero()
    kocc_idx, = np.asarray(cols == 'kocc').nonzero()
    conv_idx, file_idx, kocc_idx = conv_idx.item(), file_idx.item(), kocc_idx.item()
    entries = log_entries[:, [file_idx, kocc_idx, conv_idx]]

    all_files = [f.name for f in sorted(list(data_path.rglob('*.mp3')))]
    kl_divs = torch.load(save_path / 'kls.pt', map_location=torch.device('cpu'))
    indices = [all_files.index(f) for f in entries[:, 0]]
    k_occs = get_k_occurrence(kl_divs, indices, hub_size).numpy()

    k_occs_before = k_occs[entries[:, -1] != 'hub']
    k_occs_after = np.array(entries[entries[:, -1] != 'hub', -2], dtype=int)

    print('k-occurrences before: {} +- {}'.format(np.mean(k_occs_before), np.std(k_occs_before)))
    print('k-occurrences after: {} +- {}'.format(np.mean(k_occs_after), np.std(k_occs_after)))

    if plot:
        plot_kocc_before_after(k_occs_before, k_occs_after)


def main():
    log_file_path = Path('/insert/path/to/logfile.txt')
    hub_size = 5

    get_kocc_before_after(log_file_path, hub_size)


if __name__ == '__main__':
    main()
