import torch
import torchaudio

from recommender.data.data_utils import get_single_kl


def kl_loss(source, delta, target, model, alpha):
    """ Computes KL distance (for optimisation). """
    source_gaussian = model(source + delta)
    target_gaussian = model(target)

    if len(source_gaussian) != len(target_gaussian):
        raise ValueError('shapes should be equal!')

    kl_divs = torch.stack([get_single_kl(s, t) for s, t in zip(source_gaussian, target_gaussian)])

    return torch.mean(kl_divs)


def kl_delta_loss(source, delta, target, model, alpha):
    """ Computes KL distance and tries to minimise norm of delta for optimisation. """
    source_gaussian = model(source + delta)
    target_gaussian = model(target)

    if not (len(source_gaussian) == len(target_gaussian) == len(delta)):
        raise ValueError('shapes should be equal!')

    loss = torch.stack([get_single_kl(s, t) * alpha + torch.sum(d ** 2)
                        for s, t, d in zip(source_gaussian, target_gaussian, delta)])
    return torch.mean(loss)


def kl_multi_scale_loss(source, delta, target, model, alpha):
    """ Multi-scale loss and KL distance combination. """
    source_gaussian = model(source + delta)
    target_gaussian = model(target)

    kl_divs = torch.stack([get_single_kl(s, t) for s, t in zip(source_gaussian, target_gaussian)])
    if not (len(source) == len(kl_divs) == len(delta)):
        raise ValueError('shapes should be equal!')

    fft_sizes = [2048, 1024, 512, 256, 128, 64]
    eps = 1e-7
    multi_scale_alpha = 1.
    losses = []

    for s, k, d in zip(source, kl_divs, delta):
        spectral_loss = 0
        for n_fft in fft_sizes:
            spec_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=int(0.25 * n_fft),
                                                               window_fn=torch.hann_window, power=1).to(source.device)
            # compute magnitude specs
            spec_1 = torch.abs(spec_transform.forward(s.view(1, -1)))
            spec_2 = torch.abs(spec_transform.forward((s + d).view(1, -1)))

            log_diff = torch.log(spec_1 + eps) - torch.log(spec_2 + eps)
            diff = torch.mean(torch.abs(spec_1 - spec_2)) + multi_scale_alpha * torch.mean(torch.abs(log_diff))
            spectral_loss += diff
        losses.append(k * alpha + spectral_loss)

    return torch.mean(torch.stack(losses))
