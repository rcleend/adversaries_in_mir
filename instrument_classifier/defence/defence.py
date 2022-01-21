import os
import torch
from instrument_classifier.evaluation.evaluation_utils import get_data, get_network

def eval_def_nets(def_nets, data_loader):
    # Iterate through all the defence networks
    dataset_size = len(data_loader.dataset)
    for i, net in enumerate(def_nets):
        correct_total = 0
        for j, (x, y) in enumerate(data_loader):
            y_pred = net(x)
            y_pred_max = torch.argmax(y_pred, dim=1)
            print(f'net: {i + 1}, sample {j} out of {dataset_size}')
            correct_total += torch.sum(torch.eq(y_pred_max, y)).item()
        print('correct total:',correct_total)
        # TODO: analyse and store prediction probabilities



# Create a dataloader for the FGSM attack samples
fgsm_loader = get_data(model_name='torch16s1f',adversary='FGSM',valid_set=True)

# Create a dataloader for the PGDN attack samples
pgdn_loader = get_data(model_name='torch16s1f',adversary='PGDN',valid_set=True)

# Create a dataloader for the original samples
orig_loader = get_data(model_name='torch16s1f',adversary=None,valid_set=True)

# Create original network to get baseline prediction
orig_net = get_network(model_name='torch16s1f', epoch=-1) # epoch -1 loads the latest epoch available

# Create multiple networks for all the defence models
n_defence_nets = 3 #TODO replace with automatic directory detection or parameter

nets = []
for i in range(n_defence_nets):
    model_name = f'defence_{i+1}'
    nets.append(get_network(model_name=model_name, epoch=-1)) # add defence network to nets array

# Iterate through all the defence networks and average their baseline probabilities
# eval_def_nets(nets, orig_loader)

# Iterate through all the defence networks and average their FGSM probabilities
eval_def_nets(nets, fgsm_loader)

# Iterate through all the defence networks and average their PGDN probabilities
eval_def_nets(nets, pgdn_loader)

# Get the single label with the highest output probability

# Print to csv file: file_name, original pred, original pred pro, attack pred, attack pred prob, defence pred, defefence pred pro