from ast import keyword
import os
import torch
import torch.nn as nn
from instrument_classifier.evaluation.evaluation_utils import get_data, get_network

def _add_pred_to_csv(sample_name, y, y_avg):
    y_avg_class = torch.argmax(y_avg, dim=1)
    y_avg_prob = torch.max(nn.functional.softmax(y_avg, dim=1))

    print(sample_name, y, y_avg, y_avg_class, y_avg_prob)





def _eval_def_nets(def_nets, data_loader, device):
    # Iterate through all the defence networks
    dataset_size = len(data_loader.dataset)
    for i, (x, y, sample_name) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)  # Move the data to the device that is used

        # reset y values for each sample
        y_pred_sum = 0 

        for j, net in enumerate(def_nets):
            y_pred_sum += net(x) # Update sum of predicted y's
            
        y_avg = y_pred_sum /len(def_nets) # Calculate average predicted y based on sum of predicted y's
        _add_pred_to_csv(sample_name[0], y, y_avg)
    
    # _avg_pred_dict(predictions)

# Load Cuda device if available
if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')

print('device: ', device)


# Create a dataloader for the FGSM attack samples
fgsm_loader = get_data(model_name='torch16s1f',adversary='fgsm_test',valid_set=True)

# Create a dataloader for the PGDN attack samples
pgdn_loader = get_data(model_name='torch16s1f',adversary='pgdn_test',valid_set=True)

# Create a dataloader for the original samples
orig_loader = get_data(model_name='torch16s1f',adversary=None,valid_set=True)

# Create original network to get baseline prediction
orig_net = get_network(model_name='torch16s1f', epoch=-1).to(device) # epoch -1 loads the latest epoch available

# Create multiple networks for all the defence models
n_defence_nets = 3 #TODO replace with automatic directory detection or parameter

nets = []
for i in range(n_defence_nets):
    model_name = f'defence_{i+1}'
    nets.append(get_network(model_name=model_name, epoch=-1).to(device)) # add defence network to nets array

# Iterate through all the defence networks and average their baseline probabilities
_eval_def_nets(nets, orig_loader, device)

# Iterate through all the defence networks and average their FGSM probabilities
_eval_def_nets(nets, fgsm_loader, device)

# Iterate through all the defence networks and average their PGDN probabilities
_eval_def_nets(nets, pgdn_loader, device)

# Get the single label with the highest output probability

# Print to csv file: file_name, original pred, original pred pro, attack pred, attack pred prob, defence pred, defefence pred pro