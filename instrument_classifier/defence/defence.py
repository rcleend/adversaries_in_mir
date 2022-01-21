from ast import keyword
import os
import torch
import torch.nn as nn
from instrument_classifier.evaluation.evaluation_utils import get_data, get_network

def _avg_pred_dict(predictions): 
    avg_pred = {}
    for key, value in predictions.items():
        y = value['y']
        y_pred = value['y_pred'] / n_defence_nets
        y_pred_prob = torch.max(nn.functional.softmax(y_pred, dim=1))
        y_pred_class = torch.argmax(y_pred, dim=1)

        avg_pred[key] = {'y': y, 'y_pred_class': y_pred_class, 'y_pred_prob': y_pred_prob} 
    print(avg_pred)

def _update_pred_dict(predictions, key, values):
    if key in predictions:
        predictions[key]['y_pred'] += values['y_pred']
    else:
        predictions[key] =  values
    return predictions


def _eval_def_nets(def_nets, data_loader, device):
    # Iterate through all the defence networks
    dataset_size = len(data_loader.dataset)
    predictions = {}
    for i, net in enumerate(def_nets):
        for j, (x, y, sample_name) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)  # Move the data to the device that is used
            y_pred = net(x)
            
            # y_pred_prob = torch.max(nn.functional.softmax(y_pred, dim=1))
            # y_pred_class = torch.argmax(y_pred, dim=1)
            print(f'net: {i +1 }, sample {j + 1}/{n_defence_nets}')

            predictions = _update_pred_dict(predictions, key=sample_name[0], values={'y': y, 'y_pred': y_pred})
    
    _avg_pred_dict(predictions)

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