from ast import keyword
import os
import csv
import torch
import torch.nn as nn
from tqdm import tqdm
from instrument_classifier.evaluation.evaluation_utils import get_data, get_network
from instrument_classifier.utils.paths import misc_path
import pandas as pd


def _add_preds_to_csv(df, net_name):
    csv_path = os.path.join(misc_path, f'defences/defence_{defence_name}.csv')
    df.to_csv(csv_path, index=False) # Save CSV


def _get_pred(nets, data_loader, net_name, data_name, device):
    all_pred_df = pd.DataFrame() # Create dataframe to store all the predictions

    dataset_size = len(data_loader.dataset) # Get dataset_size and use it as an input for the progress bar
    with tqdm(total=dataset_size, desc=f'Computing {data_name} data predictions', bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:

        # Iterate through all the samples
        for i, (x, y, sample_name) in enumerate(data_loader):
            pbar.update(1) # Update progress bar
            x, y = x.to(device), y.to(device)  # Move the data to the device that is used

            y_pred_sum = 0 # reset pred y value for each sample

            # Iterate through all the defence networks
            for net in nets:
                y_pred_sum += net(x) # Update sum of predicted y's
            
            y_avg = y_pred_sum /len(nets) # Calculate average predicted y based on sum of predicted y's
            y_avg_class = torch.argmax(y_avg, dim=1) # Get the final predicted class based on the average predicion of all the networkds
            y_avg_prob = torch.max(nn.functional.softmax(y_avg, dim=1)) # Get the probability of the average predicted class

            # Create dataframe containing new prediction
            new_pred_df = pd.DataFrame(data=[[sample_name, y.item(), y_avg_class.item(), y_avg_prob.item()]]) 
            new_pred_df.columns =['Sample Name', 'Label', f'Pred {data_name} data on {net_name}', f'Prob pred {data_name} data on {net_name}']

            all_pred_df = all_pred_df.append(new_pred_df) # Add new prediction to dataframe containing all previous predictions
    
    # _add_pred_to_csv(all_pred_df, defence_name, data_name)

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

# Create multiple networks for all the defence models
n_defence_nets = 3 #TODO replace with automatic directory detection or parameter

# Add defence name, TODO: replace with input argument
defence_name = 'test'

# Create original network to get baseline prediction
orig_net = get_network(model_name='torch16s1f', epoch=-1).to(device) # epoch -1 loads the latest epoch available

nets = []
for i in range(n_defence_nets):
    model_name = f'defence_{i+1}'
    nets.append(get_network(model_name=model_name, epoch=-1).to(device)) # add defence network to nets array

pred_orig = _get_pred(nets=[orig_net], data_loader=orig_loader, net_name='orig', data_name='orig', device=device)
pred_fgsm = _get_pred(nets=[orig_net], data_loader=fgsm_loader, net_name='orig', data_name='FGSM', device=device)
pred_pgdn = _get_pred(nets=[orig_net], data_loader=pgdn_loader, net_name='orig', data_name='PGDN', device=device)

# Iterate through all the defence networks and return the average prob of the orig data pred
def_pred_orig = _get_pred(nets=nets, data_loader=orig_loader, net_name='def', data_name='orig', device=device)

# Iterate through all the defence networks and return the average prob of the orig data pred
def_pred_fgsm = _get_pred(nets=nets, data_loader=fgsm_loader, net_name='def', data_name='FGSM', device=device)

# Iterate through all the defence networks and return the average prob of the orig data pred
def_pred_pgdn = _get_pred(nets=nets, data_loader=pgdn_loader, net_name='def', data_name='PGDN', device=device)

# Concat all the pred df together into a total pred df
total_pred_df = pred_orig
total_pred_df = pd.concat([total_pred_df, pred_fgsm.reindex(total_pred_df.index)], axis=1)
total_pred_df = pd.concat([total_pred_df, pred_pgdn.reindex(total_pred_df.index)], axis=1)
total_pred_df = pd.concat([total_pred_df, def_pred_orig.reindex(total_pred_df.index)], axis=1)
total_pred_df = pd.concat([total_pred_df, def_pred_fgsm.reindex(total_pred_df.index)], axis=1)
total_pred_df = pd.concat([total_pred_df, def_pred_pgdn.reindex(total_pred_df.index)], axis=1)

_add_preds_to_csv(total_pred_df)