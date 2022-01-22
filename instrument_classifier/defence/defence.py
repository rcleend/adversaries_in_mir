from ast import keyword
import os
import csv
import torch
import torch.nn as nn
from tqdm import tqdm
from instrument_classifier.evaluation.evaluation_utils import get_data, get_network
from instrument_classifier.utils.paths import misc_path
import pandas as pd


def _add_pred_to_csv(all_pred_df, defence_name, data_name):
    csv_path = os.path.join(misc_path, f'defences/defence_{defence_name}.csv')

    if os.path.isfile(csv_path): # Check if CSV already exists
        csv_df = pd.read_csv(csv_path) # Read CSV

        # Append new columns to existing CSV dataframe
        csv_df[f'{data_name} pred label'] = all_pred_df[f'{data_name} pred label']
        csv_df[f'{data_name} pred prob'] = all_pred_df[f'{data_name} pred prob']

    all_pred_df.to_csv(csv_path, index=False) # Save CSV

    # y_avg_class = torch.argmax(y_avg, dim=1)
    # y_avg_prob = torch.max(nn.functional.softmax(y_avg, dim=1))

    # with open(os.path.join(misc_path, f'defences/defence_{data_name}.csv'), 'a', encoding='UTF8') as f:
    #     writer = csv.writer(f)

        # write the header
        # writer.writerow()

        # write the data
        # writer.writerow([sample_name, y.item(), y_avg_class.item(), y_avg_prob.item()])
        # TODO: add columns if sample already exists (for PGDN and FGSM)

    # print(sample_name, y.item(), y_avg_class.item(), y_avg_prob.item())





def _eval_def_nets(def_nets, data_loader, data_name, defence_name, device):
    all_pred_df = pd.DataFrame() # Create dataframe to store all the predictions

    dataset_size = len(data_loader.dataset) # Get dataset_size and use it as an input for the progress bar
    with tqdm(total=dataset_size, desc=f'Running defence on {data_name} samples', bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:

        # Iterate through all the samples
        for i, (x, y, sample_name) in enumerate(data_loader):
            pbar.update(1) # Update progress bar
            x, y = x.to(device), y.to(device)  # Move the data to the device that is used

            y_pred_sum = 0 # reset pred y value for each sample

            # Iterate through all the defence networks
            for net in def_nets:
                y_pred_sum += net(x) # Update sum of predicted y's
            
            y_avg = y_pred_sum /len(def_nets) # Calculate average predicted y based on sum of predicted y's
            # _add_pred_to_csv(data_name, sample_name[0], y, y_avg)

            y_avg_class = torch.argmax(y_avg, dim=1) # Get the final predicted class based on the average predicion of all the networkds
            y_avg_prob = torch.max(nn.functional.softmax(y_avg, dim=1)) # Get the probability of the average predicted class

            # Create dataframe containing new prediction
            new_pred_df = pd.DataFrame(data=[[sample_name, y.item(), y_avg_class.item(), y_avg_prob.item()]]) 
            new_pred_df.columns =['Sample Name', 'Label', f'{data_name} pred label', f'{data_name} pred prob']

            all_pred_df = all_pred_df.append(new_pred_df) # Add new prediction to dataframe containing all previous predictions
    
    _add_pred_to_csv(all_pred_df, defence_name, data_name)

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

# Add defence name, TODO: replace with input argument
defence_name = 'test'

nets = []
for i in range(n_defence_nets):
    model_name = f'defence_{i+1}'
    nets.append(get_network(model_name=model_name, epoch=-1).to(device)) # add defence network to nets array

# TODO: Make new csv file


# Iterate through all the defence networks and average their baseline probabilities
_eval_def_nets(def_nets=nets, data_loader=orig_loader, data_name='Original', defence_name=defence_name, device=device)

# Iterate through all the defence networks and average their FGSM probabilities
_eval_def_nets(def_nets=nets, data_loader=fgsm_loader, data_name='FGSM', defence_name=defence_name, device=device)

# Iterate through all the defence networks and average their PGDN probabilities
_eval_def_nets(def_nets=nets, data_loader=pgdn_loader, data_name='PGDN', defence_name=defence_name, device=device)

# Get the single label with the highest output probability

# Print to csv file: file_name, original pred, original pred pro, attack pred, attack pred prob, defence pred, defefence pred pro