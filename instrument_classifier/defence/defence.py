from ast import keyword
import os
import csv
import torch
import torch.nn as nn
from tqdm import tqdm
from instrument_classifier.evaluation.evaluation_utils import get_data, get_network
from instrument_classifier.utils.paths import misc_path
import pandas as pd


def _add_preds_to_csv(df):
    csv_path = os.path.join(misc_path, f'defences/defence_{defence_name}.csv')
    df.to_csv(csv_path, index=False) # Save CSV


def _get_pred(nets, data_loader, pred_name, device):
    all_pred_df = pd.DataFrame() # Create dataframe to store all the predictions

    dataset_size = len(data_loader.dataset) # Get dataset_size and use it as an input for the progress bar
    with tqdm(total=dataset_size, desc=f'Computing predictions {pred_name}', bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:

        # Iterate through all the samples
        for i, (x, y, sample_name) in enumerate(data_loader):
            pbar.update(1) # Update progress bar
            x, y = x.to(device), y.to(device)  # Move the data to the device that is used

            y_pred_sum = 0 # reset pred y value for each sample

            # Iterate through all the defence networks
            for net in nets:
                y_pred_sum += net(x) # Update sum of predicted y's
            
            y_avg = y_pred_sum /len(nets) # Calculate average predicted y based on sum of predicted y's
            y_avg_pred = torch.argmax(y_avg, dim=1) # Get the predicted class based on the average predicion of all the networkds
            y_avg_prob = torch.max(nn.functional.softmax(y_avg, dim=1)) # Get the probability of the average predicted class
            y_act_prob = nn.functional.softmax(y_avg[y.item()],dim=0) # Get the predicted probability of the actuall class


            # Create dataframe containing new prediction
            new_pred_df = pd.DataFrame(data=[[sample_name[0], y.item(), y_avg_pred.item(), y_act_prob.item(), y_avg_prob.item()]]) 
            new_pred_df.columns =['Sample Name', 'Label', f'Pred {pred_name}', f'Prob act {pred_name}', f'Prob pred {pred_name}']

            all_pred_df = all_pred_df.append(new_pred_df) # Add new prediction to dataframe containing all previous predictions
    
    return all_pred_df

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

# Create defence networks
nets = []
for i in range(n_defence_nets):
    model_name = f'defence_{i+1}'
    nets.append(get_network(model_name=model_name, epoch=-1).to(device)) # add defence network to nets array

# Get predictions of all the data using the original network
pred_orig_net = _get_pred(nets=[orig_net], data_loader=orig_loader, pred_name='orig net / orig data', device=device)
pred_fgsm_net = _get_pred(nets=[orig_net], data_loader=fgsm_loader, pred_name='orig net / FGSM data', device=device)
pred_pgdn_net = _get_pred(nets=[orig_net], data_loader=pgdn_loader, pred_name='orig net / PGDN data', device=device)

# Get predictions of all the data using the defence networks
pred_orig_def = _get_pred(nets=nets, data_loader=orig_loader, pred_name='defence / orig data', device=device)
pred_fgsm_def = _get_pred(nets=nets, data_loader=fgsm_loader, pred_name='defence / FGSM data', device=device)
pred_pgdn_def = _get_pred(nets=nets, data_loader=pgdn_loader, pred_name='defence / PGDN data', device=device)

# Concat all the pred df together into a total pred df
total_pred_df = pred_orig_net
total_pred_df = total_pred_df.merge(pred_fgsm_net, how='inner', on=['Sample Name', 'Label'])
total_pred_df = total_pred_df.merge(pred_pgdn_net, how='inner', on=['Sample Name', 'Label'])
total_pred_df = total_pred_df.merge(pred_orig_def, how='inner', on=['Sample Name', 'Label'])
total_pred_df = total_pred_df.merge(pred_fgsm_def, how='inner', on=['Sample Name', 'Label'])
total_pred_df = total_pred_df.merge(pred_pgdn_def, how='inner', on=['Sample Name', 'Label'])

# Add predictions to CSV
_add_preds_to_csv(total_pred_df)