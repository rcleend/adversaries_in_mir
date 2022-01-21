import os
from instrument_classifier.evaluation.evaluation_utils import get_data, get_network

# Create a dataloader for the attack samples
attack_loader = get_data(model_name='torch16s1f',adversary='FGSM',valid_set=True)

# Create a dataloader for the original samples
orig_loader = get_data(model_name='torch16s1f',adversary=None,valid_set=True)

# Create original network to get original prediction
orig_net = get_network(model_name='torch16s1f', epoch=200)

# Create multiple networks for all the defence models

# Iterate through all the networks and average their probabilities

# Get the single label with the highest output probability

# Print to csv file: file_name, original pred, original pred pro, attack pred, attack pred prob, defence pred, defefence pred pro