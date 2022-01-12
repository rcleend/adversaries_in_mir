import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

from instrument_classifier.utils.avgpool_cnn import AveragePoolCNN
from instrument_classifier.data.datasets import RawDataset, AudioDataset
from instrument_classifier.evaluation.evaluation_utils import get_data

# Train Network

# Init Model, Loss Function, Optimizer, and Scheduler
net = AveragePoolCNN(1,12)

criterion = nn.CrossEntropyLoss()

# TODO: update parameters and learning rate
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# TODO: Init Dataset and DataLoader
data_loader = get_data(model_name='hoi', adversary=None, valid_set=False)

print(data_loader)
