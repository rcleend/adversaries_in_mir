import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from torch.utils.data import DataLoader
import attrdict as attr

from instrument_classifier.utils.avgpool_cnn import AveragePoolCNN
from instrument_classifier.data.datasets import RawDataset, AudioDataset
from instrument_classifier.utils.attack_utils import get_files
from instrument_classifier.utils.paths import d_path, adversary_path, log_path

def get_data_loader():
    files = get_files()
    params = attr.AttrDict({'feature': 'torch', 'feature_length': None,
                            'pre_computed': False, 'sample_wise_norm': False})


    ads = AudioDataset(files, data_path=d_path, feature_dict=params)
    return DataLoader(ads, batch_size=1, shuffle=False)



# Parameters
use_cuda = True
n_epoch = 200
batch_size = 128

if use_cuda and torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')

print('Device: {}'.format(device))
print(device)


# Train Network

# Init Model, Loss Function, Optimizer, and Scheduler
net = AveragePoolCNN(1,12).to(device)

criterion = nn.CrossEntropyLoss()

# TODO: update parameters and learning rate
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# TODO: updata gamma
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

# Init DataLoader
data_loader = get_data_loader()



# Start Training
print('Start training Network')
start=time.time()

for epoch in range(0,n_epoch):

  net.train()  # Put the network in train mode
  for i, (x_batch, y_batch) in enumerate(data_loader):
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)  # Move the data to the device that is used
    
    optimizer.zero_grad()  # Set all currenly stored gradients to zero 

    y_pred = net(x_batch)

    loss = criterion(y_pred, y_batch)

    loss.backward()

    optimizer.step()

    # Compute relevant metrics
    
    y_pred_max = torch.argmax(y_pred, dim=1)  # Get the labels with highest output probability

    correct = torch.sum(torch.eq(y_pred_max, y_batch)).item()  # Count how many are equal to the true labels

    elapsed = time.time() - start  # Keep track of how much time has elapsed

    # Show progress every 20 batches 
    if not i % 20:
      print(f'epoch: {epoch}, time: {elapsed:.3f}s, loss: {loss.item():.3f}, train accuracy: {correct / batch_size:.3f}')
    
    correct_total = 0

#   net.eval()  # Put the network in eval mode
#   for i, (x_batch, y_batch) in enumerate(testloader):
#     x_batch, y_batch = x_batch.to(device), y_batch.to(device)  # Move the data to the device that is used

#     y_pred = net(x_batch)
#     y_pred_max = torch.argmax(y_pred, dim=1)

#     correct_total += torch.sum(torch.eq(y_pred_max, y_batch)).item()

#   print(f'Accuracy on the test set: {correct_total / len(testset):.3f}')

