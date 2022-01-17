import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import attrdict as attr
import time
import os

from instrument_classifier.utils.avgpool_cnn import AveragePoolCNN
from instrument_classifier.data.datasets import RawDataset, AudioDataset
from instrument_classifier.utils.attack_utils import get_files
from instrument_classifier.utils.paths import d_path, model_path
from instrument_classifier.train.save import save_model


def get_data_loader(valid=True, batch_size=1):
    files = get_files(valid)
    params = attr.AttrDict({'feature': 'torch', 'feature_length': None,
                            'pre_computed': False, 'sample_wise_norm': False})


    ads = AudioDataset(
                      files, 
                      data_path=d_path, 
                      feature_dict=params, 
                      valid=valid
                      )

    return DataLoader(ads, batch_size, shuffle=True, drop_last=True)


def train(net, optimizer, scheduler, criterion, data_loader, n_epoch, device, batch_size): 
    print('Start training Network')
    print('train set size: ', len(data_loader.dataset))
    start=time.time()


    for epoch in range(0,n_epoch):

        if epoch >= 90:
          scheduler.step()
          print(('lr = {}'.format(scheduler.get_lr())))

        net.train(True)  # Put the network in train mode
        for i, (x_batch, y_batch) in enumerate(data_loader):
            if len(x_batch)!=batch_size:
                continue
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

    eval(net, train_loader, device)

def eval(net, data_loader, device):
  print('Evaluating network on test data')
  print('testset size: ', len(data_loader.dataset))
  correct_total = 0

  net.eval()  # Put the network in eval mode
  for i, (x_batch, y_batch) in enumerate(data_loader):
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)  # Move the data to the device that is used

    y_pred = net(x_batch)
    y_pred_max = torch.argmax(y_pred, dim=1)

    correct_total += torch.sum(torch.eq(y_pred_max, y_batch)).item()
  print('correct total:',correct_total)
  print(f'Accuracy on the test set: {correct_total / len(data_loader.dataset):.3f}')




# Parameters ---------------------------------------------------------------
n_epoch = 1
batch_size = 1
is_training = True


# Train Network ------------------------------------------------------------

# Init training parameters
if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')

print('device: ', device)

net = AveragePoolCNN(1,12).to(device)
criterion = nn.CrossEntropyLoss()

train_loader = get_data_loader(valid=False, batch_size=batch_size)

test_loader = get_data_loader(valid=True, batch_size=batch_size)


# TODO: update parameters and learning rate
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# TODO: updata gamma and add to training function
scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=0.1)

print('batch_size: ', batch_size)

if is_training:
  train(
      net=net,
      criterion=criterion,
      data_loader=train_loader,
      optimizer=optimizer,
      batch_size=batch_size,
      device=device,
      scheduler=scheduler,
      n_epoch=n_epoch
      )

  save_model(net, n_epoch)

# net.load_state_dict(torch.load(os.path.join(model_path, 'self-trained/', model_name + '.tar')))

# Evaluate Network ---------------------------------------------------------

eval(
    net=net,
    data_loader=test_loader,
    device=device,
    )

