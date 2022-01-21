import os.path as osp
import sys

import torch

import networkx as nx
import numpy as np

from torch_geometric.data import Dataset, download_url
from torch_geometric.utils import from_networkx, add_self_loops
from torch_geometric.data import DataLoader

from torch.nn import Sequential,Linear,LeakyReLU,Softmax
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing,BatchNorm
from torch_geometric.nn import global_max_pool,global_add_pool

from torch_cluster import radius_graph
import matplotlib.pyplot as plt

#-------------------Load all Segmentation Dataset--------------------------------#
from pyg_graphs import * 

#-----------------------Load Model-----------------------------------#
from Pointnet_layer import *

#------------------------Select training datapoints-----------------------#
num_data = np.asarray([5000,10000,15000,20000], dtype = int)[int(sys.argv[2])-1]

#-----------------------Load specific dataset based on input---------------#
if int(sys.argv[1]) == 1:
  dataset_og = simple_graph_norm_sparse(root = 'Graphs/g_sparse/normalized/') # original dataset
  dataset_ry = simple_graph_norm_sparse_ry(root = 'Graphs/g_sparse/normalized/') # reflected over y axis
  dataset_rx = simple_graph_norm_sparse_rx(root = 'Graphs/g_sparse/normalized/') # reflected over y axis
  dataset_rxy = simple_graph_norm_sparse_rxy(root = 'Graphs/g_sparse/normalized/') # reflected over y axis
  print('Segmentation: sparse')
  f_seg = '/g_sparse/'

elif int(sys.argv[1]) == 2:
  dataset_og = simple_graph_norm_medium(root = 'Graphs/g_medium/normalized/') # original dataset
  dataset_ry = simple_graph_norm_medium_ry(root = 'Graphs/g_medium/normalized/') # reflected over y axis
  dataset_rx = simple_graph_norm_medium_rx(root = 'Graphs/g_medium/normalized/') # reflected over y axis
  dataset_rxy = simple_graph_norm_medium_rxy(root = 'Graphs/g_medium/normalized/') # reflected over y axis
  print('Segmentation: medium')
  f_seg = '/g_medium/'

elif int(sys.argv[1]) ==3 :
  dataset_og = simple_graph_norm_dense(root = 'Graphs/g_dense/normalized/') # original dataset
  dataset_ry = simple_graph_norm_dense_ry(root = 'Graphs/g_dense/normalized/') # reflected over y axis
  dataset_rx = simple_graph_norm_dense_rx(root = 'Graphs/g_dense/normalized/') # reflected over y axis
  dataset_rxy = simple_graph_norm_dense_rxy(root = 'Graphs/g_dense/normalized/') # reflected over y axis
  print('Segmentation: dense')
  f_seg = '/g_dense/'

#------------------------shuffle dataset with same permutations----------#
num_train = int(0.80*len(dataset_og)) # 80% of total data is used for training
num_test = int((len(dataset_og)-int(0.80*len(dataset_og)))/2)

torch.manual_seed(12345)
index = torch.randperm(25000)
dataset_og = dataset_og[index]
dataset_rx = dataset_rx[index]
dataset_ry = dataset_ry[index]
dataset_rxy = dataset_rxy[index]

num_data = int(num_data)
train_dataset_og = dataset_og[:num_train] # 80% as train
train_dataset_og = train_dataset_og[:num_data] # use portion of training data
train_dataset_rx = dataset_rx[:num_train] # 80% as train
train_dataset_rx = train_dataset_rx[:num_data] # use portion of training data
train_dataset_ry = dataset_ry[:num_train] # 80% as train
train_dataset_ry = train_dataset_ry[:num_data] # use portion of training data
train_dataset_rxy = dataset_rxy[:num_train] # 80% as train
train_dataset_rxy = train_dataset_rxy[:num_data] # use portion of training data

val_dataset = dataset_og[num_train:num_train+num_test]
test_dataset = dataset_og[num_train+num_test:]

#--------------------Shuffle and Split into test and traindataset-------------#
print('Number of unaugmented training data:',len(train_dataset_og))
print('Number of validation data:',len(val_dataset))
print('Number of test data:',len(test_dataset))

train_dataset = (train_dataset_og,train_dataset_rx,train_dataset_ry,train_dataset_rxy) # concatanate datasets for iteration
train_dataset = torch.utils.data.ConcatDataset(train_dataset)

print('Number of augmented training data:',len(train_dataset))

#-------------------------------------setup dataloader-----------------#
train_loader = DataLoader(train_dataset,batch_size = 64 , shuffle = True,num_workers = 10, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size = 64, shuffle = False,num_workers = 10, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = False,num_workers = 10, pin_memory=True)

#---------------------------Choose Model--------------------------------#
model = PointNet4Layers()
print(model)
f_lay = '4_layers_1MLP_skip_emb64_AdamW_aug'

#---------------------------Switch to GPU if available-------------------# 
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(DEVICE)
model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.

def train(model, optimizer, loader):
    model.train()
    
    total_loss = 0
    for batch,data in enumerate(loader):
        data = data.to(DEVICE)
        optimizer.zero_grad()  # Clear gradients.
        logits = model(data.pos,data.x,data.edge_index,data.batch) # Forward pass.
        loss = criterion(logits, data.y.long())  # Loss computation.
        loss.backward()  # Backward pass.
        optimizer.step()  # Update model parameters.
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(train_loader.dataset)


@torch.no_grad()

def test(model, loader):
    model.eval()

    total_correct = 0
    for data in loader:
        data = data.to(DEVICE)
        logits = model(data.pos,data.x,data.edge_index,data.batch)
        pred = torch.argmax(logits,dim=1)

        total_correct += (pred == data.y).sum()

    return total_correct / len(loader.dataset)

epochs = np.arange(1,51,dtype = int) 


train_acc = []
val_acc = []
loss = []
#--------------------Load initialization------------------------------------------#
model.load_state_dict(torch.load('init_models/init_model_seed'+sys.argv[3]+'.pt'))  
#----------------------Train-------------------------------------------------------#
for epoch in epochs: 
    loss.append(train(model, optimizer, train_loader))
    train_acc.append(test(model,train_loader))
    val_acc.append(test(model, val_loader))
    print('Epoch: {:02d}| Loss: {:.4f}| Train Accuracy: {:.4f}|Test Accuracy: {:.4f}'.format(epoch,loss[-1],train_acc[-1],val_acc[-1]))
    if epoch % 5 == 0:
            torch.save({'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'loss': loss[-1]},
                  'ML_results'+f_lay+f_seg+str(int(num_data/1000))+'kpoints/seed'+sys.argv[3]+'/model_state/epoch{}.pt'.format(epoch)) # save model state every 5 epochs
train_acc = torch.stack(train_acc).cpu().numpy()
val_acc = torch.stack(val_acc).cpu().numpy()

#--------------------------Loss and Error--------------------------------------#
plt.figure()
plt.title('Epochs vs Errors')
plt.plot(epochs,train_acc,label = 'Training Accuracy')
plt.plot(epochs,val_acc,label = 'Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.legend()
plt.savefig('ML_results'+f_lay+f_seg+str(int(num_data/1000))+'kpoints/seed'+sys.argv[3]+'/acc.png') # save train and validation error vs epochs

plt.figure()
plt.title('Epochs vs Training Loss')
plt.plot(epochs, loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.tight_layout()
plt.savefig('ML_results'+f_lay+f_seg+str(int(num_data/1000))+'kpoints/seed'+sys.argv[3]+'/loss.png') # save loss vs epochs

plt.show()