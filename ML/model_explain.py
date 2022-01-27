import os.path as osp
import sys

import torch

import networkx as nx
import numpy as np


from torch_geometric.utils import  add_self_loops


from torch.nn import Sequential,Linear,LeakyReLU,Softmax
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing,BatchNorm
from torch_geometric.nn import global_max_pool,global_add_pool

from torch_cluster import radius_graph

#-----------------------Define Pointnet Layer---------------------------------#
class PointNetLayer(MessagePassing):
    def __init__(self, in_channels, out_channels): 
        # Message passing with "max" aggregation.
        super(PointNetLayer, self).__init__('max')
        
        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden node
        # dimensionality plus point dimensionality (=3).
        self.mlp = Sequential(Linear(in_channels+2, out_channels),
                              LeakyReLU(),
                              Linear(out_channels, out_channels)
                              )
        
    def forward(self, h, pos, edge_index):
        # Start propagating messages.
        return self.propagate(edge_index, h=h, pos=pos)
    
    def message(self,h_j, pos_j, pos_i):
        # h_j defines the features of neighboring nodes as shape [num_edges, in_channels]
        # pos_j defines the position of neighboring nodes as shape [num_edges, 3]
        # pos_i defines the position of central nodes as shape [num_edges, 3]

        input = pos_j - pos_i  # Compute spatial relation.

        if h_j is not None:
            # In the first layer, we may not have any hidden node features,
            # so we only combine them in case they are present.
            input = torch.cat([h_j, input], dim=-1)

        return self.mlp(input)  # Apply our final MLP.

class PointNet4Layers(torch.nn.Module):
    def __init__(self):
        super(PointNet4Layers, self).__init__()


        enc = 64
        self.conv1 = PointNetLayer(4, enc)
        self.bn1 = BatchNorm(enc)
        
        self.conv2 = PointNetLayer(enc, enc)
        self.bn2 = BatchNorm(enc)
        
        self.conv3 = PointNetLayer(enc, enc)
        self.bn3 = BatchNorm(enc)
        
        self.conv4 = PointNetLayer(enc, enc)
        self.bn4 = BatchNorm(enc)

        self.lin1 = Linear(enc*4,2)
        
    def forward(self, pos, x, batch, r, edge_index):
        
        edge_index = radius_graph(pos.float(),r=r,batch=batch,loop=True)
        
        # 3. Start bipartite message passing.
        h = self.conv1(h=x.float(), pos=pos.float(), edge_index=edge_index)
        h = self.bn1(h)
        h1 = F.leaky_relu(h)
        
        h = self.conv2(h=h1, pos=pos.float(), edge_index=edge_index)
        h = self.bn2(h)
        h2 = F.leaky_relu(h)
        
        h = self.conv3(h=h2, pos=pos.float(), edge_index=edge_index)
        h = self.bn3(h)
        h3 = F.leaky_relu(h)
        
        h = self.conv4(h=h3, pos=pos.float(), edge_index=edge_index)
        h = self.bn4(h)
        h4 = F.leaky_relu(h)

        # 4. Global Pooling.
        h = global_max_pool(torch.cat([h1,h2,h3,h4],dim=-1), batch)

        # 5. Classifier.
        h = self.lin1(h)

        return h
