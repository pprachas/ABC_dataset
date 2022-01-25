import os.path as osp
import sys
import os

import torch

import networkx as nx
import numpy as np

from torch_geometric.utils import from_networkx, to_networkx, add_self_loops

from torch.nn import Sequential,Linear,LeakyReLU,Softmax
import torch.nn.functional as F
from torch_geometric.nn import GNNExplainer
from torch_geometric.data import Data

from torch_cluster import radius_graph

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
#---------------------------------Note--------------------------------------------------------#
# This code is used to try to visualize and interpret the GNN model. Results of GNNExplainer are 
# in the APpendix section of our manuscript. 
#----------------------------------Import Data (choose 1)-------------------------------------#
from simple_norm_segment_pyg_graph import * 

#dataset_og = simple_graph_norm_seg350(root = 'Graphs/g_seg350/normalized/') # original dataset
dataset_og = simple_graph_norm_seg600(root = 'Graphs/g_seg600/normalized/') # original dataset
#dataset_og = simple_graph_norm_seg1100(root = 'Graphs/g_seg1100/normalized/') # original dataset
#-------------------------Import Models---------------------------------------------#
from model_explain import *
#---------------------------shuffle dataset-----------------------------------#
torch.manual_seed(12345)
index = torch.randperm(25000)
dataset_og = dataset_og[index]

#------------------GNN explain on multiple structures-----------------------------#
for ii in range(20000,20008):
  
  print(ii)
  data = dataset_og[ii].clone()
  
  model = PointNet4Layers()
  
  if torch.cuda.is_available():
      DEVICE = torch.device("cuda")
  else:
      DEVICE = torch.device("cpu")
  print(DEVICE)
  model.to(DEVICE)
  
  seeds = np.arange(1,11)
  
  node_feat_mask_all = []
  for seed in seeds: # Loop over seeds
  
    #----------------------Radius graph as edge index--------------------------#
    edge_index = radius_graph(data.pos.float(),r=40,loop=True)
    data.edge_index = edge_index
    
    #--------------------Load model State--------------------------------------#
    
    model.load_state_dict(torch.load('ML_results/ball/seg/4_layers_1MLP_skip_emb64_AdamW_aug/g_seg600/20kpoints/ball40/seed'+str(seed)+'/model_state/epoch50.pt', map_location = DEVICE)['model_state_dict'])
    #---------------------GNNExplain-----------------------------------------#
    explainer = GNNExplainer(model, epochs = 150, return_type = 'raw')
     
    node_feat_mask, edge_mask = explainer.explain_graph( x = data.x, pos = data.pos, edge_index = data.edge_index, r = 40 )
    
    #-------------------------min-max scaling--------------------------------#
    edge_mask = (edge_mask-edge_mask.min())/(edge_mask.max()-edge_mask.min())
    edge_mask = edge_mask.cpu().detach().numpy()
    node_feat_mask_all.append(node_feat_mask)
    print(edge_mask)
    #--------------------------get rgb values---------------------------------#
    class MplColorHelper:
  
      def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
    
      def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)
      
    rgb = MplColorHelper('binary',0,1).get_rgb(edge_mask)
    rgb[:,-1] = edge_mask
    print(rgb[:,-1])
    
      #-------------------------Visualize Graph---------------------------------#
    nx_data = to_networkx(data, node_attrs = ['x','pos'])
    
    pos = nx.get_node_attributes(nx_data,'pos')
    
    fig = plt.figure(figsize = (1,8))
    ax = fig.add_subplot()
    ax.set_rasterization_zorder(2)
    nx.draw(nx_data, pos, node_size = 4, arrows = False, edge_color = rgb, node_color = [200./255.,0,0], 
    width = 2*edge_mask
    plt.savefig('gnn_explain/test/GNN_Explain_seed'+str(seed)+'input'+str(ii)+'.png', pad_inches =0.0, bbox_inches = 'tight')
plt.show()

