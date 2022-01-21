import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
import json

import matplotlib.pyplot as plt

import os.path as osp

import torch

import networkx as nx
import numpy as np


from torch_geometric.data import Data, DataLoader

from torch_geometric.utils import from_networkx

from simple_norm_segment_pyg_graph import * 
from models import *

#------------------------------Note---------------------------------------------------------#
#This code converts json files downloaded from dataset to netwrokx files. These networkx files
#can then be used in the code to create Pytorch geometric graphs structures
#------------------------------Decode JSON--------------------------------------------------#
G = []
for ii in range(0,25000):
  f = open('Graphs/g_seg600/json/G_og'+str(ii)+'.json')
  s_decode = json.load(f)
  s_decode = json.loads(s_decode)
  
  G_dejson=json_graph.node_link_graph(s_decode)
  
  nx.write_gpickle(G_dejson, 'Graphs/sparse/G_og'+str(ii)+'.gpickle')

