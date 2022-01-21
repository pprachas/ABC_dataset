import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sys

num_seg = 2100

print(num_seg)
#-----------------Import Graphs--------------------------#
f_path_og = 'Graphs/g_seg'+str(n_segment)+'/unnormalized/raw/g_batch/og/G.gpickle'
f_path_rx = 'Graphs/g_seg'+str(n_segment)+'/unnormalized/raw/g_batch/rx/G_rx.gpickle'
f_path_ry = 'Graphs/g_seg'+str(n_segment)+'/unnormalized/raw/g_batch/ry/G_ry.gpickle'
f_path_rxy = 'Graphs/g_seg'+str(n_segment)+'/unnormalized/raw/g_batch/rxy/G_rxy.gpickle'
print(f_path_og)
graph_og = nx.read_gpickle(f_path_og)
graph_rx = nx.read_gpickle(f_path_rx)
graph_ry = nx.read_gpickle(f_path_ry)
graph_rxy = nx.read_gpickle(f_path_rxy)

for jj in range(0,len(graph_og)):
  feat = []
  pos = []
  for kk in range(0,len(graph_og[jj])):
    feat.append(graph_og[jj].nodes(data = 'feature')[kk])
    pos.append(graph_og[jj].nodes(data='feature')[kk][0:2])
  
  pos = np.asarray(pos)
  feat = np.asarray(feat)
  max_feat = np.max(feat,axis = 0)
  min_feat = np.min(feat,axis = 0)

  for kk in range(0,len(max_feat)):
    feat[:,kk] = (feat[:,kk]-min_feat[kk])/(max_feat[kk]-min_feat[kk])
  for kk in range(0,len(graph_og[jj])):
    graph_og[jj].nodes[kk]['feature'] = feat[kk]
    graph_og[jj].nodes[kk]['pos'] = pos[kk]
  
nx.write_gpickle(graph_og,'Graphs/g_seg'+str(num_seg)+'/normalized/raw/g_batch/og/G_og.gpickle')
#----------------------------Graph reflect over x axis----------------------#
for jj in range(0,len(graph_rx)):
  feat = []
  pos = []
  for kk in range(0,len(graph_rx[jj])):
    feat.append(graph_rx[jj].nodes(data = 'feature')[kk])
    pos.append(graph_rx[jj].nodes(data='feature')[kk][0:2])
  
  pos = np.asarray(pos)
  feat = np.asarray(feat)
  max_feat = np.max(feat,axis = 0)
  min_feat = np.min(feat,axis = 0)

  for kk in range(0,len(max_feat)):
    feat[:,kk] = (feat[:,kk]-min_feat[kk])/(max_feat[kk]-min_feat[kk])
  for kk in range(0,len(graph_rx[jj])):
    graph_rx[jj].nodes[kk]['feature'] = feat[kk]
    graph_rx[jj].nodes[kk]['pos'] = pos[kk] 
  
nx.write_gpickle(graph_rx,'Graphs/g_seg'+str(num_seg)+'/normalized/raw/g_batch/rx/G_rx.gpickle')
#----------------------------Graph reflect over y axis----------------------#
for jj in range(0,len(graph_ry)):
  feat = []
  pos = []
  for kk in range(0,len(graph_ry[jj])):
    feat.append(graph_ry[jj].nodes(data = 'feature')[kk])
    pos.append(graph_ry[jj].nodes(data='feature')[kk][0:2])
  
  pos = np.asarray(pos)
  feat = np.asarray(feat)
  max_feat = np.max(feat,axis = 0)
  min_feat = np.min(feat,axis = 0)

  for kk in range(0,len(max_feat)):
    feat[:,kk] = (feat[:,kk]-min_feat[kk])/(max_feat[kk]-min_feat[kk])
  for kk in range(0,len(graph_ry[jj])):
    graph_ry[jj].nodes[kk]['feature'] = feat[kk]
    graph_ry[jj].nodes[kk]['pos'] = pos[kk] 
  
nx.write_gpickle(graph_ry,'Graphs/g_seg'+str(num_seg)+'/normalized/raw/g_batch/ry/G_ry.gpickle')
#----------------------------Graph reflect over y=x axis----------------------#
for jj in range(0,len(graph_rxy)):
  feat = []
  pos = []
  for kk in range(0,len(graph_rxy[jj])):
    feat.append(graph_rxy[jj].nodes(data = 'feature')[kk])
    pos.append(graph_rxy[jj].nodes(data='feature')[kk][0:2])
  
  pos = np.asarray(pos)
  feat = np.asarray(feat)
  max_feat = np.max(feat,axis = 0)
  min_feat = np.min(feat,axis = 0)

  for kk in range(0,len(max_feat)):
    feat[:,kk] = (feat[:,kk]-min_feat[kk])/(max_feat[kk]-min_feat[kk])
  for kk in range(0,len(graph_rxy[jj])):
    graph_rxy[jj].nodes[kk]['feature'] = feat[kk]
    graph_rxy[jj].nodes[kk]['pos'] = pos[kk] 
  
nx.write_gpickle(graph_rxy,'Graphs/g_seg'+str(num_seg)+'/normalized/raw/g_batch/rxy/G_rxy.gpickle')