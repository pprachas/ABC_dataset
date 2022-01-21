import numpy as np
import networkx as nx

#------------------Note-------------------------------------#
# This code takes in the generates the exact representation for sub-dataset 
# The nodes are the individual pixels and the edges connect the neighboring pixels
#-----------------File Paths---------------------------#
f_path = '/projectnb2/lejlab2/Jeffrey/donut/Struc_gen_FEA/intersect_donut_structure/intersect_donut_bulk/img/'

x = np.load(f_path+'x.npy',allow_pickle = True)
y = np.load(f_path+'y.npy',allow_pickle = True)

L = 800
w = 100

G_og = []
G_rx = []
G_ry = []
G_rxy = []

for ii in range(0,1):
  og = nx.Graph()
  rx = nx.Graph()
  ry = nx.Graph()
  rxy = nx.Graph()  
  for jj in range(0,len(x[ii])):
    og.add_node(jj,feature = np.asarray([x[ii][jj],y[ii][jj]]).reshape(-1))
  G_og.append(og)
  #-------------------------Reflect over x axis----------------------------#
  for jj in range(0,len(x[ii])):
    rx.add_node(jj,feature = np.asarray([w-x[ii][jj],y[ii][jj]]).reshape(-1))
  G_rx.append(rx) 
  #-------------------------Reflect over y axis----------------------------#
  for jj in range(0,len(x[ii])):
    ry.add_node(jj,feature = np.asarray([x[ii][jj],L-y[ii][jj]]).reshape(-1))
  G_ry.append(ry)
  
  #-------------------------Reflect over y=x axis----------------------------#
  for jj in range(0,len(x[ii])):
    rxy.add_node(jj,feature = np.asarray([w-x[ii][jj],L-y[ii][jj]]).reshape(-1))
  G_rxy.append(rxy)   

#-----------------------------Save Graphs--------------------------------------#
nx.write_gpickle(G_og,'Graphs/subdataset2/exact/raw/G_og.gpickle')
nx.write_gpickle(G_rx,'Graphs/subdataset2/exact/raw/G_rx.gpickle')
nx.write_gpickle(G_ry,'Graphs/subdataset2/exact/raw/G_ry.gpickle')
nx.write_gpickle(G_rxy,'Graphs/subdataset2/exact/raw/G_rxy.gpickle')
