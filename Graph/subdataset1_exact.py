import numpy as np
import math
import networkx as nx
import networkx as nx

#------------------Note-------------------------------------#
# This code takes in the generates the exact representation for sub-dataset 1
# The nodes are the individual pixels and the edges connect the neighboring pixels
#---------------------Initialize variables-----------------------------------#
n = 25000 # number of samples
num_stack = 38 # number of stacks in block

#----------------------------Load files---------------------------------------#
x = np.load('subdataset1_geo/x.npy')
l = np.load('subdatasetl_geo/l.npy')
L = 40
w = 5  
#-----------------------Generate Graph----------------------------------------#
G_all = []
for ii in range(0,len(x)): 
  
  G = nx.Graph()
  edges_hor = []
  edges_vert = []
  for jj in range(1,num_stack-1): 
    first_count = []
    first_c = []
    for c,kk in enumerate(range(x[ii][jj]-l[ii][jj],x[ii][jj]+l[ii][jj]+1)):
      prev = range(x[ii][jj-1]-l[ii][jj-1],x[ii][jj-1]+l[ii][jj-1]+1)
      G.add_node(G.number_of_nodes())
      G.nodes[G.number_of_nodes()-1]['feature'] = np.asarray([int(kk),int(40-jj)])
      if jj > 1: 
        for count,ll in enumerate(prev):
          if kk == ll:
            first_c.append(c)
            first_count.append(count)
          
            prev_connected = G.number_of_nodes()-1-len(prev)-first_c[0]+first_count[0]

            edges_vert.append((prev_connected,G.number_of_nodes()-1))
    
      if c != 0:
        edges_hor.append((G.number_of_nodes()-2,G.number_of_nodes()-1))
    edges_all = edges_vert+edges_hor
  G.add_edges_from(edges_all)
  G_all.append(G)
  print(ii)

print('Original Done')
print(len(G_all))
#--------------------------Reflect over y axis--------------------#
G_reflecty = []
for ii in range(0,len(x)): 
  
  G = nx.Graph()
  edges_hor = []
  edges_vert = []
  for jj in range(1,num_stack-1): 
    first_c = []
    for c,kk in enumerate(range(w-x[ii][jj]-l[ii][jj],w-x[ii][jj]+l[ii][jj]+1)):
      prev = range(w-x[ii][jj-1]-l[ii][jj-1],w-x[ii][jj-1]+l[ii][jj-1]+1)
      G.add_node(G.number_of_nodes())
      G.nodes[G.number_of_nodes()-1]['feature'] = np.asarray([int(kk),int(40-jj)])
      if jj > 1: 
        for count,ll in enumerate(prev):
          if kk == ll:
            first_c.append(c)
            first_count.append(count)
          
            prev_connected = G.number_of_nodes()-1-len(prev)-first_c[0]+first_count[0]

            edges_vert.append((prev_connected,G.number_of_nodes()-1))
    
      if c != 0:
        edges_hor.append((G.number_of_nodes()-2,G.number_of_nodes()-1))
    edges_all = edges_vert+edges_hor
  G.add_edges_from(edges_all)
  G_reflecty.append(G) 

#--------------------------Reflect over x axis--------------------#
G_reflectx = []
for ii in range(0,len(x)): 
  
  G = nx.Graph()
  edges_hor = []
  edges_vert = []
  for jj in range(1,num_stack-1): 
    first_count = []
    first_c = []
    for c,kk in enumerate(range(x[ii][jj]-l[ii][jj],x[ii][jj]+l[ii][jj]+1)):
      prev = range(x[ii][jj-1]-l[ii][jj-1],x[ii][jj-1]+l[ii][jj-1]+1)
      G.add_node(G.number_of_nodes())
      G.nodes[G.number_of_nodes()-1]['feature'] = np.asarray([int(kk),int(jj)])
      if jj > 1: 
        for count,ll in enumerate(prev):
          if kk == ll:
            first_c.append(c)
            first_count.append(count)
          
            prev_connected = G.number_of_nodes()-1-len(prev)-first_c[0]+first_count[0]

            edges_vert.append((prev_connected,G.number_of_nodes()-1))
    
      if c != 0:
        edges_hor.append((G.number_of_nodes()-2,G.number_of_nodes()-1))
    edges_all = edges_vert+edges_hor
  G.add_edges_from(edges_all)
  G_reflectx.append(G) 

#--------------------------Reflect over y=x axis--------------------#
G_reflectxy = []
for ii in range(0,len(x)): #len(X)
  
  G = nx.Graph()
  edges_hor = []
  edges_vert = []
  for jj in range(1,num_stack-1): 
    first_count = []
    first_c = []
    for c,kk in enumerate(range(w-x[ii][jj]-l[ii][jj],w-x[ii][jj]+l[ii][jj]+1)):
      prev = range(w-x[ii][jj-1]-l[ii][jj-1],w-x[ii][jj-1]+l[ii][jj-1]+1)
      G.add_node(G.number_of_nodes())
      G.nodes[G.number_of_nodes()-1]['feature'] = np.asarray([int(kk),int(jj)])
      if jj > 1: 
        for count,ll in enumerate(prev):
          if kk == ll:
            first_c.append(c)
            first_count.append(count)
          
            prev_connected = G.number_of_nodes()-1-len(prev)-first_c[0]+first_count[0]

            edges_vert.append((prev_connected,G.number_of_nodes()-1))
    
      if c != 0:
        edges_hor.append((G.number_of_nodes()-2,G.number_of_nodes()-1))
    edges_all = edges_vert+edges_hor
  G.add_edges_from(edges_all)
  G_reflectxy.append(G) 
#--------------------Save Graphs as pickle--------------------------------#
nx.write_gpickle(G_all,'Graphs/subdataset1/exact/G_og.gpickle')
nx.write_gpickle(G_reflecty,'Graphs/subdataset1/exact/G_ry.gpickle')
nx.write_gpickle(G_reflectx,'Graphs/subdataset1/exact/G_rx.gpickle')
nx.write_gpickle(G_reflectxy,'Graphs/subdataset1/exact/G_rxy.gpickle')