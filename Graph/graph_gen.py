import numpy as np
import sys
import networkx as nx

import matplotlib.pyplot as plt

from skimage.future import graph
from skimage.transform import resize
from skimage import img_as_bool
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.measure import regionprops


#------------------Note-------------------------------------#
# This code takes in the generated domain images and applies SLIC segmentation. This code can be applied to all Subdatasets. 
# The graphs generated also include graphs of reflected images 

# To get the densities as reported in the paper, change n_seg to:

# Subdataset 1:
# - sparse: 350 segments
# - medium: 600 segments
# - dense: 1100 segments

# Subdataset 2:
# - sparse: 220 segments
# - medium: 400 segments
# - dense: 790 segments

# Subdataset 3:
# - sparse: 700 segments
# - medium: 1000 segments
# - dense: 2300 segments

L = 800
w = 100
n_segment = 2300

#--------------------import images-------------------------#
img = np.load('subdataset1_geo/simple_bulk/img/img.npy') # file path to array of image; change as needed to other subdatasets

G_og = []
G_rx = []
G_ry = []
G_rxy = []

#-------------------------For subdataset 1, resize image to 100x800-------------------------------------------------------#
# Note: skip this step for subdataset 2 and subdataet 3
new_img = []
for ii in range(0,len(img_raw)): 
  new_img.append(img_as_bool(resize(image = img_split[num][ii],output_shape = (800,100),order = 0))[int(L/40):int(L-L/40),:])
img = new_img
#--------------------------------------------------------------------------------------------------------------------------#


#----------------------Graph creation--------------------------------------------------------------------------------------#
for ii in range(0,25000):
    
    #----------create graph from original image-----------------------------#
    slic_pix = slic(img[ii], n_segments = n_segment, sigma = 0.01, compactness = 0.1)+1
    G = graph.rag_mean_color(img[ii],slic_pix, mode = 'similarity', sigma = 0.01)
    regions = regionprops(slic_pix)
    H = nx.Graph()
    H.add_nodes_from(G)
    H.add_edges_from(G.edges())
    
    
    tol = 1e-16 #tolerance to prune weights
    u_re = []
    v_re = []
    for u,v,d in G.edges(data = True):
        if d['weight'] <= tol:
            u_re.append(u)
            v_re.append(v)
    
    for kk in range(0,len(u_re)):
        H.remove_edge(u_re[kk],v_re[kk])
    
    for jj,region in enumerate(regions):
        print(jj)
        y = region['centroid'][0]
        x = region['centroid'][1]
        a = region['area']
        e = region['eccentricity']
        H.nodes[region.label]['feature'] = [x,L-y,a,e]
        
    
    components = [H.subgraph(c).copy() for c in sorted(nx.connected_components(H))]
    largest_component=max(components, key=len)
    largest_component = nx.convert_node_labels_to_integers(largest_component)
    largest_component.max_id = len(largest_component.nodes())
    
    #------------------------------Save graph-----------------------------#
    G_og.append(largest_component) #add to set of graph with features
    
    #----------create graph from image flipped over x axis-----------------------------#
    slic_pix = slic(new_img[ii][::-1,:], n_segments = n_segment, sigma = 0.01, compactness = 0.1)+1
    G = graph.rag_mean_color(new_img[ii][::-1,:],slic_pix, mode = 'similarity', sigma = 0.01)
    regions = regionprops(slic_pix)
    H = nx.Graph()
    H.add_nodes_from(G)
    H.add_edges_from(G.edges())
    
    
    tol = 1e-16 #tolerance to prune weights
    u_re = []
    v_re = []
    for u,v,d in G.edges(data = True):
        if d['weight'] <= tol:
            u_re.append(u)
            v_re.append(v)
    
    for kk in range(0,len(u_re)):
        H.remove_edge(u_re[kk],v_re[kk])
    
    for jj,region in enumerate(regions):
        print(jj)
        y = region['centroid'][0]
        x = region['centroid'][1]
        a = region['area']
        e = region['eccentricity']
        H.nodes[region.label]['feature'] = [x,L-y,a,e]
        
    
    components = [H.subgraph(c).copy() for c in sorted(nx.connected_components(H))]
    largest_component=max(components, key=len)
    largest_component = nx.convert_node_labels_to_integers(largest_component)
    largest_component.max_id = len(largest_component.nodes())
    
    #------------------------------Save graph-----------------------------#
    G_rx.append(largest_component) #add to set of graph with features

#----------create graph from image flipped over y axis-----------------------------#
    slic_pix = slic(new_img[ii][:,::-1], n_segments = n_segment, sigma = 0.01, compactness = 0.1)+1
    G = graph.rag_mean_color(new_img[ii][:,::-1],slic_pix, mode = 'similarity', sigma = 0.01)
    regions = regionprops(slic_pix)
    H = nx.Graph()
    H.add_nodes_from(G)
    H.add_edges_from(G.edges())
    
    
    tol = 1e-16 #tolerance to prune weights
    u_re = []
    v_re = []
    for u,v,d in G.edges(data = True):
        if d['weight'] <= tol:
            u_re.append(u)
            v_re.append(v)
    
    for kk in range(0,len(u_re)):
        H.remove_edge(u_re[kk],v_re[kk])
    
    for jj,region in enumerate(regions):
        print(jj)
        y = region['centroid'][0]
        x = region['centroid'][1]
        a = region['area']
        e = region['eccentricity']
        H.nodes[region.label]['feature'] = [x,L-y,a,e]
        
    
    components = [H.subgraph(c).copy() for c in sorted(nx.connected_components(H))]
    largest_component=max(components, key=len)
    largest_component = nx.convert_node_labels_to_integers(largest_component)
    largest_component.max_id = len(largest_component.nodes())
    
    #------------------------------Save graph-----------------------------#
    G_ry.append(largest_component) #add to set of graph with features

#----------create graph from image flipped over y=x axis-----------------------------#
    slic_pix = slic(new_img[ii][::-1,::-1], n_segments = n_segment, sigma = 0.01, compactness = 0.1)+1
    G = graph.rag_mean_color(new_img[ii][::-1,::-1],slic_pix, mode = 'similarity', sigma = 0.01)
    regions = regionprops(slic_pix)
    H = nx.Graph()
    H.add_nodes_from(G)
    H.add_edges_from(G.edges())
    
    
    tol = 1e-16 #tolerance to prune weights
    u_re = []
    v_re = []
    for u,v,d in G.edges(data = True):
        if d['weight'] <= tol:
            u_re.append(u)
            v_re.append(v)
    
    for kk in range(0,len(u_re)):
        H.remove_edge(u_re[kk],v_re[kk])
    
    for jj,region in enumerate(regions):
        print(jj)
        y = region['centroid'][0]
        x = region['centroid'][1]
        a = region['area']
        e = region['eccentricity']
        H.nodes[region.label]['feature'] = [x,L-y,a,e]
        
    
    components = [H.subgraph(c).copy() for c in sorted(nx.connected_components(H))]
    largest_component=max(components, key=len)
    largest_component = nx.convert_node_labels_to_integers(largest_component)
    largest_component.max_id = len(largest_component.nodes())
    
    #------------------------------Save graph-----------------------------#
    G_rxy.append(largest_component) #add to set of graph with features
#----------------------Save list of graphs to pickle-------------------------#
# If necessary, split up the list of graphs
nx.write_gpickle(G_og, 'Graphs/g_seg'+str(n_segment)+'/unnormalized/raw/g_batch/og/G.gpickle')
nx.write_gpickle(G_rx, 'Graphs/g_seg'+str(n_segment)+'/unnormalized/raw/g_batch/rx/G_rx.gpickle')
nx.write_gpickle(G_ry, 'Graphs/g_seg'+str(n_segment)+'/unnormalized/raw/g_batch/ry/G_ry.gpickle')
nx.write_gpickle(G_rxy, 'Graphs/g_seg'+str(n_segment)+'/unnormalized/raw/g_batch/rxy/G_rxy.gpickle')

