import os.path as osp

import torch

import networkx as nx
import numpy as np

from torch_geometric.data import Dataset, download_url
from torch_geometric.utils import from_networkx, to_networkx
from torch_geometric.data import DataLoader
#--------------------------Note--------------------------------------------------#
#This code converts networkx graphs to PyG graphs. If the dataset is downloaded as
#json files, first convert the json files into networkx files (example code is given)
#
#Note that this code can be used to generate PyG graphs for all subdatasets and all node densities.

#:::::::::::::::::::::::::::SEGMENT sparse::::::::::::::::::::::::::::::::::::::#

#--------------------------------Original Dataset----------------------------#
class simple_graph_norm_sparse(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(simple_graph_norm_sparse, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
     f_names_raw = []
     for ii in range(0,25000):
       f_names_raw.append('subdataset1/Graphs/sparse/raw/og/G_og'+str(ii)+'.gpickle') # Change directory as needed
     return f_names_raw

    @property
    def processed_file_names(self):
        f_names_processed = []
        for ii in range(0,25000):
          f_names_processed.append('og/graph_og_'+str(ii)+'.pt') # Change directory as needed
        return f_names_processed

    def download(self):
        pass

    def process(self):
        labels = np.loadtxt('subdataset1/subdataset1_labels.txt') # Change directory as needed
        for ii in range(0,25000):        
          G = nx.read_gpickle('subdataset1/Graphs/sparse/raw/og/G_og'+str(ii)+'.gpickle') # Change directory as needed       
          graph = from_networkx(G)
          graph['y'] = labels[ii]
          graph['x'] = graph['feature']
          torch.save(graph, osp.join(self.processed_dir, 'og/graph_og_{}.pt'.format(ii))) # Change directory as needed
        
          
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'og/graph_og_{}.pt'.format(idx))) # Change directory as needed
        return data

#--------------------------------Reflect over y axis Dataset----------------#
class simple_graph_norm_sparse_ry(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(simple_graph_norm_sparse_ry, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
     f_names_raw = []
     for ii in range(0,25000):
       f_names_raw.append('subdataset1/Graphs/sparse/raw/og/G_ry'+str(ii)+'.gpickle') # Change directory as needed
     return f_names_raw

    @property
    def processed_file_names(self):
        f_names_processed = []
        for ii in range(0,25000):
          f_names_processed.append('ry/graph_ry_'+str(ii)+'.pt') # Change directory as needed
        return f_names_processed

    def download(self):
        pass

    def process(self):
        labels = np.loadtxt('subdataset1/subdataset1_labels.txt') # Change directory as needed
        #-------------------swapping labels----------------------------#
        indices_one = labels == 1
        indices_zero = labels == 0
        labels[indices_one] = 0 # replacing 1s with 0s
        labels[indices_zero] = 1 # replacing 0s with 1s
        
        for ii in range(0,25000):        
          G = nx.read_gpickle('subdataset1/Graphs/sparse/raw/og/G_og'+str(ii)+'.gpickle') # Change directory as needed       
          graph = from_networkx(G)
          graph['y'] = labels[ii]
          graph['x'] = graph['feature']
          torch.save(graph, osp.join(self.processed_dir, 'og/graph_og_{}.pt'.format(ii))) # Change directory as needed
          
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'ry/graph_ry_{}.pt'.format(idx))) # Change directory as needed
        return data
#-------------------------------Reflect over x axis Dataset----------------------------#
class simple_graph_norm_sparse_rx(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(simple_graph_norm_sparse, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
     f_names_raw = []
     for ii in range(0,25000):
       f_names_raw.append('subdataset1/Graphs/sparse/raw/rx/G_rx'+str(ii)+'.gpickle') # Change directory as needed
     return f_names_raw

    @property
    def processed_file_names(self):
        f_names_processed = []
        for ii in range(0,25000):
          f_names_processed.append('rx/graph_rx_'+str(ii)+'.pt') # Change directory as needed
        return f_names_processed

    def download(self):
        pass

    def process(self):
        labels = np.loadtxt('subdataset1/subdataset1_labels.txt') # Change directory as needed
        for ii in range(0,25000):        
          G = nx.read_gpickle('subdataset1/Graphs/sparse/raw/rx/G_rx'+str(ii)+'.gpickle') # Change directory as needed        
          graph = from_networkx(G)
          graph['y'] = labels[ii]
          graph['x'] = graph['feature']
          torch.save(graph, osp.join(self.processed_dir, 'rx/graph_rx_{}.pt'.format(ii))) # Change directory as needed
        
          
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'rx/graph_rx_{}.pt'.format(idx))) # Change directory as needed
        return data

#--------------------------------Reflect over y axis Dataset----------------#
class simple_graph_norm_sparse_rxy(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(simple_graph_norm_sparse_rxy, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
     f_names_raw = []
     for ii in range(0,25000):
       f_names_raw.append('subdataset1/Graphs/sparse/raw/og/G_rxy'+str(ii)+'.gpickle') # Change directory as needed
     return f_names_raw

    @property
    def processed_file_names(self):
        f_names_processed = []
        for ii in range(0,25000):
          f_names_processed.append('rxy/graph_rxy_'+str(ii)+'.pt') # Change directory as needed
        return f_names_processed

    def download(self):
        pass

    def process(self):
        labels = np.loadtxt('subdataset1/subdataset1_labels.txt') # Change directory as needed
        #-------------------swapping labels----------------------------#
        indices_one = labels == 1
        indices_zero = labels == 0
        labels[indices_one] = 0 # replacing 1s with 0s
        labels[indices_zero] = 1 # replacing 0s with 1s
        
        for ii in range(0,25000):        
          G = nx.read_gpickle('subdataset1/Graphs/sparse/raw/og/G_og'+str(ii)+'.gpickle') # Change directory as needed        
          graph = from_networkx(G)
          graph['y'] = labels[ii]
          graph['x'] = graph['feature']
          torch.save(graph, osp.join(self.processed_dir, 'og/graph_og_{}.pt'.format(ii))) # Change directory as needed
          
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'rxy/graph_rxy_{}.pt'.format(idx))) # Change directory as needed
        return data

#:::::::::::::::::::::::::::SEGMENT medium::::::::::::::::::::::::::::::::::::::#

#--------------------------------Original Dataset----------------------------#
class simple_graph_norm_medium(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(simple_graph_norm_medium, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
     f_names_raw = []
     for ii in range(0,25000):
       f_names_raw.append('subdataset1/Graphs/medium/raw/og/G_og'+str(ii)+'.gpickle') # Change directory as needed
     return f_names_raw

    @property
    def processed_file_names(self):
        f_names_processed = []
        for ii in range(0,25000):
          f_names_processed.append('og/graph_og_'+str(ii)+'.pt') # Change directory as needed
        return f_names_processed

    def download(self):
        pass

    def process(self):
        labels = np.loadtxt('subdataset1/subdataset1_labels.txt') # Change directory as needed
        for ii in range(0,25000):        
          G = nx.read_gpickle('subdataset1/Graphs/medium/raw/og/G_og'+str(ii)+'.gpickle') # Change directory as needed       
          graph = from_networkx(G)
          graph['y'] = labels[ii]
          graph['x'] = graph['feature']
          torch.save(graph, osp.join(self.processed_dir, 'og/graph_og_{}.pt'.format(ii))) # Change directory as needed
        
          
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'og/graph_og_{}.pt'.format(idx))) # Change directory as needed
        return data

#--------------------------------Reflect over y axis Dataset----------------#
class simple_graph_norm_medium_ry(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(simple_graph_norm_medium_ry, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
     f_names_raw = []
     for ii in range(0,25000):
       f_names_raw.append('subdataset1/Graphs/medium/raw/og/G_ry'+str(ii)+'.gpickle') # Change directory as needed
     return f_names_raw

    @property
    def processed_file_names(self):
        f_names_processed = []
        for ii in range(0,25000):
          f_names_processed.append('ry/graph_ry_'+str(ii)+'.pt') # Change directory as needed
        return f_names_processed

    def download(self):
        pass

    def process(self):
        labels = np.loadtxt('subdataset1/subdataset1_labels.txt') # Change directory as needed
        #-------------------swapping labels----------------------------#
        indices_one = labels == 1
        indices_zero = labels == 0
        labels[indices_one] = 0 # replacing 1s with 0s
        labels[indices_zero] = 1 # replacing 0s with 1s
        
        for ii in range(0,25000):        
          G = nx.read_gpickle('subdataset1/Graphs/medium/raw/og/G_og'+str(ii)+'.gpickle') # Change directory as needed       
          graph = from_networkx(G)
          graph['y'] = labels[ii]
          graph['x'] = graph['feature']
          torch.save(graph, osp.join(self.processed_dir, 'og/graph_og_{}.pt'.format(ii))) # Change directory as needed
          
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'ry/graph_ry_{}.pt'.format(idx))) # Change directory as needed
        return data
#-------------------------------Reflect over x axis Dataset----------------------------#
class simple_graph_norm_medium_rx(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(simple_graph_norm_medium, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
     f_names_raw = []
     for ii in range(0,25000):
       f_names_raw.append('subdataset1/Graphs/medium/raw/rx/G_rx'+str(ii)+'.gpickle') # Change directory as needed
     return f_names_raw

    @property
    def processed_file_names(self):
        f_names_processed = []
        for ii in range(0,25000):
          f_names_processed.append('rx/graph_rx_'+str(ii)+'.pt') # Change directory as needed
        return f_names_processed

    def download(self):
        pass

    def process(self):
        labels = np.loadtxt('subdataset1/subdataset1_labels.txt') # Change directory as needed
        for ii in range(0,25000):        
          G = nx.read_gpickle('subdataset1/Graphs/medium/raw/rx/G_rx'+str(ii)+'.gpickle') # Change directory as needed        
          graph = from_networkx(G)
          graph['y'] = labels[ii]
          graph['x'] = graph['feature']
          torch.save(graph, osp.join(self.processed_dir, 'rx/graph_rx_{}.pt'.format(ii))) # Change directory as needed
        
          
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'rx/graph_rx_{}.pt'.format(idx))) # Change directory as needed
        return data

#--------------------------------Reflect over y axis Dataset----------------#
class simple_graph_norm_medium_rxy(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(simple_graph_norm_medium_rxy, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
     f_names_raw = []
     for ii in range(0,25000):
       f_names_raw.append('subdataset1/Graphs/medium/raw/og/G_rxy'+str(ii)+'.gpickle') # Change directory as needed
     return f_names_raw

    @property
    def processed_file_names(self):
        f_names_processed = []
        for ii in range(0,25000):
          f_names_processed.append('rxy/graph_rxy_'+str(ii)+'.pt') # Change directory as needed
        return f_names_processed

    def download(self):
        pass

    def process(self):
        labels = np.loadtxt('subdataset1/subdataset1_labels.txt') # Change directory as needed
        #-------------------swapping labels----------------------------#
        indices_one = labels == 1
        indices_zero = labels == 0
        labels[indices_one] = 0 # replacing 1s with 0s
        labels[indices_zero] = 1 # replacing 0s with 1s
        
        for ii in range(0,25000):        
          G = nx.read_gpickle('subdataset1/Graphs/medium/raw/og/G_og'+str(ii)+'.gpickle') # Change directory as needed        
          graph = from_networkx(G)
          graph['y'] = labels[ii]
          graph['x'] = graph['feature']
          torch.save(graph, osp.join(self.processed_dir, 'og/graph_og_{}.pt'.format(ii))) # Change directory as needed
          
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'rxy/graph_rxy_{}.pt'.format(idx))) # Change directory as needed
        return data


#:::::::::::::::::::::::::::SEGMENT dense::::::::::::::::::::::::::::::::::::::#

#--------------------------------Original Dataset----------------------------#
class simple_graph_norm_dense(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(simple_graph_norm_dense, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
     f_names_raw = []
     for ii in range(0,25000):
       f_names_raw.append('subdataset1/Graphs/dense/raw/og/G_og'+str(ii)+'.gpickle') # Change directory as needed
     return f_names_raw

    @property
    def processed_file_names(self):
        f_names_processed = []
        for ii in range(0,25000):
          f_names_processed.append('og/graph_og_'+str(ii)+'.pt') # Change directory as needed
        return f_names_processed

    def download(self):
        pass

    def process(self):
        labels = np.loadtxt('subdataset1/subdataset1_labels.txt') # Change directory as needed
        for ii in range(0,25000):        
          G = nx.read_gpickle('subdataset1/Graphs/dense/raw/og/G_og'+str(ii)+'.gpickle') # Change directory as needed       
          graph = from_networkx(G)
          graph['y'] = labels[ii]
          graph['x'] = graph['feature']
          torch.save(graph, osp.join(self.processed_dir, 'og/graph_og_{}.pt'.format(ii))) # Change directory as needed
        
          
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'og/graph_og_{}.pt'.format(idx))) # Change directory as needed
        return data

#--------------------------------Reflect over y axis Dataset----------------#
class simple_graph_norm_dense_ry(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(simple_graph_norm_dense_ry, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
     f_names_raw = []
     for ii in range(0,25000):
       f_names_raw.append('subdataset1/Graphs/dense/raw/og/G_ry'+str(ii)+'.gpickle') # Change directory as needed
     return f_names_raw

    @property
    def processed_file_names(self):
        f_names_processed = []
        for ii in range(0,25000):
          f_names_processed.append('ry/graph_ry_'+str(ii)+'.pt') # Change directory as needed
        return f_names_processed

    def download(self):
        pass

    def process(self):
        labels = np.loadtxt('subdataset1/subdataset1_labels.txt') # Change directory as needed
        #-------------------swapping labels----------------------------#
        indices_one = labels == 1
        indices_zero = labels == 0
        labels[indices_one] = 0 # replacing 1s with 0s
        labels[indices_zero] = 1 # replacing 0s with 1s
        
        for ii in range(0,25000):        
          G = nx.read_gpickle('subdataset1/Graphs/dense/raw/og/G_og'+str(ii)+'.gpickle') # Change directory as needed       
          graph = from_networkx(G)
          graph['y'] = labels[ii]
          graph['x'] = graph['feature']
          torch.save(graph, osp.join(self.processed_dir, 'og/graph_og_{}.pt'.format(ii))) # Change directory as needed
          
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'ry/graph_ry_{}.pt'.format(idx))) # Change directory as needed
        return data
#-------------------------------Reflect over x axis Dataset----------------------------#
class simple_graph_norm_dense_rx(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(simple_graph_norm_dense, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
     f_names_raw = []
     for ii in range(0,25000):
       f_names_raw.append('subdataset1/Graphs/dense/raw/rx/G_rx'+str(ii)+'.gpickle') # Change directory as needed
     return f_names_raw

    @property
    def processed_file_names(self):
        f_names_processed = []
        for ii in range(0,25000):
          f_names_processed.append('rx/graph_rx_'+str(ii)+'.pt') # Change directory as needed
        return f_names_processed

    def download(self):
        pass

    def process(self):
        labels = np.loadtxt('subdataset1/subdataset1_labels.txt') # Change directory as needed
        for ii in range(0,25000):        
          G = nx.read_gpickle('subdataset1/Graphs/dense/raw/rx/G_rx'+str(ii)+'.gpickle') # Change directory as needed        
          graph = from_networkx(G)
          graph['y'] = labels[ii]
          graph['x'] = graph['feature']
          torch.save(graph, osp.join(self.processed_dir, 'rx/graph_rx_{}.pt'.format(ii))) # Change directory as needed
        
          
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'rx/graph_rx_{}.pt'.format(idx))) # Change directory as needed
        return data

#--------------------------------Reflect over y axis Dataset----------------#
class simple_graph_norm_dense_rxy(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(simple_graph_norm_dense_rxy, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
     f_names_raw = []
     for ii in range(0,25000):
       f_names_raw.append('subdataset1/Graphs/dense/raw/og/G_rxy'+str(ii)+'.gpickle') # Change directory as needed
     return f_names_raw

    @property
    def processed_file_names(self):
        f_names_processed = []
        for ii in range(0,25000):
          f_names_processed.append('rxy/graph_rxy_'+str(ii)+'.pt') # Change directory as needed
        return f_names_processed

    def download(self):
        pass

    def process(self):
        labels = np.loadtxt('subdataset1/subdataset1_labels.txt') # Change directory as needed
        #-------------------swapping labels----------------------------#
        indices_one = labels == 1
        indices_zero = labels == 0
        labels[indices_one] = 0 # replacing 1s with 0s
        labels[indices_zero] = 1 # replacing 0s with 1s
        
        for ii in range(0,25000):        
          G = nx.read_gpickle('subdataset1/Graphs/dense/raw/og/G_og'+str(ii)+'.gpickle') # Change directory as needed        
          graph = from_networkx(G)
          graph['y'] = labels[ii]
          graph['x'] = graph['feature']
          torch.save(graph, osp.join(self.processed_dir, 'og/graph_og_{}.pt'.format(ii))) # Change directory as needed
          
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'rxy/graph_rxy_{}.pt'.format(idx))) # Change directory as needed
        return data

