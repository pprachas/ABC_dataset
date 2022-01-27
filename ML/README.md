# ML codes

This directory contains code for graph preprocessing before being used in Pytorch, Pytorch Geometric (PyG), and the PyG code for the ML architectire used in the manuscript. 
This code can be applied to all sub-datasets. 

All code is run with the following versions:
* Python 3.8.6
* Pytorch: 1.8.1 + cuda 11.1
* Pytorch Geometric: 1.7.2 
* networkx: 2.5.1
The codes in this section assumes that the graphs are in networkx format. See folders Graphs to convert domain geometry or json files to networkx graphs. 
## Conversion of networkx graph to PyG dataset (pyg_graphs.py)
This code is for changing the networkx graphs to PyG datasets. More information on PyG 1.7.2 datasets are [here](https://pytorch-geometric.readthedocs.io/en/1.7.2/notes/create_dataset.html).

Once the graphs are in PyG datasets, the remaining codes can be used. 

## PointNet++ Implementation (Pointnet_layer.py)
This code is our implementation of the PointNet++ layer as well as the ML architecture. THe original paper of PointNet++ can be found [here](https://arxiv.org/abs/1706.02413).

## Initialization (init_models.py)
This code initializes the weights for the 10 seeds used in our manuscript. Tyhe initialized weights are saved and will be used during training. 

## Train ML model (train_model.py)
This code shuffles and splits the dataset into train, validation and test data.
This code also gives you validation accuracy.  

Input arguments: 
 1. Sparse Medium or dense graphs (1, 2, 3, for sparse medium dense respectively)
 2. Number of training datapoints (1, 2, 3, 4 for 5k, 10k, 15k, 20k datapoints repectively)
 3. Initialization seed (1-10)

For example:

> python3 train_model.py 3 4 1

Will train sparse graphs using 20,000 data points with the 10th initialization. The subdataset will depend on the directory of the graphs. 


Note that in the case of initlalization, the code expects the initialization code and state saved(init_models.py) to be run first.  

## Test predictions (test_prediction.py)
This code gets the test predictions from all 10 initalization seeds as class labels and probabilies and can be used for hard voting and soft voting.

## Ensemble methods (ensemble.py)
This code gives you the hard voting and soft voting results from the 10 initializations. 

## GNNExplainer (gnn_explain.py)
This code interprets our GNN models as an edge mask. More information of the original paper on GNNExplainer and how GNNExplainer works is [here](https://arxiv.org/pdf/1903.03894.pdf). The GNNExplainer code uses the Pytorch Geometric implementation of GNNExplainer. 

 


