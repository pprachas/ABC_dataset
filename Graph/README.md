# Graph codes

This directory contains code for to obtain graphs used in the manuscript. **Note that only networkx graphs are generated here. Code to preprocess and prepare the networkx graphs for the ML model are in the fold ML.**

## Graph generation from image (graph_gen.py)

This code is for generating graphs with the domain generated images with subdatax_geo.zip

The code is run with the following versions:
	* networkx: 2.5.1

Segmentation to replicate sparse, medium, and dense node densities as in paper:
 Subdataset 1:
 * sparse: 350 segments
 * medium: 600 segments
 * dense: 1100 segments

 Subdataset 2:
 * sparse: 220 segments
 * medium: 400 segments
 * dense: 790 segments

 Subdataset 3:
 * sparse: 700 segments
 * medium: 1000 segments
 * dense: 2300 segments

Note that the graph features are not normalized. Normalizing the features help with training.

## Normalizing Graph features (normalize_feature.py)
This code is used to normalize feature vectors to help training.
 
## Exact Representations (subdataset1_exact.py and subdataset2_exact.py)
This code generates perfect representations for subdataset1 and subdataset2 in our manuscript. 


 

