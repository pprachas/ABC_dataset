# ABC dataset
This is the repository for codes for ML, domain generation, graph generation of Asymmetric Buckling Columns (ABC) dataset in the paper "Learning Mechanically Driven Emergent Behavior with Message Passing Neural Networks". 

Link to paper is coming soon!

The ABC data consists of spatially heterogenous beams under compression with fixed-fixed boundary conditions. The objective is to classify each column into its buckling direction. The dataset is seperated into 3 sub-datasets, each with varying geometry. 

Link to dataset is coming soon!

If using the geometry files, then start from folders Domain &#8594 FEA (if you need labels) &#8594 Graph &#8594 ML.
If using the json files, then start from folders FEA (if you need labels )&#8594 Graph &#8594 ML.

The FEA folder can be skiiped if labels are already obtained. 

![dataset_fig](https://user-images.githubusercontent.com/89213088/150606555-056172a1-1d02-45f6-9191-ae99596bb81c.png)

# Domain
The Domain folder contains code to generate the domain geometry for ABC dataset. More deatils of the code are contained in the folder.

# FEA
The Domain folder contains code to labels for ABC dataset. More deatils of the code are contained in the folder.

# Graph
The Graph folder contains code to generate graphsfrom domain geometry and json files. 

# ML
The ML folder contains code for our  Point++ implementation, ML architecture, and voting.
