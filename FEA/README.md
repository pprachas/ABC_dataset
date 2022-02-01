# FEA

This FEA code (with a few modifications as commented in the code) can be used for all subdatasets if the corresponding mesh is generated. Most modifications are: 

* Different length and width of beam for subdatasets:
	* l = 40.0 and w = 5.0 for subdatset 1
	* l = 800 and w = 10.0 for subdataset 2 and subdataset 3
* Changing the directory for corresponding dataset:

The code is run with the following versions:

* Python version: 3.7.7
* FEniCS version: 2019.1.0-PETSc

Note: The current available code does not save the labels. The labels can be appended in a list and saved as a .txt file as needed. However, the labels for each structure used in the manuscript are provided in [The Boston University Institutional Repository](https://open.bu.edu/handle/2144/43730). 
