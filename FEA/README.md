# FEA

This FEA code (with a few modifications as commented in the code) can be used for all subdatasets if the corresponding mesh is generated. Most modifications are: 

* Different length and width of beam for subdatasets:
	* l = 40.0 and w = 5.0 for subdatset 1
	* l = 800 and w = 10.0 for subdataset 2 and subdataset 3
* Changing the directory for corresponding dataset:

The code is run with the following versions:

* Python version: 3.7.7
* FEniCS version: 2019.1.0-PETSc

Note: To properly run and save the buckling direction, a .txt file with 20 rows should be initialized. lr1.txt would correspoind to the first 20 mesh, lr2.txt would be the next 20 mesh, etc (up to 1250). Modufy this part if desired.

The file take in inputs with the first input being the batch number (1-1250), the second being the row number in the .txt file (1-20) and the last being the number of columns in the .txt file (20).

For example: 

> python3 subdataset_fea.py 5 3 20

is running a simulation on mesh83.xml. 

This method makes it easy for error detection.
