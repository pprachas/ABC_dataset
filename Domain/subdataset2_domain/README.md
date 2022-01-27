# Subdataset 2 Codes

This directory contains code for meshing, FEA, example domain generation and ML

For the domain files (subdataset2_geometry.zip): 

folder x and folder y gives the x an y coordinates for each ring with inner radius of 0.15w and outer radius of 0.25w. Note that the coordinates for all domains are stored as image arrays where the origin is in the top left. The coordinates have to flipped accordingly to generate the domain for meshing. 

## Meshing (subdataset2_mesh.py)
This code generates mesh to use in FEA. 
If using the directory to run code, we recommend to download the geoemtry dataset (subdataset2_geometry) and place the file here in this directory.

The code is run with the following versions:

* Python version: 3.7.7
* Gmsh version: 4.6.0

## Image generation (subdataset2_img.py)

This is the code to generate the domain geometry as an image from the files in subdataset2_geometry
The code is run with the following versions:

* Python version: Any should work
* Skimage: 0.18.1
