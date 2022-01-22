# Subdataset 1 Codes

This directory contains code for meshing, FEA, example domain generation and ML

For the domain files:
x.txt indicates the centers of each block and l.txt gives the length of each block. Note that the coordinates for both sub-dataset 2 and sub-dataset 3 are stored as image arrays where the origin is in the top left. The coordinates have to flipped accordingly to generate the domain for meshing.

## Meshing (subdataset1_mesh.py)
This code generates mesh to use in FEA. 
If using the directory to run code, we recommend to download the geoemtry dataset (subdataset1_geo) and place the file here in this directory.

The code is run with the following versions:

* Python version: 3.7.7
* Gmsh version: 4.6.0

## Image generation (subdataset1_img.py)

This is the code to generate the domain geometry as an image from the files in subdataset1_geo
The code is run with the following versions:

* Python version: Any should work
* Skimage: 0.18.1
