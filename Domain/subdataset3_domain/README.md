# Subdataset 3 Codes

This directory contains code for meshing, FEA, example domain generation and ML

For the domain files (subdataset3_geometry.zip):

x.txt and y.txt contains the x and y coordinate of the ring respectively, while outer.txt and inner.txt gives the ring outer thickness and inner thickness (as a ratio of outer thickness) respectively. Note that the coordinates for all domains are stored as image arrays where the origin is in the top left. The coordinates have to flipped accordingly to generate the domain for meshing.

## Meshing (subdataset3_mesh.py)
This code generates mesh to use in FEA. 
If using the directory to run code, we recommend to download the geoemtry dataset (subdataset1_geo) and place the file here in this directory.

The code is run with the following versions:

* Python version: 3.7.7
* Gmsh version: 4.6.0

## Image generation (subdataset3_img.py)

This is the code to generate the domain geometry as an image from the files in subdataset1_geo
The code is run with the following versions:

* Python version: Any should work
* Skimage: 0.18.1
