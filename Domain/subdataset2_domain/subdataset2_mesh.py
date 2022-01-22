import numpy as np

import pygmsh
import meshio

import sys
import time
#---------------------Beam and Geometric Parameters----------------------------#
ti = time.time()
L = 800 # length of beam
w = 100 # wdith of beam
r = 0.25*w # radius of each circle
p = 5 # Ring thickness

for ii in range(0,25000):
#---------------------------Import files-------------------------------------#
  # Change to directory of dowloaded txt files in folder subdataset2_geo
  f_x = 'subdataset2_geo/x'+str(ii)+'.txt' # import x coordinates
  f_y = 'subdataset2_geo/y'+str(ii)+'.txt' # import y coordinates

  x = np.loadtxt(f_x)
  y = np.loadtxt(f_y)
  
  print(x)
  print(len(x))
#-----------------------------------pygmsh structure generation-----------------------------#
  geom = pygmsh.opencascade.Geometry(characteristic_length_min=(r-p)/5,characteristic_length_max=(r+p)/5)

  circle = geom.add_disk([x[0],L-y[0],0.0],r+p)
  hole = geom.add_disk([x[0],L-y[0],0.0],r-p)
  unit = geom.boolean_difference([circle],[hole])
  #-----------------------------------Add Rings---------------------------#   
  for jj in range(1,len(x)):  
    print(x[jj],y[jj])
    circle = geom.add_disk([x[jj],L-y[jj],0.0],r+p)
    hole = geom.add_disk([x[jj],L-y[jj],0.0],r-p)
    donut = geom.boolean_difference([circle],[hole])
  
    unit = geom.boolean_union([unit,donut])
                      
      #----------------------------Add Boundaries----------------------------#
  bot = geom.add_rectangle([0.0,0.0,0.0],w,L/20)
  top = geom.add_rectangle([0.0,L-L/20,0.0],w,L/20)
      
  unit = geom.boolean_union([unit,top])
  unit = geom.boolean_union([unit,bot])
      
  
      
      #---------------------------Generate Mesh----------------------------------#
  mesh = pygmsh.generate_mesh(geom, prune_z_0 = True)

  #--------------------------------Save Mesh-------------------------------------#  
  fname_mesh = 'mesh/mesh'+str(ii) + '.xml' # Directory to meshes
  for cell in mesh.cells:
      if cell.type == "triangle":
          triangle_cells = cell.data
 
  meshio.write(fname_mesh,meshio.Mesh(points=mesh.points,cells={"triangle": triangle_cells}))
