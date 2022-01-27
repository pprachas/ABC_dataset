import numpy as np

import pygmsh
import meshio

import sys
#---------------------Beam Parameters----------------------------#
L = 40 # length of beam
w = 5 # wdith of beam
r_max = w/10
r_min = w/15
#----------------------------------Import Files---------------------#
# Change to directory of dowloaded txt files in folder subdataset1_geo
f_x = 'subdataset1_geometry/x.npy'
f_l = 'subdataset1_geometry/l.npy'

x = np.load(f_x)
l = np.load(f_l)

for ii in range(0,len(x)):
#-----------------------------------pygmsh structure generation-----------------------------#
  geom = pygmsh.opencascade.Geometry(characteristic_length_min=r_min,characteristic_length_max=r_max)
  block = []
  for jj in range(0,len(x[ii])):
    block.append(geom.add_rectangle([x[ii][jj]-l[ii][jj],L-(L*(jj+1)/40),0],2*l[ii][jj],L/40))
  
  unit = geom.boolean_union(block)


  #----------------------------Add Boundaries----------------------------#
  bot = geom.add_rectangle([0.0,0.0,0.0],w,L/40)
  top = geom.add_rectangle([0.0,L-L/40,0.0],w,L/40)
      
  unit = geom.boolean_union([unit,top])
  unit = geom.boolean_union([unit,bot])
          
      #---------------------------Generate Mesh----------------------------------#
  mesh = pygmsh.generate_mesh(geom, prune_z_0 = True)
      
      
  fname_mesh = 'mesh/mesh'+str(len(x)*(num)+ii) + '.xml' #directory to save mesh
  
  print(fname_mesh)
  
  for cell in mesh.cells:
      if cell.type == "triangle":
          triangle_cells = cell.data
  
  meshio.write(fname_mesh,meshio.Mesh(points=mesh.points,cells={"triangle": triangle_cells}))
