import numpy as np

import pygmsh
import meshio

import sys
#---------------------Beam Parameters----------------------------#
L = 800 # length of beam
w = 100 # wdith of beam
r_max = w/5
r_min = w/10
#----------------------------------Import Files---------------------#
# Change to directory of dowloaded txt files in folder subdataset3_geo
f_x = 'subdataset3_geo/x.txt'
f_y = 'subdataset3_geo/y.txt'
f_inner = 'subdataset3_geo/inner.txt'
f_outer = 'subdataset3_geo/outer.txt'

x = np.loadtxt(f_x)
y = np.loadtxt(f_y)
inner = np.loadtxt(f_inner)
outer = np.loadtxt(f_outer)
for ii in range(0,len(x)):
#-----------------------------------pygmsh structure generation-----------------------------#
  geom = pygmsh.opencascade.Geometry(characteristic_length_min=r_min/10,characteristic_length_max=r_max/10)
  
      
  circle = geom.add_disk([x[ii][0],L-y[ii][0],0.0],outer[ii][0])
  hole = geom.add_disk([x[ii][0],L-y[ii][0],0.0],outer[ii][0]*inner[ii][0])
  unit = geom.boolean_difference([circle],[hole])
  #-----------------------------------Add donuts and prune small edges---------------------------#   
  for jj in range(1,len(x[ii])):
    circle = geom.add_disk([x[ii][jj],L-y[ii][jj],0.0],outer[ii][jj])
    hole = geom.add_disk([x[ii][jj],L-y[ii][jj],0.0],outer[ii][jj]*inner[ii][jj])
  
    unit = geom.boolean_union([unit,circle])
  
    unit = geom.boolean_difference([unit],[hole])     
      #----------------------------Add Boundaries----------------------------#
  bot = geom.add_rectangle([0.0,0.0,0.0],w,L/20)
  top = geom.add_rectangle([0.0,L-L/20,0.0],w,L/20)
      
  unit = geom.boolean_union([unit,top])
  unit = geom.boolean_union([unit,bot])
      
  
      
      #---------------------------General Mesh----------------------------------#
  mesh = pygmsh.generate_mesh(geom, prune_z_0 = True)
      
      
  fname_mesh = 'mesh/mesh'+str(len(x)*(num)+ii) + '.xml'
  for cell in mesh.cells:
      if cell.type == "triangle":
          triangle_cells = cell.data
  
  meshio.write(fname_mesh,meshio.Mesh(points=mesh.points,cells={"triangle": triangle_cells}))
