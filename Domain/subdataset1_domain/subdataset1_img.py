import numpy as np
import matplotlib.pyplot as plt

#---------------------Import coordinate file-------------------------#
f_x = 'simple_bulk/img/subdataset1_geo/x.txt'
f_l = 'simple_bulk/img/subdataset1_geo/l.txt'

x = np.loadtxt(f_x, dtype = int) 
l = np.loadtxt(f_l, dtype = int)

#-------------------Column Parameters---------------------------------#
L = 40 # length of column
w = 5 # width of column
#-------------------Generate Image-------------------------------------#
def img_gen(L,w,x,l,ii):
  # L and w are coumn dimensions
  # x and l are files to generate geometry
  # ii  is the column number to be generated
  x = x
  l = l
  img = np.zeros((L,w), dtype = bool)
  
  img[0,0:w] = 1
  img[-1,0:w] = 1
  
  for jj in range(1,L-1):
    img[jj,x[ii][jj]-l[ii][jj]:x[ii][jj]+l[ii][jj]] = 1
  
  return img

for ii in range(0,x.shape):
  img = img_gen(L,w,x,l,ii) #img ouput, save  as array of images if want to convert to graph