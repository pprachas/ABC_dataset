import numpy as np
import matplotlib.pyplot as plt 

from skimage.draw import polygon, circle_perimeter,circle
from skimage import morphology

#---------------------Column Parameters----------------------------#
L =  800 # length of column
w = 100 # wdith of column
r = int(0.2*100)

#----------------------------File Path-----------------------------#
f_x = 'intersect_donut_bulk/img/subdataset2_geo/x/x'
f_y = 'intersect_donut_bulk/img/subdataset2_geo/y/y'

def img_gen(L,w,f_x,f_y,r,ii):
  L = L
  w = w
  r = r 
  p = 5 # ring thickness
  #-------------------------------Import files-------------------#
  x = np.loadtxt(f_x+str(ii)+'.txt')
  y = np.loadtxt(f_y+str(ii)+'.txt')
  
  #-------------------Generate Geometry----------------------------#
  img = np.zeros((L,w))
  
  for jj in range(0,len(x)):
    rr,cc = circle_perimeter(r=int(y[jj]),c=int(x[jj]),radius=r, shape = img.shape)
    img[rr,cc] = 1
  img = morphology.dilation(img, morphology.disk(radius=p))
    
  polytop = np.array([[0,0],[w,0],[w,L/20],[0,L/20]])

  rr, cc = polygon(polytop[:, 1], polytop[:, 0], img.shape)
  img[rr, cc] = 1

  polybot = np.array([[0,L],[w,L],[w,L-L/20],[0,L-L/20]])

  rr, cc = polygon(polybot[:, 1], polybot[:, 0], img.shape)
  img[rr, cc] = 1
  
  return img
    

for ii in range(0,25000):
  img = img_gen(L,w,f_x,f_y,r,ii) #img ouput, save  as array of images if want to convert to graph
  
