import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon, circle

#-------------------------column Parameters-------------------------#
L = 800 # column length
w = 100 # column width

#------------------------Import Files-------------------------------#
f_x = 'bulk/img/subdataset3_geo/x.txt'
f_y = 'bulk/img/subdataset3_geo/y.txt'
f_inner = 'bulk/img/subdataset3_geo/inner.txt'
f_outer = 'bulk/img/subdataset3_geo/outer.txt'

x = np.loadtxt(f_x)
y = np.loadtxt(f_y)
inner = np.loadtxt(f_inner)
outer = np.loadtxt(f_outer)
#-------------------------Generate Geometry---------------------------#
def img_gen(L,w,x,y,outer,inner,ii):
    
    x = x
    y = y
    r = outer
    outer = inner
    
    r_inner = r*inner # actual inner radii; the values in the txt file are scales W.R.T inner radii
    
    img = np.zeros((L,w), dtype = bool)
    print(x[ii].shape)
    for jj in range(0,x.shape[1]):
      rr,cc = circle(y[ii][jj],x[ii][jj],r[ii][jj], shape = img.shape)
      img[rr,cc] = 1
      rr,cc = circle(y[ii][jj],x[ii][jj],r_inner[ii][jj], shape = img.shape)
      img[rr,cc] = 0
    
    
    polytop = np.array([[0,0],[3*w/2,0],[3*w/2,L/20],[0,L/20]])

    rr, cc = polygon(polytop[:, 1], polytop[:, 0], img.shape)
    img[rr, cc] = 1

    polybot = np.array([[0,L],[3*w/2,L],[3*w/2,L-L/20],[0,L-L/20]])

    rr, cc = polygon(polybot[:, 1], polybot[:, 0], img.shape)
    img[rr, cc] = 1
    
    return img

for ii in range(0,3):
  img = img_gen(L,w,x,y,outer,inner,ii) #img ouput, save  as array of images if want to convert to graph
