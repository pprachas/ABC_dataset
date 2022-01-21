import torch 

#----------------Note-----------------------------------------------#
#This code is used to initialize the Pointet weights like the manuscript

#------------------Import Models---------------------------------------#
from Pointnet_layer import *
#------------------Weight Initialization-------------------------------#
for ii in range(1,11):
  torch.manual_seed(ii)
  
  model = PointNet4Layers()
  torch.save(model.state_dict(),'init_models/init_model_seed'+str(ii)+'.pt')