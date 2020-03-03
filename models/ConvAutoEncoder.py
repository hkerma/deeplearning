import torch.nn as nn
import torch.nn.functional as F

class ConvAutoEncoder(nn.Module):
    def __init__(self, in_planes = 1, planes = 16):
        super(ConvAutoEncoder, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 3 --> 16), (size 32x32), 3x3 kernels 
        self.conv1 = nn.Conv2d(in_planes, planes, 3, padding=1)  
        # conv layer (depth from 16 --> 8), (size 32x32), 3x3 kernels
        self.conv2 = nn.Conv2d(planes, planes//2, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2, (size 16x16)
        self.pool = nn.MaxPool2d(2, 2)
        
        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
				# conv layer (depth from 8 --> 16), (size 32x32), 3x3 kernels 
        self.t_conv1 = nn.ConvTranspose2d(planes//2, planes, 2, stride=2)
				# conv layer (depth from 16 --> 3), (size 32x32), 3x3 kernels 
        self.t_conv3 = nn.ConvTranspose2d(planes,in_planes, 2, stride=2)


    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # compressed representation
        
        ## decode ##
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        # output layer (with sigmoid for scaling from 0 to 1)
        x = F.sigmoid(self.t_conv2(x))
                
        return x