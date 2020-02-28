import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import torch.nn.init as init
import numpy as np
from ShakeShake import ShakeShake, Shortcut

#Some usefull functions
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
        return m
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
        return m


#Wide Block
class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate,stride=1):
        super(wide_basic, self).__init__()
        self.rank1 = []
        self.rank2 = []
        self.init = False
        self.bn1 = conv_init(nn.BatchNorm2d(in_planes))
        self.conv1 = conv_init(nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = conv_init(nn.BatchNorm2d(planes))
        self.conv2 = conv_init(nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True))

    def init_rank(self,conv,out):
        if conv == "conv1":
            self.rank1 = [0]*out
        elif conv == "conv2":
            self.rank2 = [0]*out

    def rank(self,x,conv):
        if conv == "conv1":
          for f in range(x[0].shape[0]):
              self.rank1[f] += torch.matrix_rank(x[0][f])
        elif conv == "conv2":
          for f in range(x[0].shape[0]):
              self.rank2[f] += torch.matrix_rank(x[0][f])
        
    def forward(self, x):
        if self.init == False:
            self.init_rank("conv1",x[0].shape[0])
            self.init_rank("conv2",x[0].shape[0])
            self.init = True
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        self.rank(x,"conv1")
        out = self.conv2(F.relu(self.bn2(out)))
        self.rank(x,"conv2")

        return out

#Shake Block
class shake_block(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(shake_block,self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.branch1 = wide_basic(in_planes,planes,dropout_rate,stride)
        self.branch2 = wide_basic(in_planes,planes,dropout_rate,stride)
        self.shortcut = (self.planes == self.in_planes) and None or Shortcut(self.in_planes, self.planes, stride=stride)

    def forward(self,x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = ShakeShake.apply(x1,x2)
        x0 = x if self.equal_io else self.shortcut(x)
        return out + x0

#Shake WideResNet
class Shake_WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Shake_WideResNet, self).__init__()
        self.in_planes = 16
        self.depth = depth

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(n,nStages[0],nStages[1],dropout_rate, stride=1)
        self.layer2 = self._wide_layer(n,nStages[1],nStages[2],dropout_rate, stride=2)
        self.layer3 = self._wide_layer(n,nStages[2],nStages[3],dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self,num_blocks,in_planes,planes,dropout_rate, stride=1):
        layers = []
        for i in range(int(num_blocks)):
            layers.append(shake_block(in_planes,planes,dropout_rate,stride=stride))
            in_planes, stride = planes, 1
        return nn.Sequential(*layers)

        # Initialize paramters

    def forward(self,x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out