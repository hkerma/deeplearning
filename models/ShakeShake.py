import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random 

class ShakeShake(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x1,x2,training=True):
        if training:
            alpha = random.randint(0,1)
        else:
            alpha = 0.5
        return alpha*x1 + (1 - alpha)*x2

    @staticmethod
    def backward(ctx,grad_output):
        beta = random.randint(0,1)
        return beta*grad_output,(1-beta)*grad_output

class Shortcut(nn.Module):

    def __init__(self, in_planes, planes, stride):
        super(Shortcut, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_planes, planes // 2, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(in_planes, planes // 2, 1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x):
        h = F.relu(x)

        h1 = F.avg_pool2d(h, 1, self.stride)
        h1 = self.conv1(h1)

        h2 = F.avg_pool2d(F.pad(h, (-1, 1, -1, 1)), 1, self.stride)
        h2 = self.conv2(h2)

        h = torch.cat((h1, h2), 1)
        return self.bn(h)