import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import sys
import time
import datetime
import Wide_ResNet
from data.AutoAugment import AutoAugment,Cutout
from data.DataAugmentation import DataAugmentation