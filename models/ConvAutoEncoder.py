import torch.nn as nn
import torch.nn.functional as F

class ConvAutoEncoder():
		def __init__(self,in_planes,planes):
			self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels = planes,kernel_size=(3,3),stride=1,padding_mode="same") #32x32X3 -> 32X32X32
		    self.bn1 = nn.BatchNorm2d(planes)
		    self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=(3,3),stride=2,padding_mode="same") #32x32X32 -> 16x16x32
		    self.conv3 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=(3,3),stride=1,padding_mode="same") #32x32X32 -> 16x16x32
		    self.up = nn.Upsample(scale_factor=2, mode='nearest')
		    self.conv4 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=(3,3),stride=1, padding_mode="same")
		    self.bn2 = nn.BatchNorm2d(planes)
		    self.conv5 = nn.Conv2d(in_channels=planes, out_channels=in_planes, kernel_size=(1,1), stride =1, padding_mode="same")

		def forward(self,x):
			out = self.conv1(F.relu(x))
			out = self.conv2((F.relu(self.bn1(out))))
			out = self.conv3(F.relu(out))
			out = self.up(out)
			out = self.conv4(F.relu(out))
			out = self.conv5(F.relu(self.bn2(out)))
			return out


