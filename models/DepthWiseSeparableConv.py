import torch.nn as nn

class DepthWiseSeparableConv(nn.Module):
	def __init__(self,in_channels,out_channels,stride,D=1):
		super(DepthWiseSeparableConv,self).__init__()
		self.depthwise = nn.Conv2d(in_channels,D*in_channels,kernel_size=3,padding=1,bias=True,stride=stride,groups = in_channels)
		self.pointwise = nn.Conv2d(D*in_channels,out_channels,stride=stride,bias=True,kernel_size=1)

	def forward(self,x):
		out = self.depthwise(x)
		out = self.pointwise(out)
		return out

		