import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def SReLu(inputs, alpha=0.1):
	return 0.5*((inputs**2+alpha)**0.5 + inputs)

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()

		self.conv1 = nn.Sequential(
									nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
									nn.BatchNorm2d(32)
									)
									# N x 32 x 32 x 32

		self.conv2 = nn.Sequential(
									nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
									nn.BatchNorm2d(64)
									)
									# N x 64 x 32 x 32

		self.conv3 = nn.Sequential(
									nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
									nn.BatchNorm2d(64),
									nn.MaxPool2d(kernel_size=2, stride=None)
									# nn.AvgPool2d(kernel_size=2, stride=2)
									)
									# N x 128 x 16 x 16

		self.conv4 = nn.Sequential(
									nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
									nn.BatchNorm2d(64),
									nn.MaxPool2d(kernel_size=2, stride=None)
									# nn.AvgPool2d(kernel_size=2, stride=2)
									)
									# N x 64 x 8 x 8

		self.FC_1 = nn.Linear(4096, 1024)
		self.FC_2 = nn.Linear(1024, 256)
		self.FC_3 = nn.Linear(256, 10)

		# self.softmax = nn.Softmax()


	def forward(self, x):
		N = x.shape[0]

		x = self.conv1(x)
		x = SReLu(x)

		x = self.conv2(x)
		x = SReLu(x)

		x = self.conv3(x)
		x = SReLu(x)		

		x = self.conv4(x)
		x = SReLu(x)

		x = x.view(N, -1)

		x = self.FC_1(x)
		x = SReLu(x)

		x = self.FC_2(x)
		x = SReLu(x)

		out = self.FC_3(x)

		return out

class CNN_ReLU(nn.Module):
	def __init__(self):
		super(CNN_ReLU, self).__init__()

		self.conv1 = nn.Sequential(
									nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
									nn.BatchNorm2d(32)
									)
									# N x 32 x 32 x 32

		self.conv2 = nn.Sequential(
									nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
									nn.BatchNorm2d(64)
									)
									# N x 64 x 32 x 32

		self.conv3 = nn.Sequential(
									nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
									nn.BatchNorm2d(64),
									nn.MaxPool2d(kernel_size=2, stride=None)
									# nn.AvgPool2d(kernel_size=2, stride=2)
									)
									# N x 128 x 16 x 16

		self.conv4 = nn.Sequential(
									nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
									nn.BatchNorm2d(64),
									nn.MaxPool2d(kernel_size=2, stride=None)
									# nn.AvgPool2d(kernel_size=2, stride=2)
									)
									# N x 64 x 8 x 8

		self.FC_1 = nn.Linear(4096, 1024)
		self.FC_2 = nn.Linear(1024, 256)
		self.FC_3 = nn.Linear(256, 10)

		# self.softmax = nn.Softmax()


	def forward(self, x):
		N = x.shape[0]

		x = self.conv1(x)
		x = F.relu(x)

		x = self.conv2(x)
		x = F.relu(x)

		x = self.conv3(x)
		x = F.relu(x)		

		x = self.conv4(x)
		x = F.relu(x)

		x = x.view(N, -1)

		x = self.FC_1(x)
		x = F.relu(x)

		x = self.FC_2(x)
		x = F.relu(x)

		out = self.FC_3(x)

		return out


# class CNN(nn.Module):
# 	def __init__(self):
# 		super(CNN, self).__init__()

# 		self.conv1 = nn.Sequential(
# 									nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
# 									nn.BatchNorm2d(32)
# 									)
# 									# N x 32 x 32 x 32
# 		self.conv2 = nn.Sequential(
# 									nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
# 									nn.BatchNorm2d(64)
# 									)
# 									# N x 64 x 32 x 32

# 		self.conv3 = nn.Sequential(
# 									nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
# 									nn.BatchNorm2d(64),
# 									# nn.AvgPool2d(kernel_size=2, stride=2)
# 									nn.MaxPool2d(kernel_size=2, stride=None)
# 									)
# 									# N x 64 x 16 x 16

# 		self.FC_3 = nn.Linear(64*16*16, 10)

# 		# self.softmax = nn.Softmax()


# 	def forward(self, x):
# 		N = x.shape[0]

# 		x = self.conv1(x)
# 		x = SReLu(x)

# 		x = self.conv2(x)
# 		x = SReLu(x)

# 		x = self.conv3(x)
# 		x = SReLu(x)

# 		x = x.view(N, -1)

# 		out = self.FC_3(x)

# 		return out
