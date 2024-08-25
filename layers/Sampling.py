import torch
import torch.nn as nn
import torch.nn.functional as F


class DownSampling(nn.Module):
	def __init__(self, in_dim, out_dim):
		super().__init__()
		self.conv1 = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=1)
		self.conv2 = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=1)
	def forward(self, x, c):
		B, M, N, D = x.shape
		x = x.reshape(B*M, N, D)
		x = x.permute(0, 2, 1)
		x = self.conv1(x)
		x = x.permute(0, 2, 1)
		_, _, D = x.shape
		x = x.reshape(B, M, N, D)
		c = c.permute(0, 2, 1)
		c = self.conv2(c)
		c = c.permute(0, 2, 1)
		return x, c

class UpSampling(nn.Module):
	def __init__(self, in_dim, out_dim):
		super().__init__()
		self.conv1 = nn.ConvTranspose1d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=1)
		self.conv2 = nn.ConvTranspose1d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=1)
	def forward(self, x, c):
		B, M, N, D = x.shape
		x = x.reshape(B*M, N, D)
		x = x.permute(0, 2, 1)
		x = self.conv1(x)
		x = x.permute(0, 2, 1)
		_, _, D = x.shape
		x = x.reshape(B, M, N, D)
		c = c.permute(0, 2, 1)
		c = self.conv2(c)
		c = c.permute(0, 2, 1)
		return x, c