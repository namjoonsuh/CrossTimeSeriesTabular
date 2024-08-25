import numpy as np
import pandas as pd
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from layers.MixerBlock import MixerBlcok
from layers.RevIN import RevIN
from layers.Embedding import NumEmbedding, CatEmbedding, TextEmbedding
from layers.Sampling import DownSampling, UpSampling
from layers.Projection import NumProjection, CatProjection

import warnings
warnings.filterwarnings("ignore")

np.random.seed(1234)
torch.manual_seed(1234)


class CVAE(nn.Module):
	def __init__(self, d_model, n_vars, seq_len, patch_size, stride_size, d_nv, d_cv):
		super().__init__()	
		self.d_nv = d_nv
		self.d_cv = d_cv
		self.revin = RevIN(n_vars = d_nv)
		self.embd1 = NumEmbedding(d_model=d_model, patch_size=patch_size, stride_size=stride_size)
		self.embd2 = CatEmbedding(d_model=d_model, patch_size=patch_size, stride_size=stride_size)
		self.embd3 = TextEmbedding(d_model=d_model, n_vars=n_vars)
		self.enc1 = MixerBlcok(d_model=d_model, n_vars=n_vars)
		self.ds1 = DownSampling(in_dim=d_model, out_dim=d_model//2)
		self.enc2 = MixerBlcok(d_model=d_model//2, n_vars=n_vars)
		self.ds2 = DownSampling(in_dim=d_model//2, out_dim=d_model//4)
		self.enc3 = MixerBlcok(d_model=d_model//4, n_vars=n_vars)
		self.linear = nn.Linear(d_model//4, d_model//2)
		self.dec1 = MixerBlcok(d_model=d_model//4, n_vars=n_vars)
		self.us1 = UpSampling(in_dim=d_model//4, out_dim=d_model//2)
		self.dec2 = MixerBlcok(d_model=d_model//2, n_vars=n_vars)
		self.us2 = UpSampling(in_dim=d_model//2, out_dim=d_model)
		self.dec3 = MixerBlcok(d_model=d_model, n_vars=n_vars)
		self.proj1 = NumProjection(d_model=d_model, patch_size=patch_size, stride_size=stride_size)
		self.proj2 = CatProjection(d_model=d_model, patch_size=patch_size, stride_size=stride_size)
	def encoding(self, x1, x2, c): 
		x1 = self.revin(x1, mode='norm') 
		x1 = self.embd1(x1) 
		x2 = self.embd2(x2) 
		x = torch.cat((x1, x2), dim=1)
		c = self.embd3(c)
		x = self.enc1(x, c)
		x, c = self.ds1(x, c)
		x = self.enc2(x, c)
		x, c = self.ds2(x, c)
		x = self.enc3(x, c)
		x = self.linear(x)
		return x, c
	def decoding(self, z, c):
		x = self.dec1(z, c)
		x, c = self.us1(x, c)
		x = self.dec2(x, c)
		x, c = self.us2(x, c)
		x = self.dec3(x, c) 
		x1, x2 = torch.split(x, [self.d_nv, self.d_cv], dim=1)
		x1 = self.proj1(x1)
		x1 = self.revin(x1, mode='denorm')
		x2 = self.proj2(x2)
		return x1, x2
	def reparameterize(self, mu, log_var):
		std = torch.exp(0.5*log_var)
		eps = torch.randn_like(std)
		return mu + eps*std
	def forward(self, x1, x2, c):
		x, c= self.encoding(x1, x2, c)
		mu, log_var = torch.chunk(x, 2, dim=3)
		z = self.reparameterize(mu, log_var)
		x1, x2 = self.decoding(z, c)
		return x1, x2, mu, log_var
	

B = 32 # batch size
T = 336 # lookback window
L = 7 # column name length
M1 = 6 # max number of num var
M2 = 4 # max number of cat var
D = 512 # embedding dimention
P = 16 # patch size
S = P // 2 # stride size

model = CVAE(d_model=D, n_vars=M1+M2, seq_len=T, patch_size=P, stride_size=S, d_nv=M1, d_cv=M2)

x1 = torch.randn(B, T, M1)
x2 = torch.randn(B, T, M2, 1024)
c = torch.randn(B, L, 1024)
rx1, rx2, mu, log_var = model(x1, x2, c)
print(rx1.shape)
print(rx2.shape)