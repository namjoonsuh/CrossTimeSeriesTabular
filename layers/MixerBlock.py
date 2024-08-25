import torch
import torch.nn as nn
import torch.nn.functional as F
# from layers.StateSpaceDuality import StateSpaceDualityBlock
from layers.MultiHeadCrossAttention import MultiHeadCrossAttentionBlock


class SqueezeExcitationBlock(nn.Module):
    def __init__(self, n_vars):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.conv1 = nn.Conv1d(in_channels=n_vars, out_channels=n_vars, kernel_size=1, stride=1, padding='same', dilation=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=n_vars, out_channels=n_vars, kernel_size=1, stride=1, padding='same', dilation=1, bias=False)
        self.act = nn.Sigmoid()
    def forward(self, x):
        identity = x
        x1 = self.avgpool(x).squeeze(-1)
        x2 = self.maxpool(x).squeeze(-1)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        y = self.act(x1.transpose(1, 2) + x2.transpose(1, 2)).transpose(1, 2).unsqueeze(-1)
        x = x * y.expand_as(x)
        x += identity
        return x

class FFNBlock(nn.Module):
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model) 
        self.gate_proj = nn.Linear(d_model, 4 * d_model)
        self.down_proj = nn.Linear(d_model, 4 * d_model)
        self.up_proj = nn.Linear(4 * d_model, d_model)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        identity = x
        x1, x2 = self.gate_proj(x), self.down_proj(x)
        x1 = self.act_fn(x1)
        x = self.up_proj(x1 * x2)
        x = self.dropout(x)
        x += identity
        x = self.norm(x)
        return x

class MixerBlcok(nn.Module):
	def __init__(self, d_model, n_vars):
		super().__init__()
		# self.temp_attn = StateSpaceDualityBlock(d_model=d_model)
		self.spat_attn = SqueezeExcitationBlock(n_vars=n_vars)
		self.cross_attn = MultiHeadCrossAttentionBlock(d_model=d_model)
		self.ffn = FFNBlock(d_model=d_model)
	def forward(self, x, c):
		B, M, N, D = x.shape
		x = x.reshape(B*M, N, D)
		# x = self.temp_attn(x)
		x = x.reshape(B, M, N, D)
		x = self.spat_attn(x)
		x = x.reshape(B*M, N, D)
		x = self.cross_attn(x, c)
		x = x.reshape(B, M, N, D)
		x = self.ffn(x)
		return x