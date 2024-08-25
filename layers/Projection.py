import torch
import torch.nn as nn
import torch.nn.functional as F


class NumProjection(nn.Module):
    def __init__(self, d_model, patch_size, stride_size):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels=d_model, out_channels=1, kernel_size=patch_size, stride=stride_size)
    def forward(self, x):
        B, M, N, D = x.shape
        x = x.reshape(B*M, N, D)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = x.squeeze(-1) 
        _, T = x.shape
        x = x.reshape(B, M, T)
        x = x.permute(0, 2, 1)
        return x
    
class CatProjection(nn.Module):
    def __init__(self, d_model, patch_size, stride_size):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels=d_model, out_channels=1024, kernel_size=patch_size, stride=stride_size)
    def forward(self, x):
        B, M, N, D = x.shape
        x = x.reshape(B*M, N, D)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        _, T, D = x.shape
        x = x.reshape(B, M, T, D)
        x = x.permute(0, 2, 1, 3)
        return x