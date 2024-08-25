import torch
import torch.nn as nn
import torch.nn.functional as F


class NumEmbedding(nn.Module):
    def __init__(self, d_model, patch_size, stride_size):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=patch_size, stride=stride_size)
    def forward(self, x):
        B, T, M = x.shape
        x = x.permute(0, 2, 1) 
        x = x.unsqueeze(2) 
        x = x.reshape(B*M, 1, T)
        x = self.conv(x)
        x = x.permute(0, 2, 1)  
        _, N, D = x.shape
        x = x.reshape(B, M, N, D)
        return x

class CatEmbedding(nn.Module):
    def __init__(self, d_model, patch_size, stride_size):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=1024, out_channels=d_model, kernel_size=patch_size, stride=stride_size)
    def forward(self, x):
        B, T, M, D = x.shape
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B*M, T, D)
        x = x.permute(0, 2, 1)
        x = self.conv(x)  
        x = x.permute(0, 2, 1)
        _, N, D = x.shape
        x = x.reshape(B, M, N, D)
        return x
    
class TextEmbedding(nn.Module):
    def __init__(self, d_model, n_vars):
        super().__init__()
        self.n_vars = n_vars
        self.conv = nn.Conv1d(in_channels=1024, out_channels=d_model, kernel_size=1, stride=1)
    def forward(self, x):
        B, L, D = x.shape
        x = x.permute(0, 2, 1)
        x = self.conv(x)  
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(1)
        x = x.repeat(1, self.n_vars, 1, 1)
        B, M, L, D = x.shape
        x = x.reshape(B * M, L, D)
        return x