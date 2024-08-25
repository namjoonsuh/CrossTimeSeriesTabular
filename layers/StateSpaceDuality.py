import torch
import torch.nn as nn
from mamba_ssm import Mamba2


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        output = x / rms * self.weight
        return output
    
class StateSpaceDualityBlock(nn.Module):
    def __init__(self, d_model, headdim=32, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba2(d_model=d_model, headdim=headdim, d_state=64, d_conv=4, expand=2)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        identity = x
        x = self.mamba(x)
        x = self.dropout(x)
        x += identity
        x = self.norm(x)
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
    
class StateSpaceDualityBackbone(nn.Module):
    def __init__(self, d_model, headdim=32, dropout=0.0):
        super().__init__()
        self.ssd = StateSpaceDualityBlock(d_model=d_model, headdim=headdim, dropout=dropout)
        self.ffn = FFNBlock(d_model=d_model, dropout=dropout)
    def forward(self, x):
        x = self.ssd(x)
        x = self.ffn(x)
        return x