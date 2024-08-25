import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        output = x / rms * self.weight
        return output

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout=0.0, res_attn=True):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attn = res_attn
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=False)
    def forward(self, q, k, v, prev=None, key_padding_mask=None, attn_mask=None):
        attn_scores = torch.matmul(q, k) * self.scale
        if prev is not None: attn_scores = attn_scores + prev
        if attn_mask is not None:                                     
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask
        if key_padding_mask is not None:                      
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)
        attn_weights = F.softmax(attn_scores, dim=-1)           
        attn_weights = self.attn_dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        if self.res_attn: return output, attn_weights, attn_scores
        else: return output, attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attn=True, dropout=0.0):
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v
        self.attn_dropout, self.proj_dropout = dropout, dropout
        self.w_q = nn.Linear(d_model, d_k * n_heads, bias=True)
        self.w_k = nn.Linear(d_model, d_k * n_heads, bias=True)
        self.w_v = nn.Linear(d_model, d_v * n_heads, bias=True)
        self.res_attn= res_attn
        self.sdp_attn = ScaledDotProductAttention(d_model, n_heads, attn_dropout=self.attn_dropout, res_attn=self.res_attn)
        self.proj = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(self.proj_dropout))
    def forward(self, q, k, v, prev=None, key_padding_mask=None, attn_mask=None):
        bs = q.size(0)
        q_s = self.w_q(q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)     
        k_s = self.w_k(k).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)     
        v_s = self.w_v(v).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)       
        if self.res_attn:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        output = self.proj(output)
        # if self.res_attn: return output, attn_weights, attn_scores
        # else: return output, attn_weights
        return output

class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads=8, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
    def forward(self, x):
        identity = x                            
        x = self.attn(x, x, x)    
        x += identity    
        x = self.norm(x)                                                                                              
        return x
      
class MultiHeadCrossAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads=8, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
    def forward(self, x1, x2):
        identity = x1
        x = self.attn(x1, x2, x2)
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
    
class MultiHeadCrossAttentionBackbone(nn.Module):
    def __init__(self, d_model, n_heads=8, dropout=0.0):
        super().__init__()
        self.attn =  MultiHeadCrossAttentionBlock(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.ffn = FFNBlock(d_model=d_model, dropout=dropout)
    def forward(self, x1, x2):
        x = self.attn(x1, x2)
        x = self.ffn(x)
        return x