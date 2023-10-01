import torch
import torch.nn as nn
import math
from einops import rearrange
from einops.layers.torch import Rearrange

from model.utils import PreLayerNorm, FeedForward

class FDA_build(nn.Module):
    def __init__(self, in_channels, out_channels, image_size, module_num=2, patch_size=2, mlp_expand=4, heads=6,
                 dim_head=128, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        assert image_size % patch_size == 0, 'The image size must be divisible by the patch size.'
        self.patch_dim = in_channels * patch_size ** 2
        self.out_channels = out_channels
        self.mlp_dim = self.out_channels * mlp_expand

        self.patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(self.patch_dim, self.out_channels),
        )
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = FrequencyAttLayer(self.out_channels, image_size, module_num, heads, dim_head,
                                         self.mlp_dim, dropout)
        self.recover_patch_embedding = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=image_size // patch_size),
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        b, n, _ = x.shape
        x = self.dropout(x)
        ax = self.transformer(x)
        out = self.recover_patch_embedding(ax)
        return out

class FrequencyAttLayer(nn.Module):
    def __init__(self, dim, image_size, module_num, heads, dim_head,
                 mlp_dim=1024, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.module_num = module_num
        for i in range(module_num):
            self.layers.append(PreLayerNorm(dim, nn.Identity()))
            self.layers.append(FrequencyTrans(dim, h=image_size, w=int(image_size//2)+1))
            self.layers.append(PreLayerNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))

            self.layers.append(PreLayerNorm(dim, nn.Identity()))
            self.layers.append(Attention_pure(dim, heads=heads, dim_head=dim_head, dropout=dropout))
            self.layers.append(PreLayerNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))

    def forward(self, x):
        layer_num = 0
        for i in range(self.module_num):
            x1 = self.layers[layer_num](x)
            x1, weight = self.layers[layer_num+1](x1)
            x1 = x1 + x
            x1 = self.layers[layer_num+2](x1) + x1

            x2 = self.layers[layer_num+3](x1)
            x2 = self.layers[layer_num+4](x2, weight)
            x2 = x2 + x1
            x2 = self.layers[layer_num+5](x2) + x2
            layer_num += (i+1)*6
        return x2

class FrequencyTrans(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        a = b = int(math.sqrt(N))
        x = x.view(B, a, b, C)
        x = x.to(torch.float32)

        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')
        x = x.reshape(B, N, C)
        weight = weight.permute(2, 0, 1)
        return x, weight.abs().detach()

class Attention_pure(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., pure_rate=0. , pure=False):
        super().__init__()

        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim), # 768 -> 128
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.FEP = FEP_module(ch_in=dim, pure_rate=pure_rate, pure=pure)

    def forward(self, x, weight): # weight:
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        q = self.FEP(q, weight)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# Frequency excitation and pruning module
class FEP_module(nn.Module):
    def __init__(self, ch_in, reduction=4, pure_rate=0., pure=False):
        super().__init__()
        self.pure_rate=pure_rate
        self.pure = pure
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(ch_in)
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, weight):
        d, h, w = weight.size()
        y = self.avg_pool(weight).view(1,d)
        y = self.norm(y)
        y = self.fc(y) * 2
        if self.pure:
            y[y < self.pure_rate]=0
        return x * y