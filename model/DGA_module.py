import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops import rearrange
from einops.layers.torch import Rearrange

from utils import TransformerMLP

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x): # weight:
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class DeformableAtt(nn.Module):
    def __init__(self,  dim,  input_resolution, current_layer, stride=1):
        super().__init__()
        self.input_H, self.input_W = input_resolution[0], input_resolution[1]
        self.nc = dim
        self.n_heads = 4
        self.n_head_channels = dim // self.n_heads
        self.scale = self.n_head_channels ** -0.5
        self.n_groups = 2
        self.n_group_channels = dim // self.n_groups
        self.offset_range_factor = 2
        ksize = 7
        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, ksize, stride, ksize // 2, groups=self.n_group_channels),
            Rearrange('b c h w -> b h w c'),
            nn.LayerNorm(self.n_group_channels),
            Rearrange('b h w c -> b c h w'),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )

        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_out = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key).mul_(2).sub_(1)
        ref[..., 0].div_(H_key).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2
        return ref

    def forward(self, x):
        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device

        q = self.proj_q(x)
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off)  # B * g 2 Hg Wg
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk

        offset_range = torch.tensor([1.0 / Hk, 1.0 / Wk], device=device).reshape(1, 2, 1, 1)
        offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)
        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        pos = offset + reference

        x_sampled = F.grid_sample(
            input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg

        x_sampled = x_sampled.reshape(B, C, 1, n_sample)

        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)

        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, HW, Ns
        attn = attn.mul(self.scale)
        attn = F.softmax(attn, dim=2)

        out = torch.einsum('b m n, b c n -> b c m', attn, v)
        out = out.reshape(B, C, H, W)
        out = self.proj_out(out)
        return out

class DA_attention(nn.Module):
    def __init__(self, imgsize, dim_stem, patch_size, current_layer=1):
        super(DA_attention, self).__init__()
        self.atten = nn.Sequential(
            nn.LayerNorm(dim_stem),
            Attention(dim_stem, heads=6, dim_head=16),
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim_stem),
            TransformerMLP(dim_stem),
        )
        self.Layer_norm = nn.LayerNorm(dim_stem)
        self.DeSpect_K = DeformableAtt(dim=dim_stem, input_resolution=(imgsize//patch_size, imgsize//patch_size),
                                       current_layer=current_layer)
        self.mlp2 = nn.Sequential(
            nn.LayerNorm(dim_stem),
            TransformerMLP(dim_stem),
        )

    def forward(self, x):
        B, C, H, W = x.size()
        x = einops.rearrange(x,'b c h w -> b (h w) c', h=H)
        x = self.atten(x)+x
        x = self.mlp(x)+x
        x0 = x
        x = self.Layer_norm(x)
        x = einops.rearrange(x, 'b (h w) c -> b c h w', h=H)
        x = self.DeSpect_K(x)
        x = einops.rearrange(x,'b c h w -> b (h w) c', h=H) + x0
        x = self.mlp2(x)+x
        x = einops.rearrange(x, 'b (h w) c -> b c h w', h=H)
        return x

class DGA_build(nn.Module):
    def __init__(self, in_channels, img_size, dim_stem = 96, dim_shrink=3, patch_size = 4, DGA_num=2):
        super().__init__()
        assert dim_stem % dim_shrink == 0, 'The dim_stem must be divisible by the dim_shrink.'
        self.patch_proj = nn.Sequential(
            nn.Conv2d(in_channels, dim_stem, patch_size, patch_size)
        )
        self.DGA_layers = nn.ModuleList([])
        for i in range(DGA_num):
            self.DGA_layers.append(DA_attention(img_size, dim_stem, patch_size))
        self.DGA_outconv = nn.Conv2d(dim_stem, int(dim_stem // dim_shrink), kernel_size=1, bias=False)

    def forward(self, x):
        x = self.patch_proj(x)
        for layer in self.DGA_layers:
            x = layer(x)
        x = self.DGA_outconv(x)
        return x