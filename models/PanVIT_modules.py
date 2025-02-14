import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import repeat
from einops.layers.torch import Rearrange
from torchsummary import summary
import math
from timm.models.vision_transformer import Attention, Mlp
from inspect import isfunction

def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    # 计算填充大小
    padding = (3 - 1) // 2 * dilation
    return nn.Sequential(
        nn.ReplicationPad2d(padding),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, groups=groups, dilation=dilation)
    )

class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = (
            torch.arange(count, dtype=noise_level.dtype, device=noise_level.device)
            / count
        )
        encoding = noise_level.unsqueeze(1) * torch.exp(
            -math.log(1e4) * step.unsqueeze(0)
        )
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding
    
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class CondBlock(nn.Module):
    def __init__(self, cond_dim, hidden_dim, patch_size, img_size, num_chunk, groups=32) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cond_dim, hidden_dim * 8, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_dim * 8),
            nn.SiLU(),
            nn.Conv2d(hidden_dim * 8, hidden_dim * num_chunk, 1, bias=True),
        )
        padding = (patch_size - img_size // patch_size) // 2
        self.global_conv = nn.Conv2d(cond_dim, hidden_dim, patch_size, stride=img_size // patch_size, padding=padding)
        
    def forward(self, cond):
        out = self.body(cond)
        global_token = self.global_conv(cond)
        global_token = global_token.view(cond.size(0), -1)
        return out, global_token
    
class DiTWithCond(nn.Module):
    def __init__(
        self,
        hidden_channels,
        img_size,
        patch_size=16, 
        cond_dim=None,
        noise_level_emb_dim=None,
        time_hidden_ratio=4,    
        norm_groups=32,
        num_heads=16,
        mlp_ratio=4,
    ):
        super().__init__()

        self.cond_inj = CondBlock(
            cond_dim=cond_dim, 
            hidden_dim=hidden_channels, 
            patch_size=patch_size, 
            img_size=img_size, 
            groups=norm_groups,
            num_chunk=6,
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(noise_level_emb_dim, patch_size * patch_size * hidden_channels // time_hidden_ratio),
            nn.SiLU(),
            nn.Linear(patch_size * patch_size * hidden_channels // time_hidden_ratio, patch_size * patch_size * hidden_channels),  
        )
        self.dit = DiTBlock(
            img_size=img_size, 
            hidden_channels=hidden_channels, 
            patch_size=patch_size, 
            num_heads=num_heads, 
            mlp_ratio=mlp_ratio
        )
        
    def forward(self, x, time_emb, cond=None):
        # condition injection
        cond, globaL_token = self.cond_inj(
            F.interpolate(cond, size=x.shape[-2:], mode="bilinear")
        )
        time_token = self.time_mlp(time_emb)
        x = self.dit(x, cond, globaL_token, time_token)
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size=16):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.PositionEncoder = PositionalEncoding((img_size//patch_size) * (img_size//patch_size))
        self.p = nn.Parameter(torch.tensor([0.5]))
        
        self.seg = Rearrange('b c (n1 s1) (n2 s2) -> b (n1 n2) c s1 s2', s1=patch_size, s2=patch_size)
        self.comb = Rearrange('b (n1 n2) c s1 s2 -> b c (n1 s1) (n2 s2)', n1=img_size // patch_size, n2= img_size // patch_size)
        
    def forward(self, noisy):
        noisy_patches = self.seg(noisy)
        position_emb = self.PositionEncoder(self.p)
        position_emb = repeat(position_emb, '1 n -> b n', b=noisy_patches.size(0))
        noisy_patches = noisy_patches + position_emb.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        return self.comb(noisy_patches)
        
def modulate(x, shift, scale):
    return x * (1 + scale) + shift   

class DiTBlock(nn.Module):
    def __init__(self, img_size, hidden_channels, patch_size, num_heads=8, mlp_ratio=4):
        super(DiTBlock, self).__init__()
        hidden_size = hidden_channels * patch_size * patch_size
        self.norm1 = nn.LayerNorm([hidden_channels, img_size, img_size], eps=1e-6, elementwise_affine=False)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0)
        self.norm2 = nn.LayerNorm([hidden_channels, img_size, img_size], eps=1e-6, elementwise_affine=False)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu)

        self.seg = Rearrange('b c (n1 s1) (n2 s2) -> b (n1 n2) (c s1 s2)', s1=patch_size, s2=patch_size)
        self.comb = Rearrange('b (n1 n2) (c s1 s2) -> b c (n1 s1) (n2 s2)', n1=img_size//patch_size, n2=img_size//patch_size, s1=patch_size, s2=patch_size)
        
    def forward(self, x, cond, global_token, time_token):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = cond.chunk(6, dim=1)
        res = self.seg(modulate(self.norm1(x), shift_msa, scale_msa))
        res = torch.cat((global_token.unsqueeze(1), time_token.unsqueeze(1), res), dim=1)
        res = self.attn(res)
        x = x + self.comb(res[:, 2:]) * gate_msa
        x = x + self.comb(self.mlp(self.seg(modulate(self.norm2(x), shift_mlp, scale_mlp)))) * gate_mlp
        return x
    
class FinalLayer(nn.Module):
    def __init__(self, img_size, hidden_channels, cond_channels, patch_size=16, out_channels=3):
        super(FinalLayer, self).__init__()
        hidden_size = hidden_channels * patch_size * patch_size
        self.norm_final = nn.LayerNorm([hidden_channels, img_size, img_size], eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.cond_inj = CondBlock(
            cond_dim=cond_channels, 
            hidden_dim=hidden_channels, 
            patch_size=patch_size, 
            img_size=img_size, 
            groups=32,
            num_chunk=2,  
        )
        
        self.seg = Rearrange('b c (n1 s1) (n2 s2) -> b (n1 n2) (c s1 s2)', s1=patch_size, s2=patch_size)
        self.comb = Rearrange('b (n1 n2) (c s1 s2) -> b c (n1 s1) (n2 s2)', n1=img_size//patch_size, n2=img_size//patch_size, s1=patch_size, s2=patch_size)
        
    def forward(self, x, cond):
        cond, _ = self.cond_inj(cond)
        shift, scale = cond.chunk(2, dim=1)
        x = self.linear(self.seg(modulate(self.norm_final(x), shift, scale)))
        return self.comb(x)

class DiT(nn.Module):
    def __init__(self, 
        img_size, 
        patch_size, 
        hidden_channels, 
        cond_dim, 
        c_dim, 
        noise_level_emb_dim,   
        time_hidden_ratio=4, 
        num_layers=6,
        groups=32,
        num_heads=16, 
        mlp_ratio=4
    ):
        super(DiT, self).__init__()
        self.c_dim = c_dim
        self.hidden_channels = hidden_channels 
        self.num_heads = num_heads 
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i < num_layers // 2:
                self.layers.append(DiTWithCond(
                    hidden_channels=hidden_channels,
                    img_size=img_size,
                    patch_size=patch_size,
                    cond_dim=c_dim,
                    noise_level_emb_dim=noise_level_emb_dim,
                    time_hidden_ratio=time_hidden_ratio,
                    norm_groups=groups,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                ))
            elif i == num_layers // 2:
                self.layers.append(DiTWithCond(
                    hidden_channels=hidden_channels,
                    img_size=img_size,
                    patch_size=patch_size,
                    cond_dim=cond_dim - c_dim,
                    noise_level_emb_dim=noise_level_emb_dim,
                    time_hidden_ratio=time_hidden_ratio,
                    norm_groups=groups,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,   
                ))
            else:
                hidden_channels += self.hidden_channels
                num_heads += self.num_heads
                self.layers.append(DiTWithCond(
                    hidden_channels=hidden_channels,
                    img_size=img_size,
                    patch_size=patch_size,
                    cond_dim=cond_dim - c_dim,
                    noise_level_emb_dim=noise_level_emb_dim,
                    time_hidden_ratio=time_hidden_ratio,
                    norm_groups=groups,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,   
                ))

    def forward(self, x, time_emb, cond):
        feats = []
        for index, layer in enumerate(self.layers):
            if index < len(self.layers) // 2 - 1:
                x = layer(x, time_emb, cond[:, :self.c_dim])
                feats.append(x)
            elif index > len(self.layers) // 2:
                x = torch.cat([x, feats.pop()], dim=1)
                x = layer(x, time_emb, cond[:, self.c_dim:])
            elif index == len(self.layers) // 2 - 1:
                x = layer(x, time_emb, cond[:, :self.c_dim])
            else:
                x = layer(x, time_emb, cond[:, self.c_dim:])
        return x

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

if __name__ == '__main__':
    patch_emb = PatchEmbedding(128).cuda()
    summary(patch_emb, (3, 128, 128))