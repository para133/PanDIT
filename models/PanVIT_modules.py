import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.fft
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

class TimeEncoding(nn.Module):
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
    
class PositionalEncoding(nn.Module):
    def __init__(self, num_patch, hidden_size):
        super().__init__()
        self.num_patch = num_patch
        self.hidden_size = hidden_size
        
    def forward(self, level):
        position = torch.arange(self.num_patch, dtype=torch.float32, device=level.device).unsqueeze(1)  # (N, 1)
        div_term = torch.exp(torch.arange(0, self.hidden_size, 2, dtype=torch.float32, device=level.device) * (-math.log(10000.0) / self.hidden_size))
        pe = torch.zeros(self.num_patch, self.hidden_size, device=level.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) * level

        return pe

        
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class DiscrepancyCommonInjctionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, img_size, patch_size, num_heads, mode):
        super().__init__()
        self.mode = mode
        self.img_size = img_size    
        self.num_heads = num_heads
        hidden_dim = in_channels * patch_size * patch_size
        self.postion_encoder = PositionalEncoding(
            num_patch=(img_size//patch_size) * (img_size//patch_size),
            hidden_size=patch_size * patch_size * in_channels,
        )
        self.p = nn.Parameter(torch.tensor([1.0]))
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.LayerNorm([out_channels, img_size, img_size], eps=1e-6, elementwise_affine=False),
            Swish(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )
        self.depth = hidden_dim // num_heads
        self.seg = Rearrange('b c (n1 s1) (n2 s2) -> b (n1 n2) (c s1 s2)', s1=patch_size, s2=patch_size)
        self.comb = Rearrange('b (n1 n2) (c s1 s2) -> b c (n1 s1) (n2 s2)', n1=img_size // patch_size, n2= img_size // patch_size, s1=patch_size, s2=patch_size)
    
    def forward(self, k, q, v):
        b, N, e = k.shape
        scale = e ** -0.5
        position_emb = self.postion_encoder(self.p)
        position_emb = repeat(position_emb, '1 n e -> b n e', b=b)
        k, q, v = k+position_emb, q+position_emb, v+position_emb
        q = q.view(b, -1, self.num_heads, self.depth).transpose(1, 2)  
        k = k.view(b, -1, self.num_heads, self.depth).transpose(1, 2)  
        v = v.view(b, -1, self.num_heads, self.depth).transpose(1, 2)
        t = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, scale=scale)
        t = t.transpose(1, 2).reshape(b, N, e)
        v = v.transpose(1, 2).reshape(b, N, e) 
        q = q.transpose(1, 2).reshape(b, N, e)
        if self.mode == 'discrepancy':
            v = v - t
        else:
            v = t
        q = self.linear(v) + q
        q = self.comb(q)
        q = self.body(q)
        
        return self.seg(q)    
       
class ModalAttentionFusion(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, img_size, num_heads):
        super().__init__()
        self.diim = DiscrepancyCommonInjctionBlock(
            in_channels=in_channels, 
            out_channels=out_channels, 
            img_size=img_size, 
            patch_size=patch_size, 
            num_heads=num_heads, 
            mode='discrepancy'
        )      
        self.aciim1 = DiscrepancyCommonInjctionBlock(
            in_channels=in_channels, 
            out_channels=out_channels, 
            img_size=img_size, 
            patch_size=patch_size, 
            num_heads=num_heads, 
            mode='common'
        )
        self.aciim2 = DiscrepancyCommonInjctionBlock(
            in_channels=in_channels, 
            out_channels=out_channels, 
            img_size=img_size, 
            patch_size=patch_size, 
            num_heads=num_heads, 
            mode='common'
        )
        self.reg = Rearrange('b c (n1 s1) (n2 s2) -> b (n1 n2) (c s1 s2)', s1=patch_size, s2=patch_size)
        self.comb = Rearrange('b (n1 n2) (c s1 s2) -> b c (n1 s1) (n2 s2)', n1=img_size // patch_size, n2= img_size // patch_size, s1=patch_size, s2=patch_size)
    
    def forward(self, ms, pan):
        pan_k, pan_q, pan_v = pan.chunk(3, dim=1)
        pan_k, pan_q, pan_v = self.reg(pan_k), self.reg(pan_q), self.reg(pan_v)
        ms_k, ms_q, ms_v = ms.chunk(3, dim=1)
        ms_k, ms_q, ms_v = self.reg(ms_k), self.reg(ms_q), self.reg(ms_v)
        q1 = self.diim(pan_k, pan_v, ms_q)
        q2 = self.aciim1(ms_k, ms_v, q1)
        q2 = self.aciim2(pan_k, pan_v, q2)
        
        return self.comb(q1 + q2)
       
class CondBlock(nn.Module):
    def __init__(self, cond_dim, hidden_dim, patch_size, img_size, num_chunk, groups) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cond_dim, hidden_dim * 8, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_dim * 8),
            nn.SiLU(),
            nn.Conv2d(hidden_dim * 8, hidden_dim * num_chunk, 1, bias=True),
        )
        cond_size = cond_dim * patch_size * patch_size
        hidden_size = hidden_dim * patch_size * patch_size
        self.global_linear = nn.Linear(cond_size, hidden_size)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.reg = Rearrange('b c (n1 s1) (n2 s2) -> b (n1 n2) (c s1 s2)', s1=patch_size, s2=patch_size)
        
        
    def forward(self, cond):
        out = self.body(cond)
        global_token = self.global_linear(self.reg(cond)).mean(dim=1)
        global_token = self.mlp(global_token)
        
        return out, global_token
    
class ConvLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, img_size, need_residual=True):
        super().__init__()
        self.body = nn.Sequential(
            nn.ReplicationPad2d(kernel_size // 2),
            nn.Conv2d(in_channels, hidden_channels, kernel_size),
            nn.LayerNorm([hidden_channels, img_size, img_size], eps=1e-6, elementwise_affine=False),
            nn.SiLU(),
            conv3x3(hidden_channels, out_channels)
        )
        if need_residual:
            self.res_conv = nn.Conv2d(in_channels, out_channels, 1) 
    
    def forward(self, x):
        return self.body(x) + self.res_conv(x) if hasattr(self, 'res_conv') else self.body(x)
        
class WaveletBlock(nn.Module):
    def __init__(self, ms_channels, pan_channels, hidden_dim, img_size, patch_size, num_heads=16, num_chunks=8):
        super().__init__()
        self.ms_channels = ms_channels
        self.ms_conv = ConvLayer(
            in_channels=ms_channels*2, 
            hidden_channels=hidden_dim*4, 
            out_channels=hidden_dim*6,
            kernel_size=3,
            img_size=img_size, 
            need_residual=True
        )
        self.pan_channels = pan_channels
        self.pan_scale_conv = nn.Sequential(
            conv3x3(pan_channels, hidden_dim*4),
            nn.LeakyReLU(0.1),
            conv3x3(hidden_dim*4, hidden_dim*6),
        )
        self.pan_shift_conv = nn.Sequential(
            conv3x3(pan_channels, hidden_dim*4),
            nn.LeakyReLU(0.1),
            conv3x3(hidden_dim*4, hidden_dim*6),
        )
        self.pan_body = ConvLayer(
            in_channels=pan_channels*3, 
            hidden_channels=hidden_dim*4,
            out_channels=hidden_dim*6, 
            kernel_size=3, 
            img_size=img_size, 
            need_residual=True
        )
        self.fusion_net = ModalAttentionFusion(
            in_channels=hidden_dim*2, 
            out_channels=hidden_dim*2, 
            patch_size=patch_size, 
            img_size=img_size, 
            num_heads=num_heads*2,
        )
        self.cond_inj = CondBlock(
            cond_dim=hidden_dim*2, 
            hidden_dim=hidden_dim, 
            patch_size=patch_size, 
            img_size=img_size, 
            num_chunk=num_chunks, 
            groups=hidden_dim,
        )   
        
        self.reg = Rearrange('b c (n1 s1) (n2 s2) -> b (n1 n2) (c s1 s2)', s1=patch_size, s2=patch_size) 
        
    def forward(self, cond):
        ms = torch.cat([cond[:, :self.ms_channels], cond[:, self.ms_channels+self.pan_channels:self.ms_channels*2+self.pan_channels]], dim=1)
        ms = self.ms_conv(ms)
        pan = cond[:, self.ms_channels:self.ms_channels+self.pan_channels]
        pan_wavelet = cond[:, self.ms_channels*2+self.pan_channels:]
        pan_scale, pan_shift = self.pan_scale_conv(pan), self.pan_shift_conv(pan)
        pan = modulate(self.pan_body(pan_wavelet), pan_shift, pan_scale)
        fusion = self.fusion_net(ms, pan)
        out, global_token = self.cond_inj(fusion)
        
        return out, global_token
        
class FFTBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, img_size, min_ratio, max_ratio, need_residual=False): 
        super().__init__()
        self.body = ConvLayer(
            in_channels=in_channels, 
            hidden_channels=out_channels//2,
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            img_size=img_size, 
            need_residual=need_residual
        )
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        
    def fourier_transform(self, img, ratio):
        fft_image = torch.fft.fft2(img)
        fft_shift = torch.fft.fftshift(fft_image, dim=(-2, -1))

        B, C, H, W = img.shape
        mask = self.generate_mask(H, W, ratio, img.device)
        fft_shift = fft_shift * mask  # 应用低频抑制
        fft_out = torch.cat([fft_shift.real, fft_shift.imag], dim=1)  
        return fft_out

    def generate_mask(self, H, W, raio, device):
        y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device))

        center_y, center_x = H // 2, W // 2
        dist = (x - center_x) ** 2 + (y - center_y) ** 2 + 1e-6

        # 计算可导 Mask
        ratio = torch.clamp(raio, self.min_ratio, self.max_ratio)
        sigma = H * ratio 
        mask = 1 - torch.exp(- dist / (2 * sigma ** 2)) 

        return mask

    def inverse_fourier_transform(self, fft_out):
        B, C2, H, W = fft_out.shape
        C = C2 // 2  
        fft_shift = torch.complex(fft_out[:, :C], fft_out[:, C:])  
        fft_ishift = torch.fft.ifftshift(fft_shift, dim=(-2, -1))
        img_reconstructed = torch.fft.ifft2(fft_ishift).real 
        return img_reconstructed

    def forward(self, x, raio):
        x = self.fourier_transform(x, raio)
        x = self.inverse_fourier_transform(x)
        x = self.body(x)
        return x
    
class AdaptiveFourierTransformBlock(nn.Module):
    def __init__(self, ms_channels, pan_channels, hidden_dim, img_size, patch_size, num_heads=16, num_chunks=8): 
        super().__init__()
        self.ms_channels = ms_channels
        self.pan_channels = pan_channels    
        self.ms_fft = FFTBlock(ms_channels, hidden_dim*6, 5, img_size=img_size, min_ratio=1e-6, max_ratio=0.05)
        self.ms_res_conv = nn.Conv2d(ms_channels, hidden_dim*6, 1)
        self.pan_scale_conv = nn.Sequential(
            conv3x3(pan_channels, hidden_dim*2),
            nn.LeakyReLU(0.1),
            conv3x3(hidden_dim*2, hidden_dim*2),
        )
        self.pan_shift_conv = nn.Sequential(
            conv3x3(pan_channels, hidden_dim*2),
            nn.LeakyReLU(0.1),
            conv3x3(hidden_dim*2, hidden_dim*2),
        )
        self.pan_low_fft = FFTBlock(pan_channels, hidden_dim*2, 7, img_size, min_ratio=1e-6, max_ratio=0.05)
        self.pan_mid_fft = FFTBlock(pan_channels, hidden_dim*2, 5, img_size, min_ratio=0.05, max_ratio=0.15)
        self.pan_high_fft = FFTBlock(pan_channels, hidden_dim*2, 3, img_size, min_ratio=0.15, max_ratio=0.25)
        self.pan_conv = ConvLayer(
            in_channels=hidden_dim*6, 
            hidden_channels=hidden_dim*4,
            out_channels= hidden_dim*6,
            kernel_size=3, 
            img_size=img_size, 
            need_residual=True
        )
        self.fusion_net = ModalAttentionFusion(
            in_channels=hidden_dim*2, 
            out_channels=hidden_dim*2, 
            patch_size=patch_size, 
            img_size=img_size, 
            num_heads=num_heads*2,
        )
        self.cond_inj = CondBlock(
            cond_dim=hidden_dim*2, 
            hidden_dim=hidden_dim, 
            patch_size=patch_size,
            img_size=img_size, 
            num_chunk=num_chunks, 
            groups=hidden_dim,
        )

        self.ms_ratio = nn.Parameter(torch.tensor(0.025))
        self.low_ratio = nn.Parameter(torch.tensor(0.025))
        self.mid_ratio = nn.Parameter(torch.tensor(0.1))
        self.high_ratio = nn.Parameter(torch.tensor(0.2))

        self.reg = Rearrange('b c (n1 s1) (n2 s2) -> b (n1 n2) (c s1 s2)', s1=patch_size, s2=patch_size) 

    def forward(self, cond):
        ms = cond[:, :self.ms_channels]
        ms = self.ms_res_conv(ms) + self.ms_fft(ms, self.ms_ratio)
        pan = cond[:, self.ms_channels:self.ms_channels+self.pan_channels]
        pan_shift, pan_scale = self.pan_scale_conv(pan), self.pan_shift_conv(pan)
        pan_low = self.pan_low_fft(pan, self.low_ratio)
        pan_low = modulate(pan_low, pan_shift, pan_scale)
        pan_mid = self.pan_mid_fft(pan, self.mid_ratio)
        pan_mid = modulate(pan_mid, pan_shift, pan_scale)
        pan_high = self.pan_high_fft(pan, self.high_ratio)
        pan_high = modulate(pan_high, pan_shift, pan_scale)
        pan = self.pan_conv(torch.cat([pan_low, pan_mid, pan_high], dim=1))
        fusion = self.fusion_net(ms, pan)
        out, global_token = self.cond_inj(fusion)
        return out, global_token
    
class PyramidalFuseBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.ms_conv = nn.Sequential(
            conv3x3(in_channels, in_channels*2),
            nn.LeakyReLU(0.1),
            conv3x3(in_channels*2, in_channels*2),
        )
        self.pan_conv = nn.Sequential(
            conv3x3(in_channels, in_channels*2),
            nn.LeakyReLU(0.1),
            conv3x3(in_channels*2, in_channels*2),
        )
    
    def forward(self, ms, pan):
        ms_scale, ms_shift = self.ms_conv(ms).chunk(2, dim=1)
        pan_scale, pan_shift = self.pan_conv(pan).chunk(2, dim=1)
        ms = modulate(ms, ms_shift, ms_scale)
        pan = modulate(pan, pan_shift, pan_scale)
        return ms, pan
        
class PyramidalUpConvs(nn.Module):
    def __init__(self, in_channels, n_levels, img_size):
        super().__init__()
        self.up_convs = nn.ModuleList(
            [nn.Sequential(
                nn.ConvTranspose2d(in_channels // (2 ** i), in_channels // (2 ** (i+1)), 4, 2, 1),
                nn.LeakyReLU(0.1),
                nn.LayerNorm([in_channels // (2 ** (i+1)), img_size // (2 ** (n_levels-1-i)), img_size // (2 ** (n_levels-1-i))], eps=1e-6, elementwise_affine=False),
            ) for i in range(n_levels)]
        )
    
    def forward(self, x):
        for up_conv in self.up_convs:
            x = up_conv(x)
        return x
            
class PyramidalSpatialBlock(nn.Module):
    def __init__(self, ms_channels, pan_channels, hidden_dim, img_size, patch_size, num_heads=16, num_chunks=8):
        super().__init__()
        self.ms_channels = ms_channels
        self.pan_channels = pan_channels
        self.ms_conv = ConvLayer(
            in_channels=ms_channels,
            hidden_channels=hidden_dim*2,
            out_channels=hidden_dim*2,
            kernel_size=3,
            img_size=img_size,
            need_residual=False,
        )
        self.pan_conv = ConvLayer(
            in_channels=pan_channels,
            hidden_channels=hidden_dim*2,
            out_channels=hidden_dim*2,
            kernel_size=3,
            img_size=img_size,
            need_residual=False,
        )
        self.ms_pan_fuse = PyramidalFuseBlock(hidden_dim*2)
        self.ms_dw_convs = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(hidden_dim * (2 ** (i+1)), hidden_dim * (2 ** (i+2)), 3, 2, 1),
                nn.LeakyReLU(0.1),
                nn.LayerNorm([hidden_dim * (2 ** (i+2)), img_size // (2 ** (i+1)), img_size // (2 ** (i+1))], eps=1e-6, elementwise_affine=False),
            ) for i in range(2)]
        )
        self.pan_dw_convs = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(hidden_dim * (2 ** (i+1)), hidden_dim * (2 ** (i+2)), 3, 2, 1),
                nn.LeakyReLU(0.1),
                nn.LayerNorm([hidden_dim * (2 ** (i+2)), img_size // (2 ** (i+1)), img_size // (2 ** (i+1))], eps=1e-6, elementwise_affine=False),
            ) for i in range(2)]
        )
        self.fuse_nets = nn.ModuleList(
            [PyramidalFuseBlock(hidden_dim * (2 ** (i+2))) for i in range(2)]
        )
        self.ms_up_convs = nn.ModuleList(
            [PyramidalUpConvs(hidden_dim * (2 ** (i+2)), i+1, img_size) for i in range(2)]
        )
        self.pan_up_convs = nn.ModuleList(
            [PyramidalUpConvs(hidden_dim * (2 ** (i+2)), i+1, img_size) for i in range(2)]
        )
        self.ms_refine_conv = nn.Conv2d(hidden_dim * 6, hidden_dim * 6, 1)
        self.pan_refine_conv = nn.Conv2d(hidden_dim * 6, hidden_dim * 6, 1)
        self.fusion_net = ModalAttentionFusion(
            in_channels=hidden_dim*2, 
            out_channels=hidden_dim*2, 
            patch_size=patch_size, 
            img_size=img_size, 
            num_heads=num_heads*2,
        )
        self.cond_inj = CondBlock(
            cond_dim=hidden_dim*2, 
            hidden_dim=hidden_dim, 
            patch_size=patch_size,
            img_size=img_size, 
            num_chunk=num_chunks, 
            groups=hidden_dim,
        )
        
        self.reg = Rearrange('b c (n1 s1) (n2 s2) -> b (n1 n2) (c s1 s2)', s1=patch_size, s2=patch_size) 
        
    def forward(self, cond):
        ms = [self.ms_conv(cond[:, :self.ms_channels])]
        pan = [self.pan_conv(cond[:, self.ms_channels:self.ms_channels+self.pan_channels])]
        for i in range(2):
            ms.append(self.ms_dw_convs[i](ms[i]))
            pan.append(self.pan_dw_convs[i](pan[i]))
        ms[0], pan[0] = self.ms_pan_fuse(ms[0], pan[0])
        for i in range(2):
            ms[i+1], pan[i+1] = self.fuse_nets[i](ms[i+1], pan[i+1]) 
            ms[i+1] = self.ms_up_convs[i](ms[i+1])
            pan[i+1] = self.pan_up_convs[i](pan[i+1])
        ms = torch.cat(ms, dim=1)
        pan = torch.cat(pan, dim=1)
        ms = self.ms_refine_conv(ms)
        pan = self.pan_refine_conv(pan)
        fusion = self.fusion_net(ms, pan)
        out, global_token = self.cond_inj(fusion)
        return out, global_token
        
class DiTWithCond(nn.Module):
    def __init__(
        self,
        hidden_channels,
        img_size,
        mode,
        ms_channels,
        pan_channels,
        patch_size=16, 
        noise_level_emb_dim=None,
        time_hidden_ratio=4,    
        num_heads=16,
        mlp_ratio=4,
    ):
        super().__init__()
        if mode == 'WAVE':
            self.cond_inj = WaveletBlock(
                ms_channels=ms_channels,
                pan_channels=pan_channels,
                hidden_dim=hidden_channels,
                img_size=img_size,
                patch_size=patch_size,
                num_chunks=8,
                num_heads=num_heads,
            )
        elif mode == 'FFT':
            self.cond_inj = AdaptiveFourierTransformBlock(
                ms_channels=ms_channels,
                pan_channels=pan_channels,
                hidden_dim=hidden_channels,
                img_size=img_size,
                patch_size=patch_size,
                num_heads=num_heads,
                num_chunks=8,
            )
        elif mode == 'PYRAMID':
            self.cond_inj = PyramidalSpatialBlock(
                ms_channels=ms_channels,
                pan_channels=pan_channels,
                hidden_dim=hidden_channels,
                img_size=img_size,
                patch_size=patch_size,
                num_heads=num_heads,
                num_chunks=8,
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
        
    def forward(self, x, time_emb, cond):
        # condition injection
        cond, globaL_token = self.cond_inj(cond)
        time_token = self.time_mlp(time_emb)
        x = self.dit(x, cond, globaL_token, time_token)
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, inner_channel, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.PositionEncoder = PositionalEncoding(
            num_patch=(img_size//patch_size) * (img_size//patch_size),
            hidden_size=patch_size * patch_size * inner_channel,
        )
        self.p = nn.Parameter(torch.tensor([1.0]))
        
        self.reg = Rearrange('b N (c s1 s2) -> b N c s1 s2', s1=patch_size, s2=patch_size)
        self.seg = Rearrange('b c (n1 s1) (n2 s2) -> b (n1 n2) c s1 s2', s1=patch_size, s2=patch_size)
        self.comb = Rearrange('b (n1 n2) c s1 s2 -> b c (n1 s1) (n2 s2)', n1=img_size // patch_size, n2= img_size // patch_size)
        
    def forward(self, noisy):
        noisy_patches = self.seg(noisy)
        position_emb = self.PositionEncoder(self.p)
        position_emb = repeat(position_emb, '1 n e -> b n e', b=noisy_patches.size(0))
        position_emb = self.reg(position_emb)
        noisy_patches = noisy_patches + position_emb
        return self.comb(noisy_patches)
        
def modulate(x, shift, scale):
    return x * (1 + scale) + shift   


class Cross_MultiAttention(nn.Module):
    def __init__(self, hidden_channels, num_heads, patch_size):
        super(Cross_MultiAttention, self).__init__()
        hidden_size = hidden_channels * patch_size * patch_size
        self.num_heads = num_heads
        self.scale = hidden_size ** -0.5
 
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.depth = hidden_size // num_heads
  
        self.kv_conv = conv3x3(hidden_channels, hidden_channels*2)
        self.q_linear = nn.Linear(hidden_size, hidden_size)
 
        self.reg = Rearrange('b c (n1 s1) (n2 s2) -> b (n1 n2) (c s1 s2)', s1=patch_size, s2=patch_size) 
        
    def forward(self, x, cond):
        b, c, h, w = x.shape

        x = self.reg(x)
        q = self.q_linear(x)        
        k, v = self.reg(self.kv_conv(cond)).chunk(2, dim=2)
 
        q = q.view(b, -1, self.num_heads, self.depth).transpose(1, 2)  
        k = k.view(b, -1, self.num_heads, self.depth).transpose(1, 2)  
        v = v.view(b, -1, self.num_heads, self.depth).transpose(1, 2)
 
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, scale=self.scale)
        x = x.transpose(1, 2).reshape(b, -1, h, w)
        return x
        
class DiTBlock(nn.Module):
    def __init__(self, img_size, hidden_channels, patch_size, num_heads=8, mlp_ratio=4):
        super(DiTBlock, self).__init__()
        hidden_size = hidden_channels * patch_size * patch_size
        self.norm1 = nn.LayerNorm([hidden_channels, img_size, img_size], eps=1e-6, elementwise_affine=False)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0)
        self.norm2 = nn.LayerNorm([hidden_channels, img_size, img_size], eps=1e-6, elementwise_affine=False)
        self.cross_attn = Cross_MultiAttention(hidden_channels, num_heads, patch_size)    
        self.norm3 = nn.LayerNorm([hidden_channels, img_size, img_size], eps=1e-6, elementwise_affine=False)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu)

        self.seg = Rearrange('b c (n1 s1) (n2 s2) -> b (n1 n2) (c s1 s2)', s1=patch_size, s2=patch_size)
        self.comb = Rearrange('b (n1 n2) (c s1 s2) -> b c (n1 s1) (n2 s2)', n1=img_size//patch_size, n2=img_size//patch_size, s1=patch_size, s2=patch_size)
        
    def forward(self, x, cond, global_token, time_token):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, cond_feat, gate_mca = cond.chunk(8, dim=1)
        res = self.seg(modulate(self.norm1(x), shift_msa, scale_msa))
        res = torch.cat((global_token.unsqueeze(1), time_token.unsqueeze(1), res), dim=1)
        res = self.attn(res)
        x = x + self.comb(res[:, 2:]) * gate_msa
        x = x + self.cross_attn(self.norm2(x), cond_feat) * gate_mca
        x = x + self.comb(self.mlp(self.seg(modulate(self.norm3(x), shift_mlp, scale_mlp)))) * gate_mlp
        return x
    

    
class FinalLayer(nn.Module):
    def __init__(self, ms_channels, pan_channels, img_size, hidden_channels, patch_size=16, out_channels=3, num_heads=16):
        super(FinalLayer, self).__init__()
        self.ms_channels = ms_channels
        self.pan_channels = pan_channels
        self.ms_conv = ConvLayer(
            in_channels=ms_channels*2,
            hidden_channels= hidden_channels*4,
            out_channels=hidden_channels*6, 
            kernel_size=3, 
            img_size=img_size, 
            need_residual=True,
        )
        self.pan_conv = ConvLayer(
            in_channels=pan_channels*4, 
            hidden_channels=hidden_channels*4, 
            out_channels=hidden_channels*6,
            kernel_size=3, 
            img_size=img_size, 
            need_residual=True,
        ) 
        hidden_size = hidden_channels * patch_size * patch_size
        self.norm_final = nn.LayerNorm([hidden_channels, img_size, img_size], eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.fusion_net = ModalAttentionFusion(
            in_channels=hidden_channels*2, 
            out_channels=hidden_channels*2, 
            patch_size=patch_size, 
            img_size=img_size, 
            num_heads=num_heads*2,
        )
        self.cond_inj = CondBlock(
            cond_dim=hidden_channels*2, 
            hidden_dim=hidden_channels, 
            patch_size=patch_size, 
            img_size=img_size, 
            num_chunk=2,  
            groups=hidden_channels,
        )
        self.seg = Rearrange('b c (n1 s1) (n2 s2) -> b (n1 n2) (c s1 s2)', s1=patch_size, s2=patch_size)
        self.comb = Rearrange('b (n1 n2) (c s1 s2) -> b c (n1 s1) (n2 s2)', n1=img_size//patch_size, n2=img_size//patch_size, s1=patch_size, s2=patch_size)
        
    def forward(self, x, cond):
        ms = torch.cat([cond[:, :self.ms_channels], cond[:, self.ms_channels+self.pan_channels:self.ms_channels*2+self.pan_channels]], dim=1)
        pan = torch.cat([cond[:, self.ms_channels:self.ms_channels+self.pan_channels], cond[:, self.ms_channels*2+self.pan_channels:]], dim=1)
        ms = self.ms_conv(ms)
        pan = self.pan_conv(pan)
        fusion = self.fusion_net(ms, pan)
        cond, _ = self.cond_inj(fusion)
        shift, scale = cond.chunk(2, dim=1)
        x = self.linear(self.seg(modulate(self.norm_final(x), shift, scale)))
        return self.comb(x)

class DiT(nn.Module):
    def __init__(self, 
        img_size, 
        patch_size, 
        hidden_channels, 
        ms_channels,
        pan_channels, 
        noise_level_emb_dim,   
        time_hidden_ratio=4, 
        num_layers=6,
        num_heads=16, 
        mlp_ratio=4
    ):
        super(DiT, self).__init__()
        self.hidden_channels = hidden_channels 
        self.num_heads = num_heads 
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i < num_layers // 2:
                self.layers.append(DiTWithCond(
                    hidden_channels=hidden_channels,
                    img_size=img_size,
                    patch_size=patch_size,
                    mode='PYRAMID',
                    ms_channels=ms_channels,
                    pan_channels=pan_channels,                   
                    noise_level_emb_dim=noise_level_emb_dim,
                    time_hidden_ratio=time_hidden_ratio,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                ))
            else:
                self.layers.append(DiTWithCond(
                    hidden_channels=hidden_channels,
                    img_size=img_size,
                    patch_size=patch_size,
                    mode='WAVE',
                    ms_channels=ms_channels,
                    pan_channels=pan_channels,
                    noise_level_emb_dim=noise_level_emb_dim,
                    time_hidden_ratio=time_hidden_ratio,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,   
                ))
                
    def forward(self, x, time_emb, cond):
        for index, layer in enumerate(self.layers):
            x = layer(x, time_emb, cond)
        return x

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

if __name__ == '__main__':
    block = PyramidalSpatialBlock(
        ms_channels=4,
        pan_channels=1,
        hidden_dim=16,
        img_size=128,
        patch_size=8,
        num_heads=16,
        num_chunks=8,
    )
    test = torch.randn(2, 10, 128, 128)
    out, token = block(test)
    print(out.shape, token.shape)