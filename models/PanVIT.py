import os
import math

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
from PIL import Image
from einops.layers.torch import Rearrange

from models.PanVIT_modules import conv3x3, exists, default, PatchEmbedding, PositionalEncoding, DiT, FinalLayer

py_path = os.path.abspath(__file__) 
file_dir = os.path.dirname(py_path)
    
class PanDiT(nn.Module):
    def __init__(
        self,
        in_channel=3,   
        out_channel=3,
        image_size=128,
        patch_size=16,  
        inner_channel=8,
        noise_level_channel=128,
        lms_channel=8,
        pan_channel=1,
        cond_dim=10,
        time_hidden_ratio=4,
        num_dit_layers=6,   
        norm_groups=32,
        num_heads=16,
        mlp_ratio=4,
        with_noise_level_emb=True,
        self_condition=False,
        pred_var=False,
    ):
        super().__init__()

        self.lms_channel = lms_channel
        self.pan_channel = pan_channel

        if with_noise_level_emb:
            self.noise_level_emb = PositionalEncoding(noise_level_channel)
                # nn.Linear(noise_level_channel, noise_level_channel * 4),
                # Swish(),
                # nn.Linear(inner_channel * 4, inner_channel),
        else:
            noise_level_channel = None
            self.noise_level_emb = None

        if self_condition:
            in_channel += out_channel
        self.pred_var = pred_var    
        
        self.self_condition = self_condition
        self.prev_conv = conv3x3(in_channel, inner_channel)
        self.noise_embedding = PatchEmbedding(img_size=image_size, patch_size=patch_size)
        self.dit = DiT(
            hidden_channels=inner_channel,
            img_size=image_size,
            patch_size=patch_size,
            cond_dim=cond_dim,
            c_dim=lms_channel+pan_channel,
            noise_level_emb_dim=noise_level_channel,
            time_hidden_ratio=time_hidden_ratio, 
            num_layers=num_dit_layers,
            groups=norm_groups,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
        self.final_layer = FinalLayer(
            img_size=image_size, 
            hidden_channels=inner_channel * (num_dit_layers // 2), 
            cond_channels=cond_dim,
            patch_size=patch_size,
            out_channels=out_channel,
        )
        
    def forward(self, x, time, cond=None, self_cond=None):
        # self-conditioning
        if self.self_condition:
            self_cond = default(self_cond, x)
            x = torch.cat([self_cond, x], dim=1)
        x = self.prev_conv(x)
        x = self.noise_embedding(x)
        t = self.noise_level_emb(time) if exists(self.noise_level_emb) else None
        x = self.dit(x, t, cond)
        x = self.final_layer(x, cond)
        return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight) 
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.MultiheadAttention):
                nn.init.xavier_uniform_(m.in_proj_weight)
                if m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)
                nn.init.xavier_uniform_(m.out_proj.weight)
                if m.out_proj.bias is not None:
                    nn.init.zeros_(m.out_proj.bias)

            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)  
                nn.init.zeros_(m.bias)  
                
            elif isinstance(m, nn.Parameter):
                if m.shape[0] > 1:  
                    nn.init.normal_(m, mean=0, std=0.02) 
    