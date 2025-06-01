import os

import torch
import torch.nn as nn

from models.PanVIT_modules import exists, default, PatchEmbedding, TimeEncoding, DiT, FinalLayer

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
        time_hidden_ratio=4,
        num_dit_layers=6,   
        num_heads=16,
        mlp_ratio=4,
        with_noise_level_emb=True,
        self_condition=False,
    ):
        super().__init__()

        self.lms_channel = lms_channel
        self.pan_channel = pan_channel

        if with_noise_level_emb:
            self.noise_level_emb = TimeEncoding(noise_level_channel)
        else:
            noise_level_channel = None
            self.noise_level_emb = None

        if self_condition:
            in_channel += out_channel
        
        self.self_condition = self_condition
        self.prev_conv = nn.Conv2d(in_channel, inner_channel, 1, 1)
        self.noise_embedding = PatchEmbedding(
            img_size=image_size, 
            patch_size=patch_size,
            inner_channel=inner_channel,
        )
        self.dit = DiT(
            hidden_channels=inner_channel,
            img_size=image_size,
            patch_size=patch_size,
            ms_channels=lms_channel,
            pan_channels=pan_channel,
            noise_level_emb_dim=noise_level_channel,
            time_hidden_ratio=time_hidden_ratio, 
            num_layers=num_dit_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
        self.final_layer = FinalLayer(
            ms_channels=lms_channel,
            pan_channels=pan_channel,
            img_size=image_size, 
            hidden_channels=inner_channel, 
            patch_size=patch_size,
            out_channels=out_channel,
            num_heads=num_heads,
        )
        
    def forward(self, x, time, cond, self_cond=None):
        # self-conditioning
        if self.self_condition:
            self_cond = default(self_cond, x)
            x = torch.cat([x, self_cond], dim=1)
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
                if m.weight is not None:
                    nn.init.ones_(m.weight)  
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                
            elif isinstance(m, nn.Parameter):
                if m.shape[0] > 1:  
                    nn.init.normal_(m, mean=0, std=0.02) 
    