import os

import torch
from torch.utils.data import DataLoader
import einops
import torch.nn.functional as F
import numpy as np
from PIL import Image

from diffusion.diffusion_ddpm_pan import make_beta_schedule
from models.sr3_dwt import UNetSR3 as Unet
from diffusion.diffusion_ddpm_pan import GaussianDiffusion
from utils.misc import model_load
from data.PanDataset import PanDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

@torch.no_grad()
def predict(
    dataset_folder,
    save_dir,
    weight_path,
    image_size,
    ms_num_channel = 3,
    pan_num_channel = 1,
    schedule_type="cosine",
    batch_size=1,
    n_steps=500,
):
    denoise_fn = Unet(
        in_channel=ms_num_channel,
        out_channel=ms_num_channel,
        lms_channel=ms_num_channel,
        pan_channel=pan_num_channel,
        inner_channel=32,
        norm_groups=1,
        channel_mults=(1,2,2,4),#(1, 2, 2, 4),  # (64, 32, 16, 8)
        attn_res=(8,),
        dropout=0.2,
        image_size=64,
        self_condition=True,
    ).to(device)

    denoise_fn = model_load(weight_path, denoise_fn, device=device)
    denoise_fn.eval()
    print(f"load weight {weight_path}")
    
    diffusion = GaussianDiffusion(
        denoise_fn,
        image_size=image_size,
        channels=ms_num_channel,
        pred_mode="x_start",
        loss_type="l1",
        device=device,
        clamp_range=(0, 1),
    )
    diffusion.set_new_noise_schedule(
        betas=make_beta_schedule(schedule=schedule_type, n_timestep=n_steps, cosine_s=8e-3)
    )
    diffusion = diffusion.to(device)

    dataset = PanDataset(
        ms_folder=os.path.join(dataset_folder, 'ms_LR'),
        pan_folder=os.path.join(dataset_folder, 'pan'),
        GT_folder=os.path.join(dataset_folder, 'ms_HR'),
    )
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    
    for i, batch in enumerate(loader):
        pan, lms, _, wavelets = map(lambda x: x.cuda(), (batch['img_pan'], batch['img_ms_up'], batch['GT'], batch['wavelets']))
        cond, _ = einops.pack(
            [lms, pan, F.interpolate(wavelets, size=lms.shape[-1], mode="bilinear")],
            "b * h w",
        )
        cond = cond.to(torch.float32)
        sr = diffusion(cond, mode="ddim_sample", section_counts="ddim25")
        sr = sr + lms
        sr = sr.clip(0, 1)
        sr = sr.detach().cpu().numpy()
        sr = (sr * 255).astype(np.uint8)
        for j in range(sr.shape[0]):
            img = np.transpose(sr[j], (1, 2, 0))  # HWC 
            img = Image.fromarray(img) 
            img.save(os.path.join(save_dir, f'{batch["file_name"][j]}'))         
            
if __name__ == '__main__':
    py_path = os.path.abspath(__file__) 
    file_dir = os.path.dirname(py_path)
    dataset_folder = os.path.join(os.path.dirname(file_dir), 'PanDataset', 'WV2_data', 'test128')
    save_dir = os.path.join(file_dir, 'output')
    weight_path = os.path.join(file_dir, 'checkpoints', 'diffusion_iter_100000.pth')
    predict(
        dataset_folder=dataset_folder,
        save_dir=save_dir,
        weight_path=weight_path,
        image_size=128,
        ms_num_channel = 3,
        pan_num_channel = 1,
        schedule_type="cosine",
        batch_size=1,
        n_steps=500,
    )